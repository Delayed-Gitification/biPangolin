"""
bench_correlations.py — streamed Spearman correlation matrices across all score columns.

Why Spearman: the probe's softmax outputs are highly saturated near 1 at true
sites, which crushes Pearson r without changing the model's actual ranking.
Spearman is rank-based and far less sensitive to that saturation (and to the
choice of Bayesian correction k).

Designed to fit in laptop memory by streaming the parquet:
    pass 1 — build per-(subset, column) histograms over 2^12 bins in [0, 1]
    pass 2 — re-stream, look up each value's rank percentile via the histograms,
             accumulate Pearson moments on those rank-percentiles.

Spearman correlation = Pearson correlation of rank-percentiles.

Total memory peak per batch ≈ a few hundred MB. Total IO ≈ 2× parquet size.

Columns analysed
================
Native (auto-detected from parquet schema):
    probe_none, probe_acc, probe_don                                 — ensemble probe
    pangolin_p_{tissues}, pangolin_psi_{tissues}                     — per tissue
    spliceai_acc, spliceai_don
    probe_none_{tissues}, probe_acc_{tissues}, probe_don_{tissues}   — per-tissue probe (if present)
    probe_none_psi_{tissues}, probe_acc_psi_{tissues}, probe_don_psi_{tissues}
                                                                     — PSI-side per-tissue probe (if present)

Derived:
    spliceai_max            = max(spliceai_acc, spliceai_don)
    probe_max               = max(probe_acc, probe_don)                       (ensemble)
    probe_max_{tissue}      = max(probe_acc_{tissue}, probe_don_{tissue})     (if per-tissue cols present)
    probe_max_psi_{tissue}  = max(probe_acc_psi_{tissue}, probe_don_psi_{tissue}) (if PSI-side cols present)
    pangolin_p_ensemble     = mean over tissues of pangolin_p
    pangolin_psi_ensemble   = mean over tissues of pangolin_psi

Subsets: all, acceptors, donors, nonsites.
NaN policy: rows with NaN in any column are dropped (consistent denominator).

Output: an .npz with
    columns                       (U60)  — column names
    corr_{subset}                 (K, K) — Spearman ρ
    n_{subset}                    int    — rows contributing

Typical usage:
    python benchmark/bench_correlations.py \\
        --scores bench_scores/ \\
        --out    bench_metrics/correlations.npz
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2
TISSUE_ORDER = ("heart", "liver", "brain", "testis")
SUBSETS = ("all", "acceptors", "donors", "nonsites")
N_BINS_DEFAULT = 4096


def _log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def detect_columns(schema_names):
    """Inspect a parquet schema and return:
      native_cols: list of physical columns to read.
      derived_specs: list of (name, fn(cols_dict) -> ndarray) to compute.
      The combined order (native + derived) defines the correlation matrix index.
    """
    available = set(schema_names)
    native = []

    # Ensemble probe (always there)
    for c in ("probe_none", "probe_acc", "probe_don"):
        if c in available:
            native.append(c)

    # Per-tissue Pangolin P and PSI
    for t in TISSUE_ORDER:
        c = f"pangolin_p_{t}"
        if c in available:
            native.append(c)
    for t in TISSUE_ORDER:
        c = f"pangolin_psi_{t}"
        if c in available:
            native.append(c)

    # SpliceAI
    for c in ("spliceai_acc", "spliceai_don"):
        if c in available:
            native.append(c)

    # Per-tissue probe outputs (only present in parquets written with
    # --per-tissue-probes from bench_score.py)
    has_per_tissue_probes = all(
        f"probe_{ch}_{t}" in available
        for ch in ("none", "acc", "don") for t in TISSUE_ORDER
    )
    if has_per_tissue_probes:
        for t in TISSUE_ORDER:
            for ch in ("none", "acc", "don"):
                native.append(f"probe_{ch}_{t}")

    # PSI-side per-tissue probe outputs (present in parquets written with
    # both --per-tissue-probes and --use-psi-models, when PSI-side probes exist)
    has_per_tissue_probes_psi = all(
        f"probe_{ch}_psi_{t}" in available
        for ch in ("none", "acc", "don") for t in TISSUE_ORDER
    )
    if has_per_tissue_probes_psi:
        for t in TISSUE_ORDER:
            for ch in ("none", "acc", "don"):
                native.append(f"probe_{ch}_psi_{t}")

    # Derived columns (operate on a dict mapping native column name -> ndarray)
    derived = []
    derived.append(("spliceai_max",
                    lambda cd: np.maximum(cd["spliceai_acc"], cd["spliceai_don"])))
    derived.append(("probe_max",
                    lambda cd: np.maximum(cd["probe_acc"], cd["probe_don"])))

    pangolin_p_tissues = [f"pangolin_p_{t}" for t in TISSUE_ORDER if f"pangolin_p_{t}" in available]
    pangolin_psi_tissues = [f"pangolin_psi_{t}" for t in TISSUE_ORDER if f"pangolin_psi_{t}" in available]
    if pangolin_p_tissues:
        derived.append(("pangolin_p_ensemble",
                        lambda cd, _cols=pangolin_p_tissues:
                            np.mean(np.stack([cd[c] for c in _cols], axis=0), axis=0)))
    if pangolin_psi_tissues:
        derived.append(("pangolin_psi_ensemble",
                        lambda cd, _cols=pangolin_psi_tissues:
                            np.mean(np.stack([cd[c] for c in _cols], axis=0), axis=0)))

    if has_per_tissue_probes:
        for t in TISSUE_ORDER:
            ca, cd_ = f"probe_acc_{t}", f"probe_don_{t}"
            derived.append((f"probe_max_{t}",
                            lambda cd, _a=ca, _d=cd_:
                                np.maximum(cd[_a], cd[_d])))

    if has_per_tissue_probes_psi:
        for t in TISSUE_ORDER:
            ca, cd_ = f"probe_acc_psi_{t}", f"probe_don_psi_{t}"
            derived.append((f"probe_max_psi_{t}",
                            lambda cd, _a=ca, _d=cd_:
                                np.maximum(cd[_a], cd[_d])))
        # PSI-side probe ensemble: tissue-mean of the PSI-side probes, then
        # max(acc, don) — mirrors pangolin_psi_ensemble (mean over tissues).
        # (The P-side ensemble `probe_max` comes from the runner's own
        # fold×tissue ensemble; the PSI side has no runner ensemble in the
        # parquet, so we build the analogous tissue-mean here.)
        _acc_psi = [f"probe_acc_psi_{t}" for t in TISSUE_ORDER]
        _don_psi = [f"probe_don_psi_{t}" for t in TISSUE_ORDER]
        derived.append(("probe_max_psi_ensemble",
                        lambda cd, _a=_acc_psi, _d=_don_psi:
                            np.maximum(
                                np.mean(np.stack([cd[c] for c in _a], axis=0), axis=0),
                                np.mean(np.stack([cd[c] for c in _d], axis=0), axis=0))))

    all_names = native + [name for name, _ in derived]
    return native, derived, all_names


def _apply_probe_correction(p_none, p_acc, p_don, k):
    """Bayesian none-class correction in-place; returns (none, acc, don) views."""
    if k == 1.0:
        return p_none, p_acc, p_don
    k = np.float32(k)
    s = p_none * k + p_acc + p_don
    np.maximum(s, np.float32(1e-12), out=s)
    return (p_none * k) / s, p_acc / s, p_don / s


def maybe_correct_probes(col_dict, k):
    """Apply correction to ensemble + per-tissue probe triples in col_dict."""
    if k == 1.0:
        return
    # Ensemble
    if all(c in col_dict for c in ("probe_none", "probe_acc", "probe_don")):
        cn, ca, cd = _apply_probe_correction(
            col_dict["probe_none"], col_dict["probe_acc"], col_dict["probe_don"], k)
        col_dict["probe_none"] = cn
        col_dict["probe_acc"] = ca
        col_dict["probe_don"] = cd
    # Per-tissue (P-side) and PSI-side per-tissue
    for prefix_a, prefix_d, prefix_n in (
        ("probe_acc_{}", "probe_don_{}", "probe_none_{}"),
        ("probe_acc_psi_{}", "probe_don_psi_{}", "probe_none_psi_{}"),
    ):
        for t in TISSUE_ORDER:
            n, a, d = prefix_n.format(t), prefix_a.format(t), prefix_d.format(t)
            if all(c in col_dict for c in (n, a, d)):
                cn, ca, cd = _apply_probe_correction(
                    col_dict[n], col_dict[a], col_dict[d], k)
                col_dict[n] = cn
                col_dict[a] = ca
                col_dict[d] = cd


# ---------------------------------------------------------------------------
# Spearman accumulator (two passes)
# ---------------------------------------------------------------------------

class SpearmanAccumulator:
    """Streamed Spearman ρ via 4096-bin rank lookup.

    Pass 1: update_hist(X) — accumulates per-column histograms.
    finalize_hist()       — converts histograms to mid-rank percentile lookups.
    Pass 2: update_pearson(X) — looks up ranks and accumulates Pearson moments.
    correlation()         — returns the K×K Spearman matrix.
    """

    def __init__(self, n_cols, n_bins=N_BINS_DEFAULT):
        self.n_cols = n_cols
        self.n_bins = n_bins
        self.hist = np.zeros((n_cols, n_bins), dtype=np.int64)
        self.n = 0
        self.sum_x = None
        self.sum_x2 = None
        self.sum_xy = None
        self._lookup = None
        self._n_finalised = 0

    def _bin(self, X):
        """Clip to [0, 1] and bin into n_bins."""
        b = np.empty(X.shape, dtype=np.int32)
        np.multiply(X, self.n_bins - 1, out=b, casting="unsafe")
        np.clip(b, 0, self.n_bins - 1, out=b)
        return b

    def _drop_nans(self, X):
        mask = np.isfinite(X).all(axis=1)
        return X if mask.all() else X[mask]

    def update_hist(self, X):
        X = self._drop_nans(X)
        if X.shape[0] == 0:
            return
        bins = self._bin(X)
        for c in range(self.n_cols):
            self.hist[c] += np.bincount(bins[:, c], minlength=self.n_bins)[:self.n_bins]

    def finalize_hist(self):
        n_per_col = self.hist.sum(axis=1)
        n = int(n_per_col[0]) if n_per_col.size else 0
        if n == 0:
            return
        self._n_finalised = n
        cum = np.cumsum(self.hist, axis=1).astype(np.float64)
        # Mid-rank percentile per bin: (count_below + count_in/2) / N
        self._lookup = (cum - self.hist / 2.0) / float(n)
        self._lookup = self._lookup.astype(np.float32, copy=False)
        # Pass-2 accumulators
        self.sum_x = np.zeros(self.n_cols, dtype=np.float64)
        self.sum_x2 = np.zeros(self.n_cols, dtype=np.float64)
        self.sum_xy = np.zeros((self.n_cols, self.n_cols), dtype=np.float64)

    def update_pearson(self, X):
        if self._lookup is None:
            return
        X = self._drop_nans(X)
        if X.shape[0] == 0:
            return
        bins = self._bin(X)
        ranks = np.empty(bins.shape, dtype=np.float32)
        for c in range(self.n_cols):
            ranks[:, c] = self._lookup[c, bins[:, c]]
        self.n += ranks.shape[0]
        rd = ranks.astype(np.float64)
        self.sum_x += rd.sum(axis=0)
        self.sum_x2 += np.einsum("ij,ij->j", rd, rd)
        self.sum_xy += rd.T @ rd

    def correlation(self):
        if self.n == 0:
            return np.full((self.n_cols, self.n_cols), np.nan)
        mean = self.sum_x / self.n
        var = self.sum_x2 / self.n - mean * mean
        var = np.clip(var, 1e-30, None)
        std = np.sqrt(var)
        cov = self.sum_xy / self.n - np.outer(mean, mean)
        return cov / np.outer(std, std)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _read_batches(files, native_cols, derived, correction_k, batch_size):
    """Generator yielding (chrom_name, label_array, X_array) for each batch
    across all files. X is (rows, n_cols) float32 in `native_cols + derived` order."""
    needed = ["label"] + native_cols
    for path in files:
        if not path.exists():
            _log(f"  [skip] missing {path}")
            continue
        pf = pq.ParquetFile(str(path))
        n_rows = pf.metadata.num_rows
        _log(f"    {path.name}: {n_rows:,} rows")
        for batch in pf.iter_batches(batch_size=batch_size, columns=needed):
            label = batch.column("label").to_numpy().astype(np.int8, copy=False)
            col_dict = {c: batch.column(c).to_numpy().astype(np.float32, copy=False)
                        for c in native_cols}
            # In-stream Bayesian correction
            maybe_correct_probes(col_dict, correction_k)
            # Stack native columns
            stacked = [col_dict[c] for c in native_cols]
            # Compute derived
            for name, fn in derived:
                stacked.append(fn(col_dict).astype(np.float32, copy=False))
            X = np.column_stack(stacked)
            yield label, X
            del col_dict, stacked, X
            gc.collect()


def run_two_passes(files, native_cols, derived, all_names, correction_k, batch_size):
    accs = {s: SpearmanAccumulator(len(all_names)) for s in SUBSETS}

    t0 = time.time()
    _log(f"== pass 1: building histograms ==")
    n_done = 0
    for label, X in _read_batches(files, native_cols, derived, correction_k, batch_size):
        accs["all"].update_hist(X)
        accs["acceptors"].update_hist(X[label == ACC_CLASS])
        accs["donors"].update_hist(X[label == DON_CLASS])
        accs["nonsites"].update_hist(X[label == NONE_CLASS])
        n_done += X.shape[0]
        if n_done % (batch_size * 10) == 0:
            _log(f"    pass1 rows: {n_done:,}  ({time.time()-t0:.1f}s)")
    _log(f"  pass 1 done: {n_done:,} rows  ({time.time()-t0:.1f}s)")

    for s in SUBSETS:
        accs[s].finalize_hist()

    t1 = time.time()
    _log(f"== pass 2: rank-space Pearson moments ==")
    n_done = 0
    for label, X in _read_batches(files, native_cols, derived, correction_k, batch_size):
        accs["all"].update_pearson(X)
        accs["acceptors"].update_pearson(X[label == ACC_CLASS])
        accs["donors"].update_pearson(X[label == DON_CLASS])
        accs["nonsites"].update_pearson(X[label == NONE_CLASS])
        n_done += X.shape[0]
        if n_done % (batch_size * 10) == 0:
            _log(f"    pass2 rows: {n_done:,}  ({time.time()-t1:.1f}s)")
    _log(f"  pass 2 done: {n_done:,} rows  ({time.time()-t1:.1f}s)")

    return accs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_correction_k(arg, optimal_json_path=None):
    if arg is None:
        return 1.0
    s = str(arg).lower()
    if s in ("raw", "none", "1", "1.0"):
        return 1.0
    if s == "optimal":
        if optimal_json_path is None:
            candidates = [
                Path("src/bipangolin/data/probes/optimal_correction.json"),
                Path("bipangolin_probes/optimal_correction.json"),
            ]
            optimal_json_path = next((p for p in candidates if p.exists()), None)
            if optimal_json_path is None:
                raise FileNotFoundError(
                    "Could not find optimal_correction.json — pass an explicit "
                    "--correction-k value or --optimal-correction-json.")
        with open(optimal_json_path) as f:
            doc = json.load(f)
        return float(doc["empirical_sweep"]["best_k"])
    return float(arg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="Directory of {chrom}.parquet files")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--chroms", nargs="+", default=None,
                    help="Restrict to these chroms (default: every *.parquet in --scores)")
    ap.add_argument("--correction-k", default="optimal",
                    help="Bayesian none-class correction k applied to probe columns "
                         "in-stream. 'optimal' (default) reads best_k from "
                         "optimal_correction.json. Spearman ρ is largely insensitive "
                         "to k, but it's still applied for consistency with downstream "
                         "Pearson-based code. Pass 'raw'/1.0 to disable.")
    ap.add_argument("--optimal-correction-json", default=None)
    args = ap.parse_args()
    correction_k = _resolve_correction_k(args.correction_k, args.optimal_correction_json)
    _log(f"correction_k = {correction_k}")

    scores_dir = Path(args.scores)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.chroms:
        files = [scores_dir / f"{c}.parquet" for c in args.chroms]
    else:
        files = sorted(scores_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found in {scores_dir}")

    # Peek schema from the first file to decide which columns exist.
    schema = pq.read_schema(str(files[0]))
    native_cols, derived, all_names = detect_columns(schema.names)
    _log(f"native columns ({len(native_cols)}): {native_cols}")
    _log(f"derived columns ({len(derived)}): {[d[0] for d in derived]}")
    _log(f"total correlation columns: {len(all_names)}")

    accs = run_two_passes(files, native_cols, derived, all_names,
                          correction_k, args.batch_size)

    out = {"columns": np.array(all_names, dtype="U60")}
    for s in SUBSETS:
        out[f"corr_{s}"] = accs[s].correlation()
        out[f"n_{s}"] = np.array([accs[s].n], dtype=np.int64)
        _log(f"  subset {s:10s}  n={accs[s].n:,}")

    np.savez(out_path, **out)
    _log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
