"""
bench_metrics.py — compute metrics from the Parquet produced by bench_score.py.

Memory-efficient design:
  * Reads one score column at a time (Parquet projection) — never holds all 14 in RAM.
  * Computes pr_auc + roc_auc + top_n_recall + top_n_precision from a single sort
    over the (score, label) pair, not four independent sorts.
  * Frees intermediates with `del` + `gc.collect()` between columns.

Methods compared (binary task = is this position an acceptor / donor / any splice site):

  biPangolin probe        @ k ∈ {0, 1, 10, 100, 502, 1000}
    acceptor score: p_acc after applying Bayesian none-class correction
    donor    score: p_don after applying Bayesian none-class correction
    any-site score: 1 - p_none after correction

  Pangolin P(spliced)     per tissue        — any-site only (no A/D split)
  Pangolin PSI            per tissue        — any-site only
  SpliceAI                                  — A, D, and any-site = max(A, D)

Plus 3-class argmax accuracy for the probe (after each k correction).

Output: a long-format summary CSV.

Typical usage:
    python benchmark/bench_metrics.py \\
        --scores bench_scores/ \\
        --out    bench_metrics/summary.csv \\
        [--k 0 1 10 100 502 1000]
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _log(msg):
    """Stdout print that actually flushes."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# numpy 2.x renamed np.trapz -> np.trapezoid (and removed the alias in newer
# builds). Keep both interpreters happy.
_trapz = getattr(np, "trapezoid", None) or np.trapz


NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2


# ---------------------------------------------------------------------------
# Shared-sort metric kernel
# ---------------------------------------------------------------------------

def metric_bundle(score, y, drop_nan=True):
    """Compute pr_auc, roc_auc, top_n_recall, top_n_precision in one sort.

    score: float32 array
    y:     int8 array, 0/1
    Returns dict with the four metrics + n_pos + n_total.
    """
    if drop_nan:
        mask = np.isfinite(score)
        if not mask.all():
            score = score[mask]
            y = y[mask]
    n_total = int(len(y))
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n_total:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                "top_n_recall": float("nan"), "top_n_precision": float("nan"),
                "n_pos": n_pos, "n_total": n_total}

    # Single sort — descending. Quicksort is ~3-5x faster than stable mergesort,
    # and tie-order doesn't change PR/ROC AUC values.
    order = np.argsort(-score, kind="quicksort")
    y_sorted = y[order].astype(np.int32, copy=False)
    del order

    # Cumulative TP/FP. Keep int32; max 2^31 is plenty for any chrom.
    tp = np.cumsum(y_sorted, dtype=np.int64)
    fp = np.cumsum(1 - y_sorted, dtype=np.int64)
    del y_sorted

    # top-N (N = n_pos)
    n = n_pos
    top_n_recall = float(tp[n - 1] / n_pos)
    top_n_precision = float(tp[n - 1] / n)

    # PR curve
    recall = tp / n_pos
    denom = (tp + fp).astype(np.float64)
    precision = tp / np.maximum(denom, 1.0)
    pr = float(_trapz(
        np.concatenate([[1.0], precision]),
        np.concatenate([[0.0], recall]),
    ))

    # ROC curve
    n_neg = n_total - n_pos
    tpr = recall                                  # already tp / n_pos
    fpr = fp / n_neg
    roc = float(_trapz(
        np.concatenate([[0.0], tpr]),
        np.concatenate([[0.0], fpr]),
    ))

    return {"pr_auc": pr, "roc_auc": roc,
            "top_n_recall": top_n_recall, "top_n_precision": top_n_precision,
            "n_pos": n_pos, "n_total": n_total}


def _append_bundle(rows, chrom, method, task, bundle):
    base = {"chrom": chrom, "method": method, "task": task,
            "n_pos": bundle["n_pos"], "n_total": bundle["n_total"]}
    for metric in ("pr_auc", "roc_auc", "top_n_recall", "top_n_precision"):
        rows.append({**base, "metric": metric, "value": bundle[metric]})


# ---------------------------------------------------------------------------
# Per-chrom driver — column-streamed
# ---------------------------------------------------------------------------

def _read_col(path, name, dtype):
    t = pq.read_table(str(path), columns=[name])
    a = t.column(name).to_numpy()
    del t
    if a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    return a


def _subsample_indices(label, max_neg_ratio, seed=0):
    """Return a sorted index array keeping all positives + at most
    max_neg_ratio × n_pos negatives. If max_neg_ratio is None, returns None
    (meaning "use everything")."""
    if max_neg_ratio is None or max_neg_ratio <= 0:
        return None
    pos_idx = np.where(label != NONE_CLASS)[0]
    neg_idx = np.where(label == NONE_CLASS)[0]
    n_neg_keep = int(max_neg_ratio * len(pos_idx))
    if len(neg_idx) <= n_neg_keep:
        return None  # already small enough — no point subsampling
    rng = np.random.default_rng(seed)
    neg_idx = rng.choice(neg_idx, n_neg_keep, replace=False)
    keep = np.concatenate([pos_idx, neg_idx])
    keep.sort()
    return keep


def compute_chrom_metrics(path, chrom, k_values, tissues, available_cols,
                          max_neg_ratio=None):
    """Stream-process one chrom's parquet, freeing arrays between columns."""
    rows = []

    t0 = time.time()
    _log(f"  reading labels...")
    label_full = _read_col(path, "label", np.int8)
    n_full = len(label_full)
    keep_idx = _subsample_indices(label_full, max_neg_ratio)
    if keep_idx is not None:
        label = label_full[keep_idx]
        _log(f"  subsampled negatives: kept {len(keep_idx):,} of {n_full:,} positions "
             f"(max_neg_ratio={max_neg_ratio})")
        del label_full
    else:
        label = label_full
        _log(f"  using all {n_full:,} positions")
    is_acc = (label == ACC_CLASS).astype(np.int8)
    is_don = (label == DON_CLASS).astype(np.int8)
    is_any = (label != NONE_CLASS).astype(np.int8)
    n_total = len(label)
    _log(f"  n_total={n_total:,}  acc={int(is_acc.sum()):,}  "
         f"don={int(is_don.sum()):,}  any={int(is_any.sum()):,}  "
         f"({time.time()-t0:.1f}s)")

    def load_score(name):
        t = time.time()
        a = _read_col(path, name, np.float32)
        if keep_idx is not None:
            a = a[keep_idx]
        _log(f"  loaded {name} ({time.time()-t:.1f}s, {a.nbytes/1e9:.2f} GB)")
        return a

    def run_bundle(method, task, score, y):
        t = time.time()
        b = metric_bundle(score, y)
        _append_bundle(rows, chrom, method, task, b)
        _log(f"    {method:25s} {task:10s}  pr_auc={b['pr_auc']:.4f}  "
             f"roc_auc={b['roc_auc']:.4f}  ({time.time()-t:.1f}s)")

    # --- biPangolin probe -----------------------------------------------------
    pn = load_score("probe_none")
    pa = load_score("probe_acc")
    pd_ = load_score("probe_don")

    for k in k_values:
        method = f"biPangolin_k{int(k)}" if float(k).is_integer() else f"biPangolin_k{k}"
        # Apply correction in-place-ish; we need three arrays but reuse buffers.
        s = pn * np.float32(k) + pa + pd_
        np.maximum(s, 1e-12, out=s)
        c_none = (pn * np.float32(k)) / s
        c_acc = pa / s
        c_don = pd_ / s
        del s
        gc.collect()

        run_bundle(method, "acceptor", c_acc, is_acc)
        run_bundle(method, "donor", c_don, is_don)
        one_minus_none = np.float32(1.0) - c_none
        run_bundle(method, "any_site", one_minus_none, is_any)
        del one_minus_none

        # 3-class argmax accuracy (cheap, no sort)
        # pred = argmax of (c_none, c_acc, c_don)
        max_acc = (c_acc > c_none) & (c_acc >= c_don)
        max_don = (c_don > c_none) & (c_don > c_acc)
        pred = np.where(max_don, 2, np.where(max_acc, 1, 0)).astype(np.int8)
        del max_acc, max_don
        full_acc = float((pred == label).mean())
        pos_mask = (label != NONE_CLASS)
        pos_acc = float((pred[pos_mask] == label[pos_mask]).mean()) if pos_mask.any() else float("nan")
        rows.append({"chrom": chrom, "method": method, "task": "3class_argmax",
                     "metric": "accuracy_all", "value": full_acc,
                     "n_pos": int(pos_mask.sum()), "n_total": n_total})
        rows.append({"chrom": chrom, "method": method, "task": "3class_argmax",
                     "metric": "accuracy_at_truesites", "value": pos_acc,
                     "n_pos": int(pos_mask.sum()), "n_total": n_total})

        del c_none, c_acc, c_don, pred, pos_mask
        gc.collect()

    del pn, pa, pd_
    gc.collect()

    # --- biPangolin PSI-side probe (ensemble average across tissues) ----------
    psi_probe_cols = [f"probe_{c}_psi_{t}"
                      for c in ("none", "acc", "don") for t in tissues
                      if f"probe_{c}_psi_{t}" in available_cols]
    if psi_probe_cols and all(
        f"probe_{c}_psi_{t}" in available_cols
        for c in ("none", "acc", "don") for t in tissues
    ):
        _log("  computing PSI-side probe ensemble (mean across tissues)...")
        pn_psi = np.mean([load_score(f"probe_none_psi_{t}") for t in tissues], axis=0).astype(np.float32)
        pa_psi = np.mean([load_score(f"probe_acc_psi_{t}")  for t in tissues], axis=0).astype(np.float32)
        pd_psi = np.mean([load_score(f"probe_don_psi_{t}")  for t in tissues], axis=0).astype(np.float32)
        gc.collect()

        for k in k_values:
            method = f"biPangolin_psi_k{int(k)}" if float(k).is_integer() else f"biPangolin_psi_k{k}"
            s = pn_psi * np.float32(k) + pa_psi + pd_psi
            np.maximum(s, 1e-12, out=s)
            c_none = (pn_psi * np.float32(k)) / s
            c_acc  = pa_psi / s
            c_don  = pd_psi / s
            del s
            gc.collect()

            run_bundle(method, "acceptor", c_acc, is_acc)
            run_bundle(method, "donor",    c_don, is_don)
            one_minus_none = np.float32(1.0) - c_none
            run_bundle(method, "any_site", one_minus_none, is_any)
            del one_minus_none

            max_acc = (c_acc > c_none) & (c_acc >= c_don)
            max_don = (c_don > c_none) & (c_don > c_acc)
            pred = np.where(max_don, 2, np.where(max_acc, 1, 0)).astype(np.int8)
            del max_acc, max_don
            full_acc = float((pred == label).mean())
            pos_mask = (label != NONE_CLASS)
            pos_acc = float((pred[pos_mask] == label[pos_mask]).mean()) if pos_mask.any() else float("nan")
            rows.append({"chrom": chrom, "method": method, "task": "3class_argmax",
                         "metric": "accuracy_all", "value": full_acc,
                         "n_pos": int(pos_mask.sum()), "n_total": n_total})
            rows.append({"chrom": chrom, "method": method, "task": "3class_argmax",
                         "metric": "accuracy_at_truesites", "value": pos_acc,
                         "n_pos": int(pos_mask.sum()), "n_total": n_total})
            del c_none, c_acc, c_don, pred, pos_mask
            gc.collect()

        del pn_psi, pa_psi, pd_psi
        gc.collect()
    else:
        _log("  PSI-side probe columns not found — skipping biPangolin_psi_k* methods.")

    # --- Pangolin per-tissue (any-site only) ----------------------------------
    for tt in tissues:
        for col, kind in ((f"pangolin_p_{tt}", "p"), (f"pangolin_psi_{tt}", "psi")):
            if col not in available_cols:
                continue
            s = load_score(col)
            method = f"pangolin_{kind}_{tt}"
            run_bundle(method, "any_site", s, is_any)
            del s
            gc.collect()

    # --- SpliceAI -------------------------------------------------------------
    if "spliceai_acc" in available_cols and "spliceai_don" in available_cols:
        sa = load_score("spliceai_acc")
        sd = load_score("spliceai_don")
        run_bundle("spliceai", "acceptor", sa, is_acc)
        run_bundle("spliceai", "donor", sd, is_don)
        any_score = np.maximum(sa, sd)
        del sa, sd
        gc.collect()
        run_bundle("spliceai", "any_site", any_score, is_any)
        del any_score
        gc.collect()

    del label, is_acc, is_don, is_any
    if keep_idx is not None:
        del keep_idx
    gc.collect()
    _log(f"  chrom {chrom} done in {time.time()-t0:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# Pooled rows (cheap weighted mean — exact pooling would need raw arrays)
# ---------------------------------------------------------------------------

def pool_across_chroms(rows):
    bucket = defaultdict(list)
    for r in rows:
        if r["chrom"] == "ALL":
            continue
        bucket[(r["method"], r["task"], r["metric"])].append(r)

    pooled = []
    for (method, task, metric), rs in bucket.items():
        weights = np.array([max(r["n_pos"], 1) for r in rs], dtype=np.float64)
        values = np.array([r["value"] if r["value"] is not None
                           and not (isinstance(r["value"], float) and np.isnan(r["value"]))
                           else np.nan for r in rs], dtype=np.float64)
        m = np.isfinite(values)
        v = float(np.average(values[m], weights=weights[m])) if m.any() else float("nan")
        pooled.append({"chrom": "ALL", "method": method, "task": task, "metric": metric,
                       "value": v,
                       "n_pos": int(sum(r["n_pos"] for r in rs)),
                       "n_total": int(sum(r["n_total"] for r in rs))})
    return pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _detect_tissues(column_names):
    prefix = "pangolin_p_"
    return sorted(c[len(prefix):] for c in column_names if c.startswith(prefix))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="Directory of {chrom}.parquet files")
    ap.add_argument("--out", required=True, help="Output summary CSV path")
    ap.add_argument("--k", nargs="+", type=float,
                    default=[0, 1, 10, 100, 502, 1000],
                    help="None-class correction values to evaluate.")
    ap.add_argument("--chroms", nargs="+", default=None,
                    help="Restrict to these chroms (default: every *.parquet in --scores)")
    ap.add_argument("--max-neg-ratio", type=float, default=None,
                    help="If set, subsample negatives to N × n_positives per chrom. "
                         "Tractable runtimes at the cost of slightly biased absolute "
                         "PR-AUC (precision goes up by ~constant factor). Method "
                         "rankings are preserved. Try 500-2000 for a good speed/fidelity "
                         "tradeoff; omit for full chrom evaluation.")
    args = ap.parse_args()

    scores_dir = Path(args.scores)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.chroms:
        files = [scores_dir / f"{c}.parquet" for c in args.chroms]
    else:
        files = sorted(scores_dir.glob("*.parquet"))

    rows_all = []
    for path in files:
        if not path.exists():
            print(f"  [skip] missing {path}")
            continue
        chrom = path.stem
        _log(f"== {chrom}: {path} ==")
        # Peek at schema to discover available columns + tissues.
        schema = pq.read_schema(str(path))
        col_names = schema.names
        tissues = _detect_tissues(col_names)
        _log(f"  tissues: {tissues}")
        rows = compute_chrom_metrics(path, chrom, args.k, tissues, set(col_names),
                                     max_neg_ratio=args.max_neg_ratio)
        rows_all.extend(rows)
        gc.collect()

    # Pooled
    rows_all.extend(pool_across_chroms(rows_all))

    cols = ["chrom", "method", "task", "metric", "value", "n_pos", "n_total"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_all:
            w.writerow(r)
    print(f"wrote {out_path}  ({len(rows_all):,} rows)")


if __name__ == "__main__":
    main()
