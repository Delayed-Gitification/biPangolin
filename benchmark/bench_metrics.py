"""
bench_metrics.py — fast, compute metrics from the Parquet produced by bench_score.py.

Methods compared (binary task = is this position an acceptor / donor / any splice site):

  biPangolin probe        @ k ∈ {0, 1, 10, 100, 502, 1000}
    acceptor score: p_acc after applying Bayesian none-class correction
    donor    score: p_don after applying Bayesian none-class correction
    any-site score: 1 - p_none after correction

  Pangolin P(spliced)     per tissue        — any-site only (no A/D split)
  Pangolin PSI            per tissue        — any-site only
  SpliceAI                                  — A, D, and any-site = max(A, D)

Metrics per (chrom, method, task):

  pr_auc, roc_auc, top_n_recall, top_n_precision, n_positives, n_total

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
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2


def _detect_tissues(column_names):
    """Pull tissue suffixes from columns named pangolin_p_{tissue}."""
    prefix = "pangolin_p_"
    return sorted(c[len(prefix):] for c in column_names if c.startswith(prefix))


# ---------------------------------------------------------------------------
# Metric helpers — vectorised, no sklearn dep
# ---------------------------------------------------------------------------

def _drop_nan(score, label):
    mask = np.isfinite(score)
    if mask.all():
        return score, label
    return score[mask], label[mask]


def pr_auc(score, y):
    """Average-precision style PR-AUC (trapezoidal). y is 0/1."""
    score, y = _drop_nan(score, y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / y.sum()
    precision = tp / (tp + fp)
    # Prepend (recall=0, precision=1) so the first interval is integrated.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def roc_auc(score, y):
    score, y = _drop_nan(score, y)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y_sorted = y[order].astype(np.float64)
    tpr = np.cumsum(y_sorted) / pos
    fpr = np.cumsum(1 - y_sorted) / neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


def top_n(score, y, n=None):
    """Top-N recall + precision. N defaults to the positive count."""
    score, y = _drop_nan(score, y)
    pos = int(y.sum())
    if pos == 0:
        return float("nan"), float("nan")
    n = n or pos
    if n > len(score):
        n = len(score)
    order = np.argsort(-score, kind="stable")
    top = y[order[:n]]
    hits = int(top.sum())
    return hits / pos, hits / n   # recall, precision


# ---------------------------------------------------------------------------
# Probe correction (post-hoc Bayesian)
# ---------------------------------------------------------------------------

def apply_correction(p_none, p_acc, p_don, k):
    """Return corrected (p_none, p_acc, p_don) after multiplying p_none by k
    and renormalising."""
    p_none = p_none.astype(np.float32) * float(k)
    p_acc = p_acc.astype(np.float32)
    p_don = p_don.astype(np.float32)
    s = p_none + p_acc + p_don
    s = np.clip(s, 1e-12, None)
    return p_none / s, p_acc / s, p_don / s


# ---------------------------------------------------------------------------
# Metric driver
# ---------------------------------------------------------------------------

def metrics_for_binary(score, y_bin, prefix, chrom, method, rows):
    """Compute pr_auc, roc_auc, top_n_recall, top_n_precision for one task."""
    rec, prec = top_n(score, y_bin)
    rows.append({
        "chrom": chrom, "method": method, "task": prefix,
        "metric": "pr_auc", "value": pr_auc(score, y_bin),
        "n_pos": int(y_bin.sum()), "n_total": int(len(y_bin)),
    })
    rows.append({
        "chrom": chrom, "method": method, "task": prefix,
        "metric": "roc_auc", "value": roc_auc(score, y_bin),
        "n_pos": int(y_bin.sum()), "n_total": int(len(y_bin)),
    })
    rows.append({
        "chrom": chrom, "method": method, "task": prefix,
        "metric": "top_n_recall", "value": rec,
        "n_pos": int(y_bin.sum()), "n_total": int(len(y_bin)),
    })
    rows.append({
        "chrom": chrom, "method": method, "task": prefix,
        "metric": "top_n_precision", "value": prec,
        "n_pos": int(y_bin.sum()), "n_total": int(len(y_bin)),
    })


def compute_chrom_metrics(table, chrom, k_values, tissues):
    """Return a list of dicts (one per row in the summary CSV)."""
    rows = []
    label = table["label"].to_numpy()
    is_acc = (label == ACC_CLASS).astype(np.int8)
    is_don = (label == DON_CLASS).astype(np.int8)
    is_any = (label != NONE_CLASS).astype(np.int8)

    pn = table["probe_none"].to_numpy().astype(np.float32)
    pa = table["probe_acc"].to_numpy().astype(np.float32)
    pd_ = table["probe_don"].to_numpy().astype(np.float32)

    # --- biPangolin probe (per k correction) -------------------------------
    for k in k_values:
        c_none, c_acc, c_don = apply_correction(pn, pa, pd_, k)
        method = f"biPangolin_k{k}"
        metrics_for_binary(c_acc, is_acc, "acceptor", chrom, method, rows)
        metrics_for_binary(c_don, is_don, "donor", chrom, method, rows)
        metrics_for_binary(1.0 - c_none, is_any, "any_site", chrom, method, rows)

        # 3-class argmax accuracy on labelled positions only
        probs = np.stack([c_none, c_acc, c_don], axis=0)
        pred = probs.argmax(axis=0)
        labelled = (label != NONE_CLASS) | (probs.max(axis=0) > 0)
        # Restrict to labelled-or-confident positions? -> just compute full acc + restricted.
        full_acc = float((pred == label).mean())
        pos_mask = (label != NONE_CLASS)
        pos_acc = float((pred[pos_mask] == label[pos_mask]).mean()) if pos_mask.any() else float("nan")
        rows.append({
            "chrom": chrom, "method": method, "task": "3class_argmax",
            "metric": "accuracy_all", "value": full_acc,
            "n_pos": int(pos_mask.sum()), "n_total": int(len(label)),
        })
        rows.append({
            "chrom": chrom, "method": method, "task": "3class_argmax",
            "metric": "accuracy_at_truesites", "value": pos_acc,
            "n_pos": int(pos_mask.sum()), "n_total": int(len(label)),
        })

    # --- Pangolin per-tissue P(spliced) and PSI (any-site only) -----------
    for t in tissues:
        for col, kind in ((f"pangolin_p_{t}", "p"), (f"pangolin_psi_{t}", "psi")):
            if col not in table.column_names:
                continue
            s = table[col].to_numpy().astype(np.float32)
            method = f"pangolin_{kind}_{t}"
            metrics_for_binary(s, is_any, "any_site", chrom, method, rows)

    # --- SpliceAI ----------------------------------------------------------
    if "spliceai_acc" in table.column_names:
        sa = table["spliceai_acc"].to_numpy().astype(np.float32)
        sd = table["spliceai_don"].to_numpy().astype(np.float32)
        # If SpliceAI was skipped, all values are NaN — pr_auc / roc_auc / top_n
        # already drop NaNs and will return NaN cleanly.
        metrics_for_binary(sa, is_acc, "acceptor", chrom, "spliceai", rows)
        metrics_for_binary(sd, is_don, "donor", chrom, "spliceai", rows)
        any_score = np.maximum(sa, sd)
        metrics_for_binary(any_score, is_any, "any_site", chrom, "spliceai", rows)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print(f"== {chrom}: reading {path} ==")
        table = pq.read_table(str(path))
        print(f"  {table.num_rows:,} rows, {len(table.column_names)} columns")
        tissues = _detect_tissues(table.column_names)
        rows = compute_chrom_metrics(table, chrom, args.k, tissues)
        rows_all.extend(rows)

    # Pool across chroms by concatenating raw arrays and recomputing.
    # Cheaper alternative: weighted-mean from per-chrom rows. We compute the
    # weighted mean by n_pos for AUC-style metrics, and total counts for top-N.
    pooled = _pool_across_chroms(rows_all)
    rows_all.extend(pooled)

    # Write summary CSV
    cols = ["chrom", "method", "task", "metric", "value", "n_pos", "n_total"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_all:
            w.writerow(r)
    print(f"wrote {out_path}  ({len(rows_all):,} rows)")


def _pool_across_chroms(rows):
    """Add chrom='ALL' rows using n_pos-weighted means (best honest approximation
    without re-reading the parquet). Note: PR-AUC pooled this way is not exact;
    if exact is needed, rerun with a single concatenated chrom file."""
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        key = (r["method"], r["task"], r["metric"])
        bucket[key].append(r)

    pooled = []
    for (method, task, metric), rs in bucket.items():
        if any(r["chrom"] == "ALL" for r in rs):
            continue
        weights = np.array([r["n_pos"] or 1 for r in rs], dtype=np.float64)
        values = np.array([r["value"] if r["value"] is not None
                           and not (isinstance(r["value"], float) and np.isnan(r["value"]))
                           else np.nan for r in rs], dtype=np.float64)
        mask = np.isfinite(values)
        if not mask.any():
            v = float("nan")
        else:
            v = float(np.average(values[mask], weights=weights[mask]))
        pooled.append({
            "chrom": "ALL",
            "method": method,
            "task": task,
            "metric": metric,
            "value": v,
            "n_pos": int(sum(r["n_pos"] for r in rs)),
            "n_total": int(sum(r["n_total"] for r in rs)),
        })
    return pooled


if __name__ == "__main__":
    main()
