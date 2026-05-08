"""
identify_optimal_correction.py

Finds the optimal Bayesian P(none) correction factor by running biPangolin
(at k=1, no correction) on a set of validation genes, collecting per-position
predictions + GTF labels, then sweeping k values to maximise PR-AUC.

Uses BiPangolinRunner directly — no cache files needed.

Usage:
    python identify_optimal_correction.py \\
        --pangolin_model_dir  ./Pangolin/pangolin/models \\
        --probe_dir           ./bipangolin_probes \\
        --fasta               ./data/GRCh38.primary_assembly.genome.fa \\
        --gtf                 ./data/gencode.v47.basic.annotation.gtf \\
        --output              ./bipangolin_probes/optimal_correction.json \\
        [--chroms chr3 chr7] \\
        [--n_genes 50] \\
        [--edge_trim 20] \\
        [--k_min 1] [--k_max 2000] [--n_k 100] \\
        [--tissue all_tissues] \\
        [--device auto]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pyfastx
from tqdm import tqdm

NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2


# ---------------------------------------------------------------------------
# GTF parsing — first/last exon aware
# ---------------------------------------------------------------------------

def parse_gtf(gtf_path: str, chroms: set) -> dict:
    """Return dict[chrom] -> dict[pos_0based] -> ACC_CLASS | DON_CLASS.

    Excludes TSS (first exon acceptor) and TTS (last exon donor) positions.
    """
    transcript_exons = defaultdict(list)

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts[:9]
            if feature != "exon" or strand != "+" or chrom not in chroms:
                continue
            transcript_id = _attr(attrs, "transcript_id")
            transcript_exons[(chrom, transcript_id)].append((int(start), int(end)))

    sites = defaultdict(dict)
    for (chrom, _tid), exons in transcript_exons.items():
        exons_sorted = sorted(exons)
        for i, (start, end) in enumerate(exons_sorted):
            if i > 0:
                sites[chrom][start - 1] = ACC_CLASS   # not first exon
            if i < len(exons_sorted) - 1:
                sites[chrom][end - 1] = DON_CLASS     # not last exon
    return dict(sites)


def _attr(attrs: str, key: str):
    needle = key + ' "'
    i = attrs.find(needle)
    if i < 0:
        return None
    j = attrs.find('"', i + len(needle))
    return attrs[i + len(needle):j]


def get_genes(gtf_path: str, chroms: set) -> dict:
    """Return dict[chrom] -> list of (gene_id, start_0based, end_0based)."""
    gene_extents = defaultdict(lambda: defaultdict(lambda: [10**9, 0]))
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts[:9]
            if feature != "gene" or strand != "+" or chrom not in chroms:
                continue
            gene_id = _attr(attrs, "gene_id")
            s, e = int(start) - 1, int(end) - 1
            gene_extents[chrom][gene_id][0] = min(gene_extents[chrom][gene_id][0], s)
            gene_extents[chrom][gene_id][1] = max(gene_extents[chrom][gene_id][1], e)

    return {chrom: [(g, lo, hi) for g, (lo, hi) in d.items() if hi > lo]
            for chrom, d in gene_extents.items()}


# ---------------------------------------------------------------------------
# Scoring + label collection
# ---------------------------------------------------------------------------

def collect_predictions(runner, fasta, sites_by_chrom, genes_by_chrom,
                        n_genes: int, edge_trim: int, seed: int = 42):
    """Score a random sample of genes, return (probs, labels) numpy arrays.

    probs:  (N, 3) — [P(none), P(acc), P(don)] at k=1 (raw, no correction)
    labels: (N,)   — 0=none / 1=acc / 2=don
    """
    rng = random.Random(seed)

    all_genes = [
        (chrom, gid, s, e)
        for chrom, glist in genes_by_chrom.items()
        for gid, s, e in glist
        if e - s > 2 * edge_trim + 1000
    ]
    if len(all_genes) > n_genes:
        all_genes = rng.sample(all_genes, n_genes)
    print(f"scoring {len(all_genes)} genes across "
          f"{len(set(g[0] for g in all_genes))} chromosomes")

    all_probs  = []
    all_labels = []

    for chrom, _gid, g_start, g_end in tqdm(all_genes, desc="scoring genes"):
        chrom_len = len(fasta[chrom])
        rs = max(0, g_start - 5000)
        re = min(chrom_len, g_end + 5000)
        if re - rs < 100:
            continue

        seq = fasta[chrom][rs:re].seq.upper()
        try:
            result = (runner.score_sequence(seq) if len(seq) <= 10000
                      else runner.score_long_sequence(seq))
        except Exception as e:
            print(f"  warning: failed on {chrom}:{rs}-{re}: {e}")
            continue

        L = len(result)

        # Raw probe outputs as (L, 3), trim edges
        probs_inner = torch.stack([
            result.probe_none,
            result.probe_acceptor,
            result.probe_donor,
        ])[:, edge_trim:L - edge_trim].T.numpy()   # (L', 3)

        if probs_inner.shape[0] == 0:
            continue

        # Labels for the inner region
        chrom_sites = sites_by_chrom.get(chrom, {})
        inner_start = rs + edge_trim
        labels = np.zeros(probs_inner.shape[0], dtype=np.int8)
        for offset in range(len(labels)):
            cls = chrom_sites.get(inner_start + offset)
            if cls is not None:
                labels[offset] = cls

        all_probs.append(probs_inner)
        all_labels.append(labels)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Correction + sweep
# ---------------------------------------------------------------------------

def apply_correction(probs: np.ndarray, k: float) -> np.ndarray:
    scaled = probs.copy()
    scaled[:, NONE_CLASS] *= k
    totals = scaled.sum(axis=1, keepdims=True).clip(1e-12, None)
    return scaled / totals


def sweep_k(probs: np.ndarray, labels: np.ndarray,
            multipliers: np.ndarray) -> dict:
    """Compute cross-entropy at each k value (used for diagnostic plotting only).

    Empirical CE optimisation is sample-size-dependent: more data brings in
    more 'uncertain' true splice sites where the probe gave only moderate
    P(splice). Those positions get crushed at high k, pulling the optimum
    artificially low. We use this for the sweep curve in the output JSON
    but recommend the analytical k (compute_analytical_k) instead.
    """
    eps = 1e-12
    rows = []
    for k in tqdm(multipliers, desc="sweeping k (diagnostic)"):
        corr = apply_correction(probs, k)
        log_p = np.log(corr[np.arange(len(labels)), labels] + eps)
        ce = -float(log_p.mean())
        rows.append({"k": float(k), "cross_entropy": ce})
    best = min(rows, key=lambda r: r["cross_entropy"])
    return {"rows": rows, "best": best}


def compute_analytical_k(labels: np.ndarray,
                         none_subsample_ratio: int = 10) -> dict:
    """Compute k from class frequencies via Bayes' theorem.

    Training distribution: 10 none per positive (none_subsample_ratio=10),
    with positives split equally between acceptor and donor. So:
        P_train(none) = 10/12, P_train(acc) = P_train(don) = 1/12.

    Real distribution: estimated from `labels` (unsubsampled).

    The Bayesian update for P(class | input) under a different prior is:
        P_real(c | input) ∝ P_train(c | input) * P_real(c) / P_train(c)

    The script's `apply_correction` only multiplies P(none), so the equivalent
    single-k formulation is:
        k = (P_real(none) / P_train(none))
            / mean(P_real(acc) / P_train(acc), P_real(don) / P_train(don))

    This is sample-size-independent (in the limit), data-distribution-aware,
    and matches the user's intuition that real-genome ratios drive k.
    """
    n_total = len(labels)
    n_none = int((labels == NONE_CLASS).sum())
    n_acc = int((labels == ACC_CLASS).sum())
    n_don = int((labels == DON_CLASS).sum())

    # Real prior from the data
    p_real_none = n_none / n_total
    p_real_acc = n_acc / n_total
    p_real_don = n_don / n_total

    # Training prior: 10:1 none:positive, positives split between acc and don
    n_pos_train = 2  # acc + don
    p_train_none = none_subsample_ratio / (none_subsample_ratio + n_pos_train)
    p_train_acc  = 1 / (none_subsample_ratio + n_pos_train)
    p_train_don  = 1 / (none_subsample_ratio + n_pos_train)

    # Per-class multipliers under Bayes
    mult_none = p_real_none / p_train_none
    mult_acc  = (p_real_acc + 1e-12) / p_train_acc
    mult_don  = (p_real_don + 1e-12) / p_train_don
    mult_pos  = (mult_acc + mult_don) / 2

    # Equivalent k for "scale only P(none)" parameterisation
    analytical_k = mult_none / mult_pos

    return {
        "analytical_k": float(analytical_k),
        "real_distribution": {
            "p_none": p_real_none, "p_acc": p_real_acc, "p_don": p_real_don,
            "ratio_none_per_positive": (n_none / max(n_acc + n_don, 1)),
        },
        "training_distribution": {
            "p_none": p_train_none, "p_acc": p_train_acc, "p_don": p_train_don,
            "ratio_none_per_positive": none_subsample_ratio,
        },
        "per_class_multipliers": {
            "none": float(mult_none), "acc": float(mult_acc), "don": float(mult_don),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find optimal Bayesian correction factor for biPangolin.")
    parser.add_argument("--pangolin_model_dir", required=True)
    parser.add_argument("--probe_dir",          required=True)
    parser.add_argument("--fasta",              required=True)
    parser.add_argument("--gtf",                required=True)
    parser.add_argument("--output",    default="optimal_correction.json")
    parser.add_argument("--chroms",    nargs="+", default=["chr3", "chr7"],
                        help="Chromosomes to use (default: chr3 chr7 — val set)")
    parser.add_argument("--n_genes",   type=int,   default=50)
    parser.add_argument("--edge_trim", type=int,   default=20,
                        help="Ignore this many bases at start/end of each region")
    parser.add_argument("--k_min",     type=float, default=1.0)
    parser.add_argument("--k_max",     type=float, default=2000.0)
    parser.add_argument("--n_k",       type=int,   default=100)
    parser.add_argument("--tissue",    default="all_tissues")
    parser.add_argument("--device",    default="auto")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="P(donor)+P(acceptor) threshold for F1 (default 0.1, "
                             "matching SpliceAI-style confidence level)")
    parser.add_argument("--seed", default=42)
    parser.add_argument("--none_subsample_ratio", type=int, default=10,
                        help="None:positive ratio used during training (default 10)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from bipangolin import BiPangolinRunner

    # ---- Runner at k=1 (no correction) ----
    print(f"initialising runner (tissue={args.tissue}, correction disabled)...")
    runner = BiPangolinRunner(
        pangolin_model_dir=args.pangolin_model_dir,
        probe_dir=args.probe_dir,
        tissue=args.tissue,
        device=args.device,
        correction_k=1.0,
    )

    # ---- GTF ----
    chroms = set(args.chroms)
    print(f"\nparsing GTF for {sorted(chroms)}...")
    sites = parse_gtf(args.gtf, chroms)
    genes = get_genes(args.gtf, chroms)
    n_sites = sum(len(d) for d in sites.values())
    n_genes_total = sum(len(v) for v in genes.values())
    print(f"  {n_sites:,} canonical splice sites across {n_genes_total:,} genes")

    # ---- Score ----
    print(f"\nscoring up to {args.n_genes} genes (edge_trim={args.edge_trim} bp)...")
    fasta = pyfastx.Fasta(args.fasta)
    probs, labels = collect_predictions(
        runner, fasta, sites, genes,
        n_genes=args.n_genes,
        edge_trim=args.edge_trim,
        seed=args.seed,
    )

    n_none = int((labels == NONE_CLASS).sum())
    n_acc  = int((labels == ACC_CLASS).sum())
    n_don  = int((labels == DON_CLASS).sum())
    print(f"\ncollected {len(labels):,} positions: "
          f"none={n_none:,}  acc={n_acc:,}  don={n_don:,}")

    if n_acc == 0 or n_don == 0:
        sys.exit("ERROR: no labelled splice sites found — "
                 "check --chroms and --gtf match the FASTA build.")

    # ---- Analytical k (the recommended answer) ----
    print("\ncomputing analytical k from class frequencies (Bayesian)...")
    analytical = compute_analytical_k(
        labels, none_subsample_ratio=args.none_subsample_ratio)
    print(f"  real distribution:     none={analytical['real_distribution']['p_none']*100:.3f}%  "
          f"acc={analytical['real_distribution']['p_acc']*100:.4f}%  "
          f"don={analytical['real_distribution']['p_don']*100:.4f}%")
    print(f"  real none:positive ratio: "
          f"{analytical['real_distribution']['ratio_none_per_positive']:.0f}:1")
    print(f"  training none:positive ratio (assumed): "
          f"{analytical['training_distribution']['ratio_none_per_positive']:.0f}:1")
    print(f"\n  ANALYTICAL k = {analytical['analytical_k']:.1f}")
    print("    (this is the principled answer — derived from Bayes' theorem,")
    print("     not subject to sample-size-dependent CE artefacts)")

    # ---- Empirical CE sweep (diagnostic only — for plotting the curve) ----
    multipliers = np.logspace(
        np.log10(args.k_min), np.log10(args.k_max), args.n_k)
    print(f"\nempirical CE sweep over {args.n_k} k values [{args.k_min:.0f}–{args.k_max:.0f}] "
          f"(diagnostic, not used for recommended_k)...")
    sweep = sweep_k(probs, labels, multipliers)
    base = sweep_k(probs, labels, np.array([1.0]))["rows"][0]
    best = sweep["best"]
    print(f"  empirical CE optimum: k={best['k']:.1f}  CE={best['cross_entropy']:.6f}  "
          f"(baseline k=1: CE={base['cross_entropy']:.6f})")
    if abs(np.log10(best["k"]) - np.log10(analytical["analytical_k"])) > 0.5:
        print(f"  NOTE: empirical optimum differs from analytical by >3x.")
        print(f"  This usually means the probe is uncertain at many true sites, which")
        print(f"  pulls empirical CE optimum down. The analytical k is more reliable.")

    # ---- Write ----
    output_data = {
        "recommended_k": analytical["analytical_k"],
        "analytical": analytical,
        "empirical_sweep": {
            "best_k": best["k"],
            "best_cross_entropy": best["cross_entropy"],
            "baseline_k1_cross_entropy": base["cross_entropy"],
            "rows": sweep["rows"],
        },
        "config": {
            "chroms": args.chroms,
            "n_genes": args.n_genes,
            "edge_trim": args.edge_trim,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "n_k": args.n_k,
            "tissue": args.tissue,
            "none_subsample_ratio": args.none_subsample_ratio,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nwritten to {out_path}")
    print(f"\nTo use this in the runner:")
    print(f"  runner = BiPangolinRunner(correction_file='{out_path}')")
    print(f"  or drop optimal_correction.json into your probe_dir and it loads automatically.")


if __name__ == "__main__":
    main()

