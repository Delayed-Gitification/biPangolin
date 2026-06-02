"""Command-line interface for biPangolin."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .runner import (
    BiPangolinRunner, BiPangolinResult, TISSUE_NAMES, selftest,
    score_sequence_or_long_sequence,
)


def _routed_summary_track(result: BiPangolinResult, prob_routed, psi_routed):
    """Pick the per-position acceptor/donor track to summarise: routed Pangolin
    P if present, else routed PSI (psi-only mode). Reduced across tissues by max
    so the top-K reflects the strongest tissue at each position. Returns
    (acceptor (L,), donor (L,), metric_name)."""
    # Summarise routed P normally; fall back to routed PSI only in psi-only mode
    # (prob is all-zero there). Guard against psi_routed being None.
    if psi_routed is not None and prob_routed.abs().sum() == 0:
        track, metric = psi_routed, "PSI"
    else:
        track, metric = prob_routed, "P"
    acc = track[0].max(dim=0).values
    don = track[1].max(dim=0).values
    return acc, don, metric


def _print_summary(result: BiPangolinResult, prob_routed, psi_routed,
                   top_k: int = 10) -> None:
    """Print top-K routed donor/acceptor Pangolin scores (not raw probe Ps)."""
    acc, don, metric = _routed_summary_track(result, prob_routed, psi_routed)
    don_top = torch.topk(don, k=min(top_k, len(result)))
    acc_top = torch.topk(acc, k=min(top_k, len(result)))
    print(f"\nTop {top_k} donor positions (routed Pangolin {metric}, max over tissues):")
    for p, v in zip(don_top.indices.tolist(), don_top.values.tolist()):
        print(f"  pos={p:>8}  donor {metric}={v:.4f}")
    print(f"\nTop {top_k} acceptor positions (routed Pangolin {metric}, max over tissues):")
    for p, v in zip(acc_top.indices.tolist(), acc_top.values.tolist()):
        print(f"  pos={p:>8}  acceptor {metric}={v:.4f}")


def _write_one_bg(path: str, track, chrom: str, start: int) -> None:
    with open(path, "w") as f:
        for i, p in enumerate(track.tolist()):
            f.write(f"{chrom}\t{start+i}\t{start+i+1}\t{p:.4f}\n")
    print(f"wrote {path}", file=sys.stderr)


def _write_routed_bedgraph(result: BiPangolinResult, prefix: str,
                           prob_routed, psi_routed,
                           chrom: str = "seq", start: int = 0,
                           write_prob: bool = True, write_psi: bool = False,
                           raw_probes: bool = False) -> None:
    """Write per-tissue routed donor/acceptor bedGraphs.

    For each tissue and metric (prob / psi) two files are written:
        {prefix}.{tissue}.{metric}.acceptor.bg
        {prefix}.{tissue}.{metric}.donor.bg
    where the value is the Pangolin metric routed into the column the probe
    chose (the other column is exactly 0). With raw_probes, the corrected probe
    acceptor/donor tracks are additionally written (single, tissue-agnostic).
    """
    tissues = result.tissues

    def _emit(track2, metric):  # track2: (2, n_tissues, L)
        for ti, tname in enumerate(tissues):
            for ci, cname in ((0, "acceptor"), (1, "donor")):
                _write_one_bg(f"{prefix}.{tname}.{metric}.{cname}.bg",
                              track2[ci, ti], chrom, start)

    if write_prob:
        _emit(prob_routed, "prob")
    if write_psi and psi_routed is not None:
        _emit(psi_routed, "psi")
    if raw_probes:
        _write_one_bg(f"{prefix}.probe.acceptor.bg", result.probe_acceptor, chrom, start)
        _write_one_bg(f"{prefix}.probe.donor.bg", result.probe_donor, chrom, start)


def _write_four_track_per_tissue_matrix(result: BiPangolinResult, path: str,
                                        double_val_floor: float = 0.01,
                                        double_val_ratio: float = 0.1) -> None:
    """Write 4 x n_tissues x L matrix as .npy (same routing as the bedGraphs)."""
    matrix = result.four_track_per_tissue(
        double_val_floor=double_val_floor,
        double_val_ratio=double_val_ratio).detach().cpu().numpy()
    np.save(path, matrix)
    print(f"wrote {path} shape={matrix.shape}", file=sys.stderr)


def _add_scoring_args(p) -> None:
    """Shared output/routing args for score-seq and score-region."""
    p.add_argument("--models", default=None)
    p.add_argument("--probes", default=None)
    p.add_argument("--tissue", default="all_tissues",
                   choices=("all_tissues",) + TISSUE_NAMES)
    p.add_argument("--out", default=None,
                   help="Prefix for routed per-tissue bedGraph output "
                        "(writes {prefix}.{tissue}.prob.{acceptor,donor}.bg)")
    p.add_argument("--psi", action="store_true",
                   help="Also load the PSI-tuned models and emit routed PSI "
                        "tracks alongside P (doubles Pangolin inference)")
    p.add_argument("--psi-only", action="store_true",
                   help="Emit ONLY routed PSI (skip the P-tuned models). "
                        "Routing uses the PSI-side probes; requires probe files "
                        "for the PSI-tuned models in --probes")
    p.add_argument("--raw-probes", action="store_true",
                   help="Additionally emit the corrected probe acceptor/donor "
                        "tracks ({prefix}.probe.{acceptor,donor}.bg)")
    p.add_argument("--double-val-floor", type=float, default=0.01,
                   help="Min corrected probe prob for the 'both columns' rule "
                        "(default 0.01)")
    p.add_argument("--double-val-ratio", type=float, default=0.1,
                   help="Min min/max ratio of acceptor vs donor probe prob for "
                        "the 'both columns' rule (default 0.1)")
    p.add_argument("--four-track-per-tissue-out", default=None,
                   help="Write 4 x n_tissues x L donor/acceptor-routed Pangolin "
                        "matrix as .npy (same routing as the bedGraphs; implies --psi)")
    p.add_argument("--n-models-per-tissue", type=int, default=3, choices=(1, 2, 3),
                   help="Folds per tissue to ensemble: 3 (default, full), or "
                        "2 / 1 for faster, lower-robustness scoring")
    p.add_argument("--top", type=int, default=10, help="Top-K to print")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bipangolin",
        description="Per-base donor/acceptor splice predictions from Pangolin.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # bipangolin selftest
    p_test = sub.add_parser("selftest", help="Run calibration sequence sanity check.")
    p_test.add_argument("--models", default=None, help="Pangolin weights dir")
    p_test.add_argument("--probes", default=None, help="Probe weights dir")
    p_test.add_argument("--tissue", default="all_tissues",
                        choices=("all_tissues",) + TISSUE_NAMES)

    # bipangolin score-seq
    p_seq = sub.add_parser("score-seq", help="Score a sequence (string or fasta).")
    p_seq.add_argument("input", help="Sequence (ACGT...) or path to FASTA")
    _add_scoring_args(p_seq)

    # bipangolin score-region
    p_reg = sub.add_parser("score-region", help="Score a genomic region from a FASTA.")
    p_reg.add_argument("fasta")
    p_reg.add_argument("chrom")
    p_reg.add_argument("start", type=int)
    p_reg.add_argument("end", type=int)
    _add_scoring_args(p_reg)

    # bipangolin score-vcf
    p_vcf = sub.add_parser("score-vcf", help="Annotate a VCF with variant effect predictions.")
    p_vcf.add_argument("vcf_in", help="Input VCF (.vcf or .vcf.gz)")
    p_vcf.add_argument("vcf_out", help="Output annotated VCF")
    p_vcf.add_argument("--fasta", required=True, help="Reference FASTA matching VCF build")
    p_vcf.add_argument("--models", default=None)
    p_vcf.add_argument("--probes", default=None)
    p_vcf.add_argument("--tissue", default="all_tissues",
                       choices=("all_tissues",) + TISSUE_NAMES,
                       help="Which tissue(s) to use. 'all_tissues' uses the full ensemble.")
    p_vcf.add_argument("--distance", type=int, default=50,
                       help="Report max delta within +/- this many nt of variant (default 50)")
    p_vcf.add_argument("--n-models-per-tissue", type=int, default=3, choices=(1, 2, 3),
                       help="Folds per tissue to ensemble: 3 (default, full), or "
                            "2 / 1 for faster, lower-robustness scoring")
    p_vcf.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar")

    args = parser.parse_args(argv)

    if args.cmd == "selftest":
        selftest(args.models, args.probes, tissue=args.tissue)
        return

    if args.cmd in ("score-seq", "score-region"):
        return _run_scoring(args)

    if args.cmd == "score-vcf":
        runner = BiPangolinRunner(args.models, args.probes, tissue=args.tissue,
                                  n_models_per_tissue=args.n_models_per_tissue)
        n = runner.score_vcf(
            args.vcf_in, args.vcf_out,
            fasta_path=args.fasta,
            distance=args.distance,
            tissue_for_info=None if args.tissue == "all_tissues" else args.tissue,
            progress=not args.no_progress,
        )
        print(f"\nScored {n} variants -> {args.vcf_out}", file=sys.stderr)


def _run_scoring(args):
    if args.psi and args.psi_only:
        sys.exit("error: --psi and --psi-only are mutually exclusive")

    if args.psi_only and args.four_track_per_tissue_out:
        print("biPangolin: WARNING — --four-track-per-tissue-out with --psi-only: "
              "the P channels (1, 3) of the matrix will be all-zero (the P-tuned "
              "models are not run in psi-only mode); only the PSI channels (0, 2) "
              "are meaningful.", file=sys.stderr)

    # PSI models are needed for --psi, --psi-only, and the four-track matrix.
    use_psi = bool(args.psi or args.psi_only or args.four_track_per_tissue_out)
    runner = BiPangolinRunner(args.models, args.probes, tissue=args.tissue,
                              use_psi_models=use_psi,
                              n_models_per_tissue=args.n_models_per_tissue)

    if args.cmd == "score-seq":
        if Path(args.input).is_file():
            try:
                import pyfastx
            except ImportError:
                sys.exit("score-seq with FASTA input requires pyfastx: pip install pyfastx")
            fa = pyfastx.Fasta(args.input)
            seqs = list(fa)
            if len(seqs) != 1:
                sys.exit(f"FASTA must contain exactly one sequence, found {len(seqs)}")
            seq = seqs[0].seq
            chrom = seqs[0].name
        else:
            seq = args.input
            chrom = "seq"
        try:
            result = score_sequence_or_long_sequence(runner, seq, psi_only=args.psi_only)
        except ValueError as e:
            sys.exit(f"error: {e}")
        start = 0
    else:  # score-region
        try:
            result = runner.score_region(args.fasta, args.chrom, args.start, args.end,
                                         psi_only=args.psi_only)
            chrom, start = args.chrom, args.start
        except ValueError as e:
            sys.exit(f"error: {e}")

    prob_routed, psi_routed = result.routed_tracks(
        double_val_floor=args.double_val_floor,
        double_val_ratio=args.double_val_ratio)

    _print_summary(result, prob_routed, psi_routed, top_k=args.top)

    if args.out:
        _write_routed_bedgraph(
            result, args.out, prob_routed, psi_routed,
            chrom=chrom, start=start,
            write_prob=not args.psi_only,
            write_psi=bool(args.psi or args.psi_only),
            raw_probes=args.raw_probes,
        )
    if args.four_track_per_tissue_out:
        _write_four_track_per_tissue_matrix(
            result, args.four_track_per_tissue_out,
            double_val_floor=args.double_val_floor,
            double_val_ratio=args.double_val_ratio)


if __name__ == "__main__":
    main()
