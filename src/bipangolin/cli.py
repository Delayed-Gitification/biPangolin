"""Command-line interface for biPangolin."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .runner import (
    BiPangolinRunner, BiPangolinResult, USABLE_LEN, TISSUE_NAMES, selftest,
)


def _print_summary(result: BiPangolinResult, top_k: int = 10) -> None:
    """Print top-K donor and acceptor predictions to stdout."""
    don_top = torch.topk(result.probe_donor, k=min(top_k, len(result)))
    acc_top = torch.topk(result.probe_acceptor, k=min(top_k, len(result)))
    print(f"\nTop {top_k} donor predictions (position, P):")
    for p, v in zip(don_top.indices.tolist(), don_top.values.tolist()):
        print(f"  pos={p:>8}  P(donor)={v:.4f}")
    print(f"\nTop {top_k} acceptor predictions (position, P):")
    for p, v in zip(acc_top.indices.tolist(), acc_top.values.tolist()):
        print(f"  pos={p:>8}  P(acceptor)={v:.4f}")


def _write_bedgraph(result: BiPangolinResult, prefix: str, chrom: str = "seq",
                    start: int = 0) -> None:
    """Write donor.bg, acceptor.bg, none.bg as bedGraph at `prefix`."""
    for name, track in [("donor", result.probe_donor),
                        ("acceptor", result.probe_acceptor),
                        ("none", result.probe_none)]:
        path = f"{prefix}.{name}.bg"
        with open(path, "w") as f:
            for i, p in enumerate(track.tolist()):
                f.write(f"{chrom}\t{start+i}\t{start+i+1}\t{p:.4f}\n")
        print(f"wrote {path}", file=sys.stderr)


def _write_four_track_per_tissue_matrix(result: BiPangolinResult, path: str) -> None:
    """Write 4 x n_tissues x L matrix as .npy."""
    matrix = result.four_track_per_tissue().detach().cpu().numpy()
    np.save(path, matrix)
    print(f"wrote {path} shape={matrix.shape}", file=sys.stderr)


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
    p_seq.add_argument("--models", default=None)
    p_seq.add_argument("--probes", default=None)
    p_seq.add_argument("--tissue", default="all_tissues",
                       choices=("all_tissues",) + TISSUE_NAMES)
    p_seq.add_argument("--out", default=None,
                       help="Prefix for bedGraph output (writes .donor.bg/.acceptor.bg/.none.bg)")
    p_seq.add_argument("--four-track-per-tissue-out", default=None,
                       help="Write 4 x n_tissues x L donor/acceptor-routed Pangolin matrix as .npy")
    p_seq.add_argument("--top", type=int, default=10, help="Top-K to print")

    # bipangolin score-region
    p_reg = sub.add_parser("score-region", help="Score a genomic region from a FASTA.")
    p_reg.add_argument("fasta")
    p_reg.add_argument("chrom")
    p_reg.add_argument("start", type=int)
    p_reg.add_argument("end", type=int)
    p_reg.add_argument("--models", default=None)
    p_reg.add_argument("--probes", default=None)
    p_reg.add_argument("--tissue", default="all_tissues",
                       choices=("all_tissues",) + TISSUE_NAMES)
    p_reg.add_argument("--out", default=None,
                       help="Prefix for bedGraph output")
    p_reg.add_argument("--four-track-per-tissue-out", default=None,
                       help="Write 4 x n_tissues x L donor/acceptor-routed Pangolin matrix as .npy")
    p_reg.add_argument("--top", type=int, default=10)

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
    p_vcf.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar")

    args = parser.parse_args(argv)

    if args.cmd == "selftest":
        selftest(args.models, args.probes, tissue=args.tissue)
        return

    runner = BiPangolinRunner(args.models, args.probes, tissue=args.tissue)

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

        if len(seq) <= USABLE_LEN:
            result = runner.score_sequence(seq)
        else:
            result = runner.score_long_sequence(seq)
        _print_summary(result, top_k=args.top)
        if args.out:
            _write_bedgraph(result, args.out, chrom=chrom, start=0)
        if args.four_track_per_tissue_out:
            _write_four_track_per_tissue_matrix(result, args.four_track_per_tissue_out)

    elif args.cmd == "score-region":
        result = runner.score_region(args.fasta, args.chrom, args.start, args.end)
        _print_summary(result, top_k=args.top)
        if args.out:
            _write_bedgraph(result, args.out, chrom=args.chrom, start=args.start)
        if args.four_track_per_tissue_out:
            _write_four_track_per_tissue_matrix(result, args.four_track_per_tissue_out)

    elif args.cmd == "score-vcf":
        n = runner.score_vcf(
            args.vcf_in, args.vcf_out,
            fasta_path=args.fasta,
            distance=args.distance,
            tissue_for_info=None if args.tissue == "all_tissues" else args.tissue,
            progress=not args.no_progress,
        )
        print(f"\nScored {n} variants -> {args.vcf_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
