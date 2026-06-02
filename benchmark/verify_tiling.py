#!/usr/bin/env python
"""Verify the no-blend tiling is correct and seamless.

The claim behind removing the triangular blend: Pangolin internally crops its
receptive-field radius (PANGOLIN_CROP = 5000) off each side, so every output
position has full +/-5000bp context and its prediction is independent of which
tile produced it. If that's true, then scoring a sequence is invariant to the
tile size (window_len) — including the degenerate "one giant window" case.

This script scores the same random sequence with several window_len values and
asserts they all agree to within a tiny tolerance. A pass means the tiling
introduces no seams and no gaps, and that switching away from the old
overlap+blend scheme did not perturb results.

Usage (on a machine with torch + weights available):
    python benchmark/verify_tiling.py
    python benchmark/verify_tiling.py --length 120000 --tissue heart \
        --windows 30000 50000 80000 --models /path/to/models --probes /path/to/probes
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from bipangolin.runner import BiPangolinRunner, PANGOLIN_CROP


def random_seq(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=n))


def tracks(result):
    """Stack every per-position output track into one array for comparison."""
    parts = [
        result.pangolin_prob.detach().cpu().numpy(),          # (T, L)
        result.probe_none.detach().cpu().numpy()[None],       # (1, L)
        result.probe_acceptor.detach().cpu().numpy()[None],
        result.probe_donor.detach().cpu().numpy()[None],
    ]
    return np.concatenate(parts, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=120_000,
                    help="Length of the random test sequence (default 120000)")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[30_000, 50_000, 80_000],
                    help="window_len values to compare (default 30k 50k 80k)")
    ap.add_argument("--tissue", default="all_tissues")
    ap.add_argument("--models", default=None)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="Max allowed abs difference between window sizes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    seq = random_seq(args.length, seed=args.seed)
    print(f"Test sequence: {len(seq)} bp, tissue={args.tissue}")

    ref = None
    ref_w = None
    max_diff_overall = 0.0
    for w in args.windows:
        usable = w - 2 * PANGOLIN_CROP
        mode = "single-window" if len(seq) <= usable else \
               f"tiled ({-(-len(seq) // usable)} tiles)"
        runner = BiPangolinRunner(args.models, args.probes, device=args.device,
                                  tissue=args.tissue, window_len=w)
        result = runner.score_long_sequence(seq) if len(seq) > usable \
            else runner.score_sequence(seq)
        arr = tracks(result)
        print(f"  window_len={w:>7} (usable={usable:>7})  {mode:<22} "
              f"output shape={arr.shape}")
        if ref is None:
            ref, ref_w = arr, w
        else:
            if arr.shape != ref.shape:
                sys.exit(f"FAIL: shape mismatch {arr.shape} vs {ref.shape}")
            d = float(np.max(np.abs(arr - ref)))
            max_diff_overall = max(max_diff_overall, d)
            print(f"      max|diff| vs window_len={ref_w}: {d:.3e}")

    print(f"\nLargest difference across all window sizes: {max_diff_overall:.3e}")
    if max_diff_overall <= args.tol:
        print(f"PASS — tiling is seamless within tol={args.tol:g}. "
              "Removing the blend did not change results.")
        return 0
    print(f"FAIL — differences exceed tol={args.tol:g}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
