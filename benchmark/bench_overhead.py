#!/usr/bin/env python
"""Quick runtime benchmark for biPangolin overhead.

Compares:
  1. plain frozen Pangolin P-model forward passes, using the same model files
     selected by BiPangolinRunner; and
  2. the same Pangolin forward passes with biPangolin's activation hooks attached
     but without running probes; and
  3. full biPangolin scoring, including activation hooks, probe forward passes,
     result assembly, and routed-track construction by default.

Model loading and first-use cache construction are intentionally kept outside
the timed region. This is meant as a small sanity benchmark, not a full
throughput study.

Example:
    conda run -n pangolin python benchmark/bench_overhead.py
    conda run -n pangolin python benchmark/bench_overhead.py --lengths 200 1000 5000
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bipangolin.runner import (  # noqa: E402
    BiPangolinRunner,
    PANGOLIN_CROP,
    PROB_CHANNEL_PER_TISSUE,
    load_frozen_pangolin,
    one_hot_encode,
)


def random_seq(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def plain_pangolin_score(seq: str, runner: BiPangolinRunner, plain_models) -> torch.Tensor:
    """Run the selected Pangolin P-tuned models without hooks or probes."""
    length = len(seq)
    padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
    seq_t = one_hot_encode(padded).unsqueeze(0).to(runner.device)

    sums = {
        tissue_idx: torch.zeros(length, device=runner.device)
        for tissue_idx in runner.tissues_present
    }
    counts = {tissue_idx: 0 for tissue_idx in runner.tissues_present}

    for model, tissue_idx in plain_models:
        out = model(seq_t)[0]
        channel = PROB_CHANNEL_PER_TISSUE[tissue_idx]
        sums[tissue_idx] += out[channel, :length]
        counts[tissue_idx] += 1

    return torch.stack([
        (sums[tissue_idx] / max(counts[tissue_idx], 1)).cpu()
        for tissue_idx in runner.tissues_present
    ])


@torch.no_grad()
def hooked_pangolin_score(seq: str, runner: BiPangolinRunner) -> torch.Tensor:
    """Run biPangolin's cached Pangolin models with hooks, but without probes."""
    length = len(seq)
    padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
    seq_t = one_hot_encode(padded).unsqueeze(0).to(runner.device)

    sums = {
        tissue_idx: torch.zeros(length, device=runner.device)
        for tissue_idx in runner.tissues_present
    }
    counts = {tissue_idx: 0 for tissue_idx in runner.tissues_present}

    for model, _probe, _handles, _cfg, tissue_idx in runner._iter_pairs():
        out = model(seq_t)[0]
        channel = PROB_CHANNEL_PER_TISSUE[tissue_idx]
        sums[tissue_idx] += out[channel, :length]
        counts[tissue_idx] += 1

    return torch.stack([
        (sums[tissue_idx] / max(counts[tissue_idx], 1)).cpu()
        for tissue_idx in runner.tissues_present
    ])


@torch.no_grad()
def bipangolin_score(seq: str, runner: BiPangolinRunner, include_routing: bool) -> torch.Tensor:
    result = runner.score_sequence(seq)
    if include_routing:
        routed, _ = result.routed_tracks()
        return routed
    return result.pangolin_prob


def time_call(fn, seq: str, device: torch.device) -> float:
    sync(device)
    t0 = time.perf_counter()
    _ = fn(seq)
    sync(device)
    return time.perf_counter() - t0


def print_times(label: str, times: list[float]) -> None:
    print(
        f"    {label:<16} mean={statistics.mean(times):.4f}s  "
        f"median={statistics.median(times):.4f}s"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", default=[200, 1000, 5000])
    parser.add_argument("--num-seqs", type=int, default=30,
                        help="Random sequences per length")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tissue", default="brain")
    parser.add_argument("--n-models-per-tissue", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--models", default=None)
    parser.add_argument("--probes", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="Time score_sequence only, without routed_tracks construction.",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    runner = BiPangolinRunner(
        pangolin_model_dir=args.models,
        probe_dir=args.probes,
        device=args.device,
        tissue=args.tissue,
        n_models_per_tissue=args.n_models_per_tissue,
    )

    plain_models = [
        (load_frozen_pangolin(pangolin_path, runner.device), tissue_idx)
        for pangolin_path, _probe_path, tissue_idx in runner._pair_specs
    ]
    print(
        f"Benchmarking on {runner.device}: tissue={args.tissue}, "
        f"models={len(plain_models)}, routing={not args.no_routing}"
    )

    warmup = random_seq(max(50, min(args.lengths)), rng)
    print("Warming models...")
    _ = plain_pangolin_score(warmup, runner, plain_models)
    _ = hooked_pangolin_score(warmup, runner)
    _ = bipangolin_score(warmup, runner, include_routing=not args.no_routing)

    for length in args.lengths:
        seqs = [random_seq(length, rng) for _ in range(args.num_seqs)]
        print(f"\nLength {length:,} bp ({args.num_seqs} random sequences)")
        plain_fn = lambda seq: plain_pangolin_score(seq, runner, plain_models)
        hooked_fn = lambda seq: hooked_pangolin_score(seq, runner)
        bi_fn = lambda seq: bipangolin_score(seq, runner, include_routing=not args.no_routing)
        plain = []
        hooked = []
        bi = []
        for i, seq in enumerate(seqs):
            # Alternate order to reduce one-sided cache/frequency effects.
            if i % 3 == 0:
                plain.append(time_call(plain_fn, seq, runner.device))
                hooked.append(time_call(hooked_fn, seq, runner.device))
                bi.append(time_call(bi_fn, seq, runner.device))
            elif i % 3 == 1:
                hooked.append(time_call(hooked_fn, seq, runner.device))
                bi.append(time_call(bi_fn, seq, runner.device))
                plain.append(time_call(plain_fn, seq, runner.device))
            else:
                bi.append(time_call(bi_fn, seq, runner.device))
                plain.append(time_call(plain_fn, seq, runner.device))
                hooked.append(time_call(hooked_fn, seq, runner.device))
        print_times("plain Pangolin", plain)
        print_times("Pangolin + hooks", hooked)
        print_times("biPangolin", bi)
        ratio_plain = statistics.mean(bi) / statistics.mean(plain)
        ratio_hooked = statistics.mean(bi) / statistics.mean(hooked)
        overhead_plain = (ratio_plain - 1.0) * 100.0
        overhead_hooked = (ratio_hooked - 1.0) * 100.0
        print(
            f"    bi/plain={ratio_plain:.2f}x ({overhead_plain:+.1f}%)  "
            f"bi/hooked={ratio_hooked:.2f}x ({overhead_hooked:+.1f}%)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
