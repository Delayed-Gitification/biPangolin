"""
bench_score.py — slow, run once.

For each + strand gene on the held-out test chromosomes (chr1, chr9 by default,
matching `train_probes.py`'s TEST_CHROMS), score the gene region with:
  - biPangolin (correction_k=1.0, tissue="all_tissues") — raw probe + Pangolin
    P(spliced) and PSI per tissue
  - SpliceAI ensemble in a subprocess (avoids TF / torch crashes)

Per-position scores are written to one Parquet per chromosome:
    {out_dir}/{chrom}.parquet

Schema (float16 except where noted):
    chrom         str
    gene_id       str
    pos           int32   (0-based genomic position, forward strand)
    label         int8    (0 = none, 1 = acceptor, 2 = donor)
    probe_none    f16
    probe_acc     f16
    probe_don     f16
    pangolin_p_{tissue}   f16    × 4 tissues
    pangolin_psi_{tissue} f16    × 4 tissues   (only if --use-psi-models)
    spliceai_acc  f16
    spliceai_don  f16

Labels come from the same TSS / TTS-aware GTF parser used in train_probes.py
(first-exon acceptors and last-exon donors excluded; conflicting positions
across transcripts dropped).

Typical usage:
    python benchmark/bench_score.py \\
        --fasta  data/GRCh38.primary_assembly.genome.fa \\
        --gtf    data/gencode.v47.basic.annotation.gtf \\
        --out    bench_scores/ \\
        --chroms chr1 chr9 \\
        [--device auto] [--skip-spliceai] [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

# Prefer the in-repo bipangolin source over any pip-installed version on
# PYTHONPATH. This script lives at <repo>/benchmark/bench_score.py, so the
# package source lives at <repo>/src/bipangolin/. Inserting that path first
# guarantees we get THIS repo's runner.py (with use_psi_models etc.) and the
# probes shipped at <repo>/src/bipangolin/data/probes/, never the older
# installed copy.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_SRC = _REPO_ROOT / "src"
if _LOCAL_SRC.is_dir():
    sys.path.insert(0, str(_LOCAL_SRC))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyfastx
import torch

from bipangolin import BiPangolinRunner
from bipangolin.runner import TISSUE_NAMES
import bipangolin as _bp
print(f"using bipangolin from: {Path(_bp.__file__).parent}", file=sys.stderr)


NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2
GENE_FLANK = 5000  # matches train_probes.py


# ---------------------------------------------------------------------------
# GTF parsing — first/last-exon-aware (mirrors train_probes.py.parse_gtf)
# ---------------------------------------------------------------------------

def _extract_attr(attrs, key):
    needle = key + ' "'
    i = attrs.find(needle)
    if i < 0:
        return None
    j = attrs.find('"', i + len(needle))
    return attrs[i + len(needle):j] if j > 0 else None


def parse_gtf(gtf_path, chroms):
    """Return (sites, genes).

    sites: dict[chrom] -> dict[pos_0based] -> ACC_CLASS | DON_CLASS
    genes: dict[chrom] -> list[(gene_id, set_of_all_site_positions)]

    + strand only. First-exon acceptors and last-exon donors excluded.
    Positions with conflicting labels across transcripts are dropped.
    """
    chroms = set(chroms)
    transcript_exons = {}
    transcript_gene = {}

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
            transcript_id = _extract_attr(attrs, "transcript_id")
            gene_id = _extract_attr(attrs, "gene_id")
            if not transcript_id or not gene_id:
                continue
            key = (chrom, transcript_id)
            transcript_exons.setdefault(key, []).append((int(start), int(end)))
            transcript_gene[transcript_id] = gene_id

    raw = {}
    gene_sites = {}

    for (chrom, transcript_id), exons in transcript_exons.items():
        exons_sorted = sorted(exons, key=lambda e: e[0])
        gene_id = transcript_gene[transcript_id]
        chrom_raw = raw.setdefault(chrom, {})
        chrom_genes = gene_sites.setdefault(chrom, {})
        site_set = chrom_genes.setdefault(gene_id, set())

        for i, (start, end) in enumerate(exons_sorted):
            acc_pos = start - 1
            don_pos = end - 1
            is_first = (i == 0)
            is_last = (i == len(exons_sorted) - 1)

            site_set.add(acc_pos)
            site_set.add(don_pos)

            if not is_first:
                prev = chrom_raw.get(acc_pos)
                if prev is None:
                    chrom_raw[acc_pos] = ACC_CLASS
                elif prev != ACC_CLASS:
                    chrom_raw[acc_pos] = "CONFLICT"
            if not is_last:
                prev = chrom_raw.get(don_pos)
                if prev is None:
                    chrom_raw[don_pos] = DON_CLASS
                elif prev != DON_CLASS:
                    chrom_raw[don_pos] = "CONFLICT"

    sites = {chrom: {p: c for p, c in d.items() if c != "CONFLICT"}
             for chrom, d in raw.items()}
    genes = {chrom: list(d.items()) for chrom, d in gene_sites.items()}
    n_sites = sum(len(d) for d in sites.values())
    n_genes = sum(len(d) for d in genes.values())
    print(f"  parsed {n_sites:,} clean splice sites in {n_genes:,} + strand genes "
          f"across {sorted(sites)}")
    return sites, genes


# ---------------------------------------------------------------------------
# SpliceAI subprocess
# ---------------------------------------------------------------------------

class SpliceAIWorker:
    """Persistent SpliceAI subprocess. Loads models once, scores many sequences.

    Usage:
        with SpliceAIWorker(python_exe=...) as w:
            acc, don = w.predict(seq)

    Communicates via line-delimited JSON over the worker's stdin/stdout.
    The worker's stderr is forwarded to this process's stderr so we see
    real tracebacks rather than just CalledProcessError.
    """

    def __init__(self, python_exe=None, context=10000, worker_script=None):
        if worker_script is None:
            worker_script = str(Path(__file__).with_name("spliceai_worker.py"))
        self.worker_script = worker_script
        self.python_exe = python_exe or sys.executable
        self.context = context
        self.proc = None
        self._stderr_thread = None

    def __enter__(self):
        import threading
        print(f"[spliceai] starting persistent worker: {self.python_exe} {self.worker_script}")
        self.proc = subprocess.Popen(
            [self.python_exe, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Forward worker stderr to ours so tracebacks are visible.
        def _pump_stderr(stream):
            for line in stream:
                sys.stderr.write(f"[spliceai_worker] {line}")
                sys.stderr.flush()
        self._stderr_thread = threading.Thread(
            target=_pump_stderr, args=(self.proc.stderr,), daemon=True)
        self._stderr_thread.start()

        # Wait for READY
        line = self.proc.stdout.readline()
        if not line or line.strip() != "READY":
            self._die(f"worker did not report READY, got: {line!r}")
        print("[spliceai] worker READY")
        return self

    def predict(self, seq):
        """Return (acceptor, donor) float32 arrays. Raises RuntimeError on failure."""
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("SpliceAI worker is not running")

        seq_fd, seq_path = tempfile.mkstemp(suffix=".seq.txt")
        out_fd, out_path = tempfile.mkstemp(suffix=".npz")
        os.close(seq_fd); os.close(out_fd)
        try:
            with open(seq_path, "w") as f:
                f.write(seq)
            req = {"seq_path": seq_path, "out_path": out_path, "context": self.context}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            resp_line = self.proc.stdout.readline()
            if not resp_line:
                self._die("worker closed stdout unexpectedly")
            resp = json.loads(resp_line)
            if resp.get("status") != "ok":
                raise RuntimeError(f"SpliceAI worker error: {resp.get('msg')}")
            data = np.load(out_path)
            return data["acceptor"].astype(np.float32), data["donor"].astype(np.float32)
        finally:
            for p in (seq_path, out_path):
                if os.path.exists(p):
                    os.unlink(p)

    def _die(self, msg):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        raise RuntimeError(msg)

    def __exit__(self, exc_type, exc, tb):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False


# ---------------------------------------------------------------------------
# Scoring orchestration
# ---------------------------------------------------------------------------

def score_gene(runner, fasta, chrom, gene_id, site_set, chrom_sites,
               spliceai_worker):
    """Score one gene; return a dict of arrays per column (or None if skipped)."""
    if not site_set:
        return None
    chrom_len = len(fasta[chrom])
    region_start = max(0, min(site_set) - GENE_FLANK)
    region_end = min(chrom_len, max(site_set) + GENE_FLANK)
    if region_end <= region_start:
        return None

    seq = fasta[chrom][region_start:region_end].seq.upper()
    L = len(seq)
    if L == 0:
        return None

    # biPangolin (raw, ensembled across all 12 model+probe pairs;
    # optionally also per-tissue probe outputs if runner was constructed with
    # per_tissue_probes=True)
    try:
        from bipangolin.runner import score_sequence_or_long_sequence
        result = score_sequence_or_long_sequence(runner, seq)
    except Exception as e:
        print(f"  [skip] biPangolin failed on {chrom}:{region_start}-{region_end}: {e}")
        return None

    probe_none = result.probe_none.detach().cpu().numpy()
    probe_acc = result.probe_acceptor.detach().cpu().numpy()
    probe_don = result.probe_donor.detach().cpu().numpy()
    pangolin_p = result.pangolin_prob.detach().cpu().numpy()    # (T, L)
    # pangolin_psi is None unless the runner was constructed with
    # use_psi_models=True (which loads the PSI-tuned weight files).
    pangolin_psi = (
        result.pangolin_psi.detach().cpu().numpy()
        if result.pangolin_psi is not None else None
    )
    tissues = list(result.tissues)
    # Per-tissue probe outputs (P-tuned side), shape (3, n_tissues, L) or None.
    probe_per_tissue = (
        result.probe_per_tissue.detach().cpu().numpy()
        if result.probe_per_tissue is not None else None
    )
    # Per-tissue probe outputs from probes attached to PSI-tuned Pangolin
    # models. Shape (3, n_tissues, L) or None. Use these when comparing
    # against pangolin_psi_* — comparing P-tuned-probe outputs against PSI
    # mixes activations from two different fine-tunes.
    probe_per_tissue_psi = (
        result.probe_per_tissue_psi.detach().cpu().numpy()
        if result.probe_per_tissue_psi is not None else None
    )

    # SpliceAI (per-position acceptor / donor; may be shorter than L)
    if spliceai_worker is not None:
        try:
            sai_acc, sai_don = spliceai_worker.predict(seq)
        except Exception as e:
            print(f"  [skip-spliceai] {chrom}:{region_start}-{region_end}: {e}")
            sai_acc = np.full(L, np.nan, dtype=np.float32)
            sai_don = np.full(L, np.nan, dtype=np.float32)
        if len(sai_acc) != L:
            # SpliceAI returned a different-length track. Centre-align (it
            # symmetrically crops its output) and pad with NaN.
            pad = (L - len(sai_acc)) // 2
            full_acc = np.full(L, np.nan, dtype=np.float32)
            full_don = np.full(L, np.nan, dtype=np.float32)
            end = pad + len(sai_acc)
            if pad >= 0 and end <= L:
                full_acc[pad:end] = sai_acc
                full_don[pad:end] = sai_don
            sai_acc, sai_don = full_acc, full_don
    else:
        sai_acc = np.full(L, np.nan, dtype=np.float32)
        sai_don = np.full(L, np.nan, dtype=np.float32)

    # Labels — vectorise lookup from chrom_sites
    label = np.zeros(L, dtype=np.int8)
    abs_positions = np.arange(region_start, region_start + L)
    # site_set inside region:
    for site_pos in site_set:
        if region_start <= site_pos < region_start + L:
            cls = chrom_sites.get(site_pos)
            if cls is not None:
                label[site_pos - region_start] = cls

    out = {
        "chrom": np.array([chrom] * L, dtype=object),
        "gene_id": np.array([gene_id] * L, dtype=object),
        "pos": abs_positions.astype(np.int32),
        "label": label,
        "probe_none": probe_none.astype(np.float16),
        "probe_acc": probe_acc.astype(np.float16),
        "probe_don": probe_don.astype(np.float16),
        "spliceai_acc": sai_acc.astype(np.float16),
        "spliceai_don": sai_don.astype(np.float16),
    }
    for i, t in enumerate(tissues):
        out[f"pangolin_p_{t}"] = pangolin_p[i].astype(np.float16)
        if pangolin_psi is not None:
            out[f"pangolin_psi_{t}"] = pangolin_psi[i].astype(np.float16)
    if probe_per_tissue is not None:
        # probe_per_tissue: (3, n_tissues, L) — channel order none/acc/don
        for i, t in enumerate(tissues):
            out[f"probe_none_{t}"] = probe_per_tissue[0, i].astype(np.float16)
            out[f"probe_acc_{t}"] = probe_per_tissue[1, i].astype(np.float16)
            out[f"probe_don_{t}"] = probe_per_tissue[2, i].astype(np.float16)
    if probe_per_tissue_psi is not None:
        # Probe outputs from probes attached to the PSI-tuned Pangolin models.
        for i, t in enumerate(tissues):
            out[f"probe_none_psi_{t}"] = probe_per_tissue_psi[0, i].astype(np.float16)
            out[f"probe_acc_psi_{t}"]  = probe_per_tissue_psi[1, i].astype(np.float16)
            out[f"probe_don_psi_{t}"]  = probe_per_tissue_psi[2, i].astype(np.float16)
    return out


def _arrow_schema(tissues, per_tissue_probes=False, include_psi=True,
                  per_tissue_probes_psi=False):
    fields = [
        ("chrom", pa.string()),
        ("gene_id", pa.string()),
        ("pos", pa.int32()),
        ("label", pa.int8()),
        ("probe_none", pa.float16()),
        ("probe_acc", pa.float16()),
        ("probe_don", pa.float16()),
    ]
    for t in tissues:
        fields.append((f"pangolin_p_{t}", pa.float16()))
        if include_psi:
            fields.append((f"pangolin_psi_{t}", pa.float16()))
    fields += [
        ("spliceai_acc", pa.float16()),
        ("spliceai_don", pa.float16()),
    ]
    if per_tissue_probes:
        for t in tissues:
            fields.append((f"probe_none_{t}", pa.float16()))
            fields.append((f"probe_acc_{t}", pa.float16()))
            fields.append((f"probe_don_{t}", pa.float16()))
    if per_tissue_probes_psi:
        for t in tissues:
            fields.append((f"probe_none_psi_{t}", pa.float16()))
            fields.append((f"probe_acc_psi_{t}",  pa.float16()))
            fields.append((f"probe_don_psi_{t}",  pa.float16()))
    return pa.schema(fields)


def _record_batch(arrays, schema):
    cols = [pa.array(arrays[name].astype(np.float32)
                     if pa.types.is_floating(schema.field(name).type)
                     and arrays[name].dtype == np.float16
                     else arrays[name],
                     type=schema.field(name).type)
            for name in schema.names]
    return pa.RecordBatch.from_arrays(cols, schema=schema)


def write_chrom_parquet(out_path, schema, gene_outputs):
    """Stream-write per-gene arrays to a single Parquet file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), schema, compression="zstd",
                              use_dictionary=True)
    n_rows = 0
    try:
        for arrays in gene_outputs:
            if arrays is None:
                continue
            batch = _record_batch(arrays, schema)
            writer.write_batch(batch)
            n_rows += batch.num_rows
    finally:
        writer.close()
    return n_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True, help="Output directory for {chrom}.parquet files")
    ap.add_argument("--chroms", nargs="+", default=["chr1", "chr9"])
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | mps | cpu (default: auto)")
    ap.add_argument("--probe-dir",
                    default=str(_REPO_ROOT / "src" / "bipangolin" / "data" / "probes"),
                    help="Probe directory (default: <repo>/src/bipangolin/data/probes — "
                         "the freshly-trained probes in this checkout, NOT the pip-installed copy).")
    ap.add_argument("--pangolin-model-dir", default=None,
                    help="Override Pangolin weights directory")
    ap.add_argument("--skip-spliceai", action="store_true",
                    help="Skip SpliceAI inference (stores NaN columns).")
    ap.add_argument("--spliceai-python", default=None,
                    help="Python executable for SpliceAI subprocess "
                         "(default: same interpreter as this script).")
    ap.add_argument("--spliceai-context", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Score at most N genes per chrom (smoke test).")
    ap.add_argument("--per-tissue-probes", action="store_true",
                    help="Also store per-tissue probe outputs (averaged across "
                         "the 3 folds for each tissue), in addition to the "
                         "global ensemble probe. Adds 12 columns to the parquet.")
    ap.add_argument("--use-psi-models", action="store_true",
                    help="Load Pangolin's PSI-tuned weight files (final.*.[1357].3.v2) "
                         "and read PSI from those. Without this flag, pangolin_psi_* "
                         "columns are NOT written, because reading PSI from P-tuned "
                         "models gives a misleading side-output. Costs ~2x Pangolin "
                         "inference time.")
    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"device: {device}")

    # Load biPangolin once
    print(f"loading biPangolin (all_tissues, correction_k=1.0, "
          f"per_tissue_probes={args.per_tissue_probes}, "
          f"use_psi_models={args.use_psi_models})...")
    runner = BiPangolinRunner(
        pangolin_model_dir=args.pangolin_model_dir,
        probe_dir=args.probe_dir,
        device=device,
        tissue="all_tissues",
        correction_k=1.0,
        per_tissue_probes=args.per_tissue_probes,
        use_psi_models=args.use_psi_models,
    )

    print(f"parsing GTF for {args.chroms}...")
    sites, genes = parse_gtf(args.gtf, chroms=args.chroms)

    fasta = pyfastx.Fasta(args.fasta)

    # PSI-side per-tissue probe columns are written when BOTH flags are on
    # AND the user has trained probes for the PSI-tuned Pangolin models.
    per_tissue_probes_psi = (args.per_tissue_probes and args.use_psi_models
                             and getattr(runner, "_psi_has_probes", False))
    schema = _arrow_schema(list(TISSUE_NAMES),
                           per_tissue_probes=args.per_tissue_probes,
                           include_psi=args.use_psi_models,
                           per_tissue_probes_psi=per_tissue_probes_psi)
    if per_tissue_probes_psi:
        print("biPangolin: PSI-side per-tissue probe columns will be written.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spawn one persistent SpliceAI worker for the whole run (or None if skipped).
    spliceai_cm = (SpliceAIWorker(python_exe=args.spliceai_python,
                                  context=args.spliceai_context)
                   if not args.skip_spliceai else None)
    try:
        spliceai_worker = spliceai_cm.__enter__() if spliceai_cm else None

        for chrom in args.chroms:
            chrom_genes = genes.get(chrom, [])
            chrom_sites = sites.get(chrom, {})
            if not chrom_genes:
                print(f"== {chrom}: no genes, skipping ==")
                continue
            if args.limit:
                chrom_genes = chrom_genes[:args.limit]
            print(f"== {chrom}: {len(chrom_genes)} + strand genes "
                  f"({len(chrom_sites):,} clean sites) ==")

            out_path = out_dir / f"{chrom}.parquet"

            def gene_iter(chrom=chrom, chrom_genes=chrom_genes, chrom_sites=chrom_sites):
                t0 = time.time()
                for i, (gene_id, site_set) in enumerate(chrom_genes):
                    t_gene = time.time()
                    arrays = score_gene(
                        runner, fasta, chrom, gene_id, site_set, chrom_sites,
                        spliceai_worker=spliceai_worker,
                    )
                    dt = time.time() - t_gene
                    if arrays is not None:
                        print(f"  [{i+1}/{len(chrom_genes)}] {gene_id} "
                              f"L={len(arrays['pos']):,} sites_in_region={int((arrays['label']!=0).sum())} "
                              f"{dt:.1f}s")
                    else:
                        print(f"  [{i+1}/{len(chrom_genes)}] {gene_id} skipped ({dt:.1f}s)")
                    yield arrays
                print(f"  {chrom} total: {time.time()-t0:.1f}s")

            n_rows = write_chrom_parquet(out_path, schema, gene_iter())
            print(f"  wrote {out_path}  ({n_rows:,} rows)")
    finally:
        if spliceai_cm is not None:
            spliceai_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
