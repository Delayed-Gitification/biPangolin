"""Variant effect prediction for biPangolin.

Given a variant (chrom, pos, ref, alt), score the reference and alternate
sequences and report the max change in donor/acceptor probability within
a window around the variant. Output format follows the SpliceAI/Pangolin
convention:

  DS_AG / DS_AL : delta score for acceptor gain / loss
  DS_DG / DS_DL : delta score for donor gain / loss
  DP_AG ...      : positions (relative to variant) of those max deltas

Plus per-tissue Pangolin P(spliced) deltas, since those ARE tissue-specific.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

import torch


@dataclass
class VariantScore:
    """Effect of a single variant on splicing predictions.

    Position fields (DP_*) are 0-based offsets relative to the variant's
    leftmost reference base. Negative = upstream, positive = downstream.
    """
    chrom: str
    pos: int           # 1-based VCF position (POS column)
    ref: str
    alt: str

    # Probe-based delta scores (tissue-agnostic)
    ds_ag: float = 0.0   # max acceptor gain   (max positive delta)
    ds_al: float = 0.0   # max acceptor loss   (max negative delta, reported positive)
    ds_dg: float = 0.0   # max donor gain
    ds_dl: float = 0.0   # max donor loss
    dp_ag: int = 0       # position of acceptor gain
    dp_al: int = 0
    dp_dg: int = 0
    dp_dl: int = 0

    # Per-tissue Pangolin spliced-probability deltas
    pangolin_per_tissue: dict = field(default_factory=dict)
    # {tissue_name: {"ds_gain": float, "ds_loss": float,
    #                "dp_gain": int, "dp_loss": int}}

    metadata: dict = field(default_factory=dict)

    def to_info_string(self, tissue: Optional[str] = None) -> str:
        """SpliceAI-style INFO string for VCF output.

        Format:
          ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL[|TISSUE:DS_GAIN:DS_LOSS:DP_GAIN:DP_LOSS]
        """
        base = (f"{self.alt}"
                f"|{self.ds_ag:.3f}|{self.ds_al:.3f}|{self.ds_dg:.3f}|{self.ds_dl:.3f}"
                f"|{self.dp_ag}|{self.dp_al}|{self.dp_dg}|{self.dp_dl}")
        for t, d in self.pangolin_per_tissue.items():
            if tissue is not None and t != tissue:
                continue
            base += (f"|{t}:{d['ds_gain']:.3f}:{d['ds_loss']:.3f}"
                     f":{d['dp_gain']}:{d['dp_loss']}")
        return base


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def _build_ref_alt_sequences(fasta, chrom: str, pos_1based: int,
                              ref: str, alt: str, half_window: int):
    """Extract ref + alt sequences centred on the variant.

    Returns (ref_seq, alt_seq, var_offset_in_window) all as upper-case strings.
    var_offset_in_window is the index of the variant's first base within both
    sequences (same in ref and alt — they are aligned up to the variant start).

    For indels, the two sequences will have different lengths.
    """
    pos0 = pos_1based - 1
    # Verify the reference matches what's in the FASTA
    chrom_len = len(fasta[chrom])
    if pos0 + len(ref) > chrom_len:
        raise ValueError(f"Variant {chrom}:{pos_1based} {ref}>{alt} "
                         f"extends past end of {chrom} (length {chrom_len})")
    actual_ref = fasta[chrom][pos0:pos0 + len(ref)].seq.upper()
    if actual_ref != ref.upper():
        raise ValueError(
            f"REF mismatch at {chrom}:{pos_1based}: VCF says {ref!r}, "
            f"FASTA has {actual_ref!r}. Wrong genome build?")

    # Window: half_window upstream + variant + half_window downstream
    win_start = max(0, pos0 - half_window)
    win_end_ref = min(chrom_len, pos0 + len(ref) + half_window)
    upstream = fasta[chrom][win_start:pos0].seq.upper()
    downstream = fasta[chrom][pos0 + len(ref):win_end_ref].seq.upper()

    # If we're near a chromosome edge, pad with N to keep window centred
    n_upstream_missing = half_window - (pos0 - win_start)
    n_downstream_missing = half_window - (win_end_ref - (pos0 + len(ref)))
    upstream = "N" * n_upstream_missing + upstream
    downstream = downstream + "N" * n_downstream_missing

    ref_seq = upstream + ref.upper() + downstream
    alt_seq = upstream + alt.upper() + downstream
    var_offset = len(upstream)
    return ref_seq, alt_seq, var_offset


def _align_for_delta(ref_track: torch.Tensor, alt_track: torch.Tensor,
                     var_offset: int, ref_len: int, alt_len: int):
    """Align ref and alt 1D probability tracks for delta computation.

    Indels make the two sequences different lengths. We pad the shorter one
    with zeros at the variant site so positions upstream of the variant align
    by reference coordinate, and positions downstream align by reference
    coordinate after accounting for the length difference.

    Returns (ref_aligned, alt_aligned) of the same length, where index i
    corresponds to a position on the reference (negative i => upstream of
    variant, 0 => first base of variant on ref, positive i => downstream).
    """
    # Upstream segment is identical-length in both
    up_ref = ref_track[:var_offset]
    up_alt = alt_track[:var_offset]

    # Variant region: ref has ref_len bases, alt has alt_len. We use the longer
    # one as the canonical span; the other is N-padded so deltas show as
    # losses (alt - ref negative) where the ref had signal.
    var_span = max(ref_len, alt_len)
    var_ref = torch.zeros(var_span, dtype=ref_track.dtype)
    var_alt = torch.zeros(var_span, dtype=alt_track.dtype)
    var_ref[:ref_len] = ref_track[var_offset:var_offset + ref_len]
    var_alt[:alt_len] = alt_track[var_offset:var_offset + alt_len]

    # Downstream segment
    down_ref = ref_track[var_offset + ref_len:]
    down_alt = alt_track[var_offset + alt_len:]
    down_len = min(len(down_ref), len(down_alt))
    down_ref = down_ref[:down_len]
    down_alt = down_alt[:down_len]

    ref_a = torch.cat([up_ref, var_ref, down_ref])
    alt_a = torch.cat([up_alt, var_alt, down_alt])
    return ref_a, alt_a, var_offset


# ---------------------------------------------------------------------------
# Single-variant scoring
# ---------------------------------------------------------------------------

def score_variant(runner, fasta, chrom: str, pos: int, ref: str, alt: str,
                  distance: int = 50) -> VariantScore:
    """Score a single variant.

    Args:
        runner: a BiPangolinRunner instance
        fasta: open pyfastx.Fasta object
        chrom, pos, ref, alt: VCF-style coordinates (pos is 1-based)
        distance: report max delta within ±distance of the variant (default 50)

    Returns a VariantScore.
    """
    # Use a wide enough half-window that a single forward pass is enough.
    # USABLE_LEN = 10000, so half_window = 5000 - max(len(ref), len(alt)) // 2
    # gives us room. We just use score_long_sequence to handle any size.
    from .runner import score_sequence_or_long_sequence
    ref_seq, alt_seq, var_offset = _build_ref_alt_sequences(
        fasta, chrom, pos, ref, alt, half_window=5000)

    ref_result = score_sequence_or_long_sequence(runner, ref_seq)
    alt_result = score_sequence_or_long_sequence(runner, alt_seq)

    # --- Probe deltas (tissue-agnostic) ---
    ref_acc, alt_acc, _ = _align_for_delta(
        ref_result.probe_acceptor, alt_result.probe_acceptor,
        var_offset, len(ref), len(alt))
    ref_don, alt_don, _ = _align_for_delta(
        ref_result.probe_donor, alt_result.probe_donor,
        var_offset, len(ref), len(alt))

    # Restrict to ±distance window centred on variant
    lo = max(0, var_offset - distance)
    hi = min(len(ref_acc), var_offset + max(len(ref), len(alt)) + distance)

    acc_delta = (alt_acc - ref_acc)[lo:hi]
    don_delta = (alt_don - ref_don)[lo:hi]

    score = VariantScore(chrom=chrom, pos=pos, ref=ref, alt=alt,
                         metadata={"distance": distance,
                                   "n_pairs": len(runner._pair_specs)})

    # Acceptor gain/loss
    score.ds_ag = float(max(acc_delta.max().item(), 0.0))
    score.ds_al = float(max(-acc_delta.min().item(), 0.0))
    score.dp_ag = int(acc_delta.argmax().item()) + lo - var_offset
    score.dp_al = int(acc_delta.argmin().item()) + lo - var_offset

    # Donor gain/loss
    score.ds_dg = float(max(don_delta.max().item(), 0.0))
    score.ds_dl = float(max(-don_delta.min().item(), 0.0))
    score.dp_dg = int(don_delta.argmax().item()) + lo - var_offset
    score.dp_dl = int(don_delta.argmin().item()) + lo - var_offset

    # --- Per-tissue Pangolin deltas ---
    for ti, tname in enumerate(ref_result.tissues):
        ref_p, alt_p, _ = _align_for_delta(
            ref_result.pangolin_prob[ti], alt_result.pangolin_prob[ti],
            var_offset, len(ref), len(alt))
        delta = (alt_p - ref_p)[lo:hi]
        score.pangolin_per_tissue[tname] = {
            "ds_gain": float(max(delta.max().item(), 0.0)),
            "ds_loss": float(max(-delta.min().item(), 0.0)),
            "dp_gain": int(delta.argmax().item()) + lo - var_offset,
            "dp_loss": int(delta.argmin().item()) + lo - var_offset,
        }

    return score


# ---------------------------------------------------------------------------
# VCF I/O — minimal manual parser, no pysam dependency
# ---------------------------------------------------------------------------

def _iter_vcf(vcf_path: Path) -> Iterator[tuple]:
    """Yield (header_lines, chrom, pos, _id, ref, alt, qual, filt, info)
    for each variant. Header lines accumulate before the first variant.

    Multi-allelic ALT are split into separate records.
    """
    header = []
    opener = (lambda p: __import__("gzip").open(p, "rt")) \
             if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                header.append(line)
                continue
            parts = line.split("\t")
            chrom, pos, vid, ref, alts, qual, filt, info = parts[:8]
            for alt in alts.split(","):
                yield header, chrom, int(pos), vid, ref, alt, qual, filt, info
            header = []   # only emit header on the first record


def score_vcf(runner, vcf_in: Union[str, Path], vcf_out: Union[str, Path],
              fasta_path: Union[str, Path], distance: int = 50,
              tissue_for_info: Optional[str] = None,
              progress: bool = True) -> int:
    """Annotate a VCF with biPangolin variant effect predictions.

    Adds an INFO field `biPangolin=` with format:
      ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL[|TISSUE:DS_GAIN:DS_LOSS:DP_GAIN:DP_LOSS]...

    Multi-allelic records are split internally and reassembled in output.

    Returns number of variants scored.
    """
    try:
        import pyfastx
    except ImportError as e:
        raise ImportError("score_vcf requires pyfastx: pip install pyfastx") from e

    fasta = pyfastx.Fasta(str(fasta_path))
    vcf_in = Path(vcf_in)
    vcf_out = Path(vcf_out)

    info_header = (
        '##INFO=<ID=biPangolin,Number=.,Type=String,Description='
        '"biPangolin splice predictions: '
        'ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL'
        '[|TISSUE:DS_GAIN:DS_LOSS:DP_GAIN:DP_LOSS]">'
    )

    # Collect per-line annotations grouped by VCF line, since multi-allelics
    # split into multiple records but we want to write one VCF row per line.
    pending = {}    # line_id -> {"line": str, "annots": [str, ...]}
    n_scored = 0

    # First pass: write header + score variants
    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **k): return x
    else:
        def tqdm(x, **k): return x

    out_fh = open(vcf_out, "w")
    written_header = False

    try:
        opener = (lambda p: __import__("gzip").open(p, "rt")) \
                 if str(vcf_in).endswith(".gz") else open
        with opener(vcf_in) as fh:
            lines = fh.readlines()

        # Write headers, injecting our INFO definition before #CHROM
        for line in lines:
            if line.startswith("##"):
                out_fh.write(line)
            elif line.startswith("#CHROM"):
                out_fh.write(info_header + "\n")
                out_fh.write(line)
                written_header = True
                break

        # Score variants
        annots_by_line_idx = {}
        variant_lines = [(i, ln) for i, ln in enumerate(lines)
                         if not ln.startswith("#")]

        for line_idx, line in tqdm(variant_lines, desc="Scoring variants"):
            parts = line.rstrip("\n").split("\t")
            chrom, pos, vid, ref, alts, qual, filt, info = parts[:8]
            line_annots = []
            for alt in alts.split(","):
                if alt in (".", "*", ""):
                    line_annots.append(f"{alt}|.|.|.|.|.|.|.|.")
                    continue
                try:
                    score = score_variant(runner, fasta, chrom, int(pos), ref, alt,
                                          distance=distance)
                    line_annots.append(score.to_info_string(tissue=tissue_for_info))
                    n_scored += 1
                except Exception as e:
                    print(f"  warning: failed on {chrom}:{pos} {ref}>{alt}: {e}",
                          file=sys.stderr)
                    line_annots.append(f"{alt}|.|.|.|.|.|.|.|.")
            annots_by_line_idx[line_idx] = line_annots

        # Write variant lines with annotation appended to INFO
        for line_idx, line in variant_lines:
            parts = line.rstrip("\n").split("\t")
            info_field = parts[7] if parts[7] != "." else ""
            annot_str = "biPangolin=" + ",".join(annots_by_line_idx[line_idx])
            new_info = (info_field + ";" + annot_str) if info_field else annot_str
            parts[7] = new_info
            out_fh.write("\t".join(parts) + "\n")

    finally:
        out_fh.close()

    return n_scored
