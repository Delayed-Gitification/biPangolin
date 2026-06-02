from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bipangolin._variants import (
    VariantScore,
    _align_for_delta,
    _build_ref_alt_sequences,
    _iter_vcf,
)


class _Slice:
    def __init__(self, seq: str):
        self.seq = seq


class _Chrom:
    def __init__(self, seq: str):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _Slice(self.seq[item])
        return self.seq[item]


class _Fasta(dict):
    def __getitem__(self, chrom):
        return super().__getitem__(chrom)


def test_build_ref_alt_sequences_pads_near_chromosome_edges():
    fasta = _Fasta(chr1=_Chrom("ACGTACGT"))

    ref_seq, alt_seq, offset = _build_ref_alt_sequences(
        fasta, "chr1", 2, "C", "TT", half_window=4
    )

    assert ref_seq == "NNNACGTAC"
    assert alt_seq == "NNNATTGTAC"
    assert offset == 4


def test_build_ref_alt_sequences_rejects_reference_mismatch():
    fasta = _Fasta(chr1=_Chrom("ACGTACGT"))

    with pytest.raises(ValueError, match="REF mismatch"):
        _build_ref_alt_sequences(fasta, "chr1", 2, "G", "T", half_window=2)


def test_align_for_delta_snv_keeps_tracks_unchanged():
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    alt = torch.tensor([5.0, 6.0, 7.0, 8.0])

    ref_a, alt_a, offset = _align_for_delta(ref, alt, var_offset=1, ref_len=1, alt_len=1)

    assert offset == 1
    assert torch.equal(ref_a, ref)
    assert torch.equal(alt_a, alt)


def test_align_for_delta_insertion_pads_reference_at_variant():
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    alt = torch.tensor([1.0, 20.0, 21.0, 3.0, 4.0])

    ref_a, alt_a, _ = _align_for_delta(ref, alt, var_offset=1, ref_len=1, alt_len=2)

    assert torch.equal(ref_a, torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0]))
    assert torch.equal(alt_a, torch.tensor([1.0, 20.0, 21.0, 3.0, 4.0]))


def test_align_for_delta_deletion_pads_alternate_at_variant():
    ref = torch.tensor([1.0, 20.0, 21.0, 3.0, 4.0])
    alt = torch.tensor([1.0, 2.0, 3.0, 4.0])

    ref_a, alt_a, _ = _align_for_delta(ref, alt, var_offset=1, ref_len=2, alt_len=1)

    assert torch.equal(ref_a, torch.tensor([1.0, 20.0, 21.0, 3.0, 4.0]))
    assert torch.equal(alt_a, torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0]))


def test_variant_score_info_string_formats_core_and_tissue_blocks():
    score = VariantScore(
        chrom="chr1",
        pos=10,
        ref="A",
        alt="G",
        ds_ag=0.12345,
        ds_al=0.0,
        ds_dg=0.98765,
        ds_dl=0.11111,
        dp_ag=-3,
        dp_al=0,
        dp_dg=4,
        dp_dl=-1,
        pangolin_per_tissue={
            "heart": {"ds_gain": 0.2, "ds_loss": 0.3, "dp_gain": 1, "dp_loss": -1},
            "brain": {"ds_gain": 0.4, "ds_loss": 0.5, "dp_gain": 2, "dp_loss": -2},
        },
    )

    assert score.to_info_string() == (
        "G|0.123|0.000|0.988|0.111|-3|0|4|-1"
        "|heart:0.200:0.300:1:-1|brain:0.400:0.500:2:-2"
    )
    assert score.to_info_string(tissue="brain").endswith("|brain:0.400:0.500:2:-2")
    assert "heart:" not in score.to_info_string(tissue="brain")


def test_iter_vcf_splits_multiallelic_records(tmp_path: Path):
    vcf = tmp_path / "input.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t5\t.\tA\tC,G\t.\tPASS\t.\n"
    )

    rows = list(_iter_vcf(vcf))

    assert len(rows) == 2
    assert rows[0][0] == ["##fileformat=VCFv4.2", "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
    assert rows[0][5] == "C"
    assert rows[1][0] == []
    assert rows[1][5] == "G"
