from __future__ import annotations

import gzip
from pathlib import Path

import pytest
import torch

import bipangolin._variants as variants
from bipangolin._variants import (
    VariantScore,
    _align_for_delta,
    _build_ref_alt_sequences,
    _iter_vcf,
    score_vcf,
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


def test_score_vcf_adds_header_preserves_info_and_annotates_multiallelics(
    tmp_path: Path, monkeypatch
):
    calls = []

    def fake_score_variant(runner, fasta, chrom, pos, ref, alt, distance=50):
        calls.append((chrom, pos, ref, alt, distance))
        return VariantScore(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            ds_ag=0.1 if alt == "C" else 0.2,
            ds_al=0.3,
            ds_dg=0.4,
            ds_dl=0.5,
            dp_ag=-1,
            dp_al=0,
            dp_dg=1,
            dp_dl=2,
            pangolin_per_tissue={
                "heart": {"ds_gain": 0.6, "ds_loss": 0.7, "dp_gain": 3, "dp_loss": 4},
                "brain": {"ds_gain": 0.8, "ds_loss": 0.9, "dp_gain": 5, "dp_loss": 6},
            },
        )

    monkeypatch.setattr(variants, "score_variant", fake_score_variant)

    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1\nAAAAAA\n")
    vcf_in = tmp_path / "input.vcf"
    vcf_in.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\tC,G\t.\tPASS\tEXISTING=1\n"
    )
    vcf_out = tmp_path / "output.vcf"

    n = score_vcf(
        runner=object(),
        vcf_in=vcf_in,
        vcf_out=vcf_out,
        fasta_path=fasta,
        distance=17,
        tissue_for_info="brain",
        progress=False,
    )

    lines = vcf_out.read_text().splitlines()
    assert n == 2
    assert calls == [("chr1", 2, "A", "C", 17), ("chr1", 2, "A", "G", 17)]
    assert lines[1].startswith("##INFO=<ID=biPangolin")
    assert lines[2].startswith("#CHROM")
    row = lines[3].split("\t")
    assert row[7].startswith("EXISTING=1;biPangolin=")
    assert "C|0.100|0.300|0.400|0.500|-1|0|1|2|brain:0.800:0.900:5:6" in row[7]
    assert "G|0.200|0.300|0.400|0.500|-1|0|1|2|brain:0.800:0.900:5:6" in row[7]
    assert "heart:" not in row[7]


def test_score_vcf_skips_symbolic_and_star_alleles_without_scoring(
    tmp_path: Path, monkeypatch
):
    def fail_score_variant(*args, **kwargs):
        raise AssertionError("symbolic alleles should not be scored")

    monkeypatch.setattr(variants, "score_variant", fail_score_variant)

    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1\nAAAAAA\n")
    vcf_in = tmp_path / "input.vcf"
    vcf_in.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\t<DEL>,*\t.\tPASS\t.\n"
    )
    vcf_out = tmp_path / "output.vcf"

    n = score_vcf(object(), vcf_in, vcf_out, fasta, progress=False)

    assert n == 0
    row = [line for line in vcf_out.read_text().splitlines() if not line.startswith("#")][0]
    assert row.split("\t")[7] == "biPangolin=<DEL>|.|.|.|.|.|.|.|.,*|.|.|.|.|.|.|.|."


def test_score_vcf_reads_and_writes_gzip(tmp_path: Path, monkeypatch):
    def fake_score_variant(runner, fasta, chrom, pos, ref, alt, distance=50):
        return VariantScore(chrom=chrom, pos=pos, ref=ref, alt=alt, ds_ag=1.0)

    monkeypatch.setattr(variants, "score_variant", fake_score_variant)

    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1\nAAAAAA\n")
    vcf_in = tmp_path / "input.vcf.gz"
    with gzip.open(vcf_in, "wt") as fh:
        fh.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t2\t.\tA\tC\t.\tPASS\t.\n"
        )
    vcf_out = tmp_path / "output.vcf.gz"

    n = score_vcf(object(), vcf_in, vcf_out, fasta, progress=False)

    assert n == 1
    with gzip.open(vcf_out, "rt") as fh:
        text = fh.read()
    assert "##INFO=<ID=biPangolin" in text
    assert "biPangolin=C|1.000|0.000|0.000|0.000|0|0|0|0" in text
