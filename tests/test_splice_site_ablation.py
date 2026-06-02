"""End-to-end ablation tests across every public interface.

Strategy: take the built-in calibration sequence (which has a strong donor at
position 69 and a strong acceptor at position 163), then destroy the canonical
splice motif — the donor ``GT`` (indices 70-71) or the acceptor ``AG`` (indices
161-162) — and assert the corresponding splice site disappears while the other
site is left essentially intact.

The same ablation is exercised through all four entry points we ship:
  * the Python API          (``runner.score_sequence``)
  * FASTA / region scoring  (``runner.score_region``)
  * the command line        (``bipangolin score-seq ... --out``)
  * VCF annotation          (``runner.score_vcf`` / ``score_variant``)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from bipangolin import CALIBRATION_SEQ
from bipangolin.cli import main as cli_main


pytestmark = pytest.mark.integration


# --- Motif coordinates within CALIBRATION_SEQ (0-based) --------------------
DONOR_PROBE_POS = 69        # where the donor probe peaks
ACCEPTOR_PROBE_POS = 163    # where the acceptor probe peaks
DONOR_GT_START = 70         # canonical donor "GT" occupies indices 70-71
ACCEPTOR_AG_START = 161     # canonical acceptor "AG" occupies indices 161-162

# Confidence thresholds (the real model is far more extreme than these).
PRESENT = 0.5      # a healthy site
GONE = 0.05        # an ablated site
PRESERVED = 0.3    # the untouched neighbour must survive


def _mutate(seq: str, i: int, new: str) -> str:
    """Replace ``len(new)`` bases of ``seq`` starting at index ``i``."""
    return seq[:i] + new + seq[i + len(new):]


# Knock out each motif by overwriting its dinucleotide with "CC".
DONOR_KO_SEQ = _mutate(CALIBRATION_SEQ, DONOR_GT_START, "CC")
ACCEPTOR_KO_SEQ = _mutate(CALIBRATION_SEQ, ACCEPTOR_AG_START, "CC")

# (name, ablated_seq, killed_attr, killed_pos, survives_attr, survives_pos)
ABLATIONS = [
    ("donor", DONOR_KO_SEQ, "probe_donor", DONOR_PROBE_POS,
     "probe_acceptor", ACCEPTOR_PROBE_POS),
    ("acceptor", ACCEPTOR_KO_SEQ, "probe_acceptor", ACCEPTOR_PROBE_POS,
     "probe_donor", DONOR_PROBE_POS),
]
ABLATION_IDS = [a[0] for a in ABLATIONS]


def _weight_cli_args() -> list[str]:
    """Forward the same explicit weight dirs the integration_runner uses, so
    the CLI hits identical models instead of trying to download."""
    args: list[str] = []
    if (m := os.environ.get("BIPANGOLIN_TEST_PANGOLIN")):
        args += ["--models", m]
    if (p := os.environ.get("BIPANGOLIN_TEST_PROBES")):
        args += ["--probes", p]
    return args


def _read_bedgraph(path: str | Path) -> dict[int, float]:
    """Parse a 4-column bedGraph into {start_position: value}."""
    vals: dict[int, float] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        _chrom, start, _end, value = line.split("\t")
        vals[int(start)] = float(value)
    return vals


def _bipangolin_info(info_field: str) -> list[str]:
    """Return the pipe-split biPangolin= annotation from a VCF INFO column."""
    for kv in info_field.split(";"):
        if kv.startswith("biPangolin="):
            return kv[len("biPangolin="):].split("|")
    raise AssertionError(f"no biPangolin= field in INFO: {info_field!r}")


# ---------------------------------------------------------------------------
# 1. Python API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,ko_seq,killed_attr,killed_pos,survives_attr,survives_pos",
    ABLATIONS, ids=ABLATION_IDS)
def test_python_api_ablation(integration_runner, name, ko_seq, killed_attr,
                             killed_pos, survives_attr, survives_pos):
    wt = integration_runner.score_sequence(CALIBRATION_SEQ)
    ko = integration_runner.score_sequence(ko_seq)

    assert getattr(wt, killed_attr)[killed_pos] > PRESENT
    assert getattr(ko, killed_attr)[killed_pos] < GONE
    # The neighbouring (untouched) site must remain.
    assert getattr(ko, survives_attr)[survives_pos] > PRESERVED


# ---------------------------------------------------------------------------
# 2. FASTA / region scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,ko_seq,killed_attr,killed_pos,survives_attr,survives_pos",
    ABLATIONS, ids=ABLATION_IDS)
def test_fasta_region_ablation(integration_runner, tmp_path, name, ko_seq,
                               killed_attr, killed_pos, survives_attr,
                               survives_pos):
    wt_fa = tmp_path / f"wt_{name}.fa"
    ko_fa = tmp_path / f"ko_{name}.fa"
    wt_fa.write_text(f">chr1\n{CALIBRATION_SEQ}\n")
    ko_fa.write_text(f">chr1\n{ko_seq}\n")

    wt = integration_runner.score_region(str(wt_fa), "chr1", 0, len(CALIBRATION_SEQ))
    ko = integration_runner.score_region(str(ko_fa), "chr1", 0, len(ko_seq))

    assert getattr(wt, killed_attr)[killed_pos] > PRESENT
    assert getattr(ko, killed_attr)[killed_pos] < GONE
    assert getattr(ko, survives_attr)[survives_pos] > PRESERVED


# ---------------------------------------------------------------------------
# 3. Command line (score-seq -> bedGraph)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,ko_seq,bg_kind,killed_pos",
    [("donor", DONOR_KO_SEQ, "donor", DONOR_PROBE_POS),
     ("acceptor", ACCEPTOR_KO_SEQ, "acceptor", ACCEPTOR_PROBE_POS)],
    ids=["donor", "acceptor"])
def test_cli_score_seq_ablation(integration_runner, tmp_path, name, ko_seq,
                                bg_kind, killed_pos):
    # integration_runner is requested only so the test is skipped when weights
    # are unavailable; the CLI builds its own runner from the same weight dirs.
    wt_prefix = tmp_path / f"wt_{name}"
    ko_prefix = tmp_path / f"ko_{name}"
    common = ["--tissue", "brain", "--n-models-per-tissue", "1",
              "--raw-probes", *_weight_cli_args()]

    cli_main(["score-seq", CALIBRATION_SEQ, "--out", str(wt_prefix), *common])
    cli_main(["score-seq", ko_seq, "--out", str(ko_prefix), *common])

    wt = _read_bedgraph(f"{wt_prefix}.probe.{bg_kind}.bg")
    ko = _read_bedgraph(f"{ko_prefix}.probe.{bg_kind}.bg")

    assert wt[killed_pos] > PRESENT
    assert ko[killed_pos] < GONE


# ---------------------------------------------------------------------------
# 4. VCF annotation
# ---------------------------------------------------------------------------
def test_vcf_ablation(integration_runner, tmp_path):
    """A SNV that breaks the donor GT (or acceptor AG) should annotate a large
    donor-loss (resp. acceptor-loss) delta in the biPangolin= INFO field."""
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{CALIBRATION_SEQ}\n")

    # POS is 1-based: donor G at index 70 -> POS 71; acceptor A at 161 -> 162.
    # The control SNV at POS 66 (index 65) sits just 4 nt upstream of the donor
    # but does NOT touch the GT/AG motif, so every delta should stay ~0 — this
    # is what makes the donor/acceptor losses above *specific* to the motif
    # rather than a generic reaction to any nearby substitution.
    control_ref = CALIBRATION_SEQ[65]
    control_alt = "C" if control_ref != "C" else "A"
    vcf_in = tmp_path / "in.vcf"
    vcf_in.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t71\t.\tG\tC\t.\tPASS\t.\n"
        "chr1\t162\t.\tA\tC\t.\tPASS\t.\n"
        f"chr1\t66\t.\t{control_ref}\t{control_alt}\t.\tPASS\t.\n"
    )
    vcf_out = tmp_path / "out.vcf"

    n = integration_runner.score_vcf(
        str(vcf_in), str(vcf_out), fasta_path=str(ref),
        distance=50, progress=False)
    assert n == 3

    rows: dict[int, list[str]] = {}
    for line in vcf_out.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        rows[int(cols[1])] = _bipangolin_info(cols[7])

    # INFO order: ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL|...
    DS_AG, DS_AL, DS_DG, DS_DL = 1, 2, 3, 4
    # Donor-breaking SNV -> strong donor loss, negligible acceptor loss.
    assert float(rows[71][DS_DL]) > PRESENT
    assert float(rows[71][DS_AL]) < GONE
    # Acceptor-breaking SNV -> strong acceptor loss, negligible donor loss.
    assert float(rows[162][DS_AL]) > PRESENT
    assert float(rows[162][DS_DL]) < GONE
    # Control SNV 4 nt from the donor but off-motif -> nothing moves.
    assert all(float(rows[66][i]) < GONE for i in (DS_AG, DS_AL, DS_DG, DS_DL))
