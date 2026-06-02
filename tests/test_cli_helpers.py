from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bipangolin import cli
from bipangolin.runner import BiPangolinResult


def _result():
    return BiPangolinResult(
        pangolin_prob=torch.tensor([[0.1, 0.2, 0.3]]),
        pangolin_psi=None,
        probe_none=torch.tensor([0.8, 0.8, 0.8]),
        probe_acceptor=torch.tensor([0.9, 0.1, 0.5]),
        probe_donor=torch.tensor([0.1, 0.9, 0.5]),
        tissues=("brain",),
        metadata={},
    )


def test_routed_summary_prefers_prob_tracks():
    prob = torch.tensor(
        [
            [[0.1, 0.0, 0.3]],  # acceptor
            [[0.0, 0.2, 0.3]],  # donor
        ]
    )
    psi = torch.tensor(
        [
            [[0.9, 0.0, 0.9]],
            [[0.0, 0.9, 0.9]],
        ]
    )

    acc, don, metric = cli._routed_summary_track(_result(), prob, psi)

    assert metric == "P"
    assert torch.equal(acc, torch.tensor([0.1, 0.0, 0.3]))
    assert torch.equal(don, torch.tensor([0.0, 0.2, 0.3]))


def test_routed_summary_falls_back_to_psi_for_psi_only():
    prob = torch.zeros(2, 1, 3)
    psi = torch.tensor(
        [
            [[0.9, 0.0, 0.7]],
            [[0.0, 0.8, 0.7]],
        ]
    )

    acc, don, metric = cli._routed_summary_track(_result(), prob, psi)

    assert metric == "PSI"
    assert torch.equal(acc, torch.tensor([0.9, 0.0, 0.7]))
    assert torch.equal(don, torch.tensor([0.0, 0.8, 0.7]))


def test_write_routed_bedgraph_outputs_expected_files(tmp_path: Path):
    result = _result()
    prob = torch.tensor(
        [
            [[0.1, 0.0, 0.3]],
            [[0.0, 0.2, 0.3]],
        ]
    )
    psi = torch.tensor(
        [
            [[0.4, 0.0, 0.6]],
            [[0.0, 0.5, 0.6]],
        ]
    )
    prefix = tmp_path / "pred"

    cli._write_routed_bedgraph(
        result,
        str(prefix),
        prob,
        psi,
        chrom="chr1",
        start=10,
        write_prob=True,
        write_psi=True,
        raw_probes=True,
    )

    expected = {
        "pred.brain.prob.acceptor.bg",
        "pred.brain.prob.donor.bg",
        "pred.brain.psi.acceptor.bg",
        "pred.brain.psi.donor.bg",
        "pred.probe.acceptor.bg",
        "pred.probe.donor.bg",
    }
    assert {p.name for p in tmp_path.iterdir()} == expected
    assert (tmp_path / "pred.brain.prob.acceptor.bg").read_text().splitlines()[0] == (
        "chr1\t10\t11\t0.1000"
    )
    assert len((tmp_path / "pred.probe.donor.bg").read_text().splitlines()) == 3


def test_score_seq_rejects_psi_and_psi_only_together():
    with pytest.raises(SystemExit, match="--psi and --psi-only"):
        cli._run_scoring(SimpleNamespace(psi=True, psi_only=True))
