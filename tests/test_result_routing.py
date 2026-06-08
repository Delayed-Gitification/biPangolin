from __future__ import annotations

import pytest
import torch

from bipangolin.runner import BiPangolinResult


def _result(**overrides):
    base = {
        "pangolin_prob": torch.tensor([[0.14, 0.23, 0.32, 0.41]]),
        "pangolin_psi": None,
        "probe_none": torch.tensor([0.8, 0.8, 0.8, 0.8]),
        "probe_acceptor": torch.tensor([0.90, 0.01, 0.40, 0.001]),
        "probe_donor": torch.tensor([0.04, 0.90, 0.20, 0.001]),
        "tissues": ("brain",),
        "metadata": {},
    }
    base.update(overrides)
    return BiPangolinResult(**base)


def test_routed_tracks_channel_order_and_both_column_rule():
    result = _result()

    prob_routed, psi_routed = result.routed_tracks(
        double_val_floor=0.05,
        double_val_ratio=0.2,
    )

    assert psi_routed is None
    assert prob_routed.shape == (2, 1, 4)
    # routed_tracks row 0 is acceptor, row 1 is donor.
    assert torch.allclose(prob_routed[0, 0], torch.tensor([0.1, 0.0, 0.3, 0.4]))
    assert torch.allclose(prob_routed[1, 0], torch.tensor([0.0, 0.2, 0.3, 0.0]))


def test_routing_floor_prevents_near_zero_double_routing():
    result = _result(
        pangolin_prob=torch.tensor([[1.0]]),
        probe_none=torch.tensor([0.998]),
        probe_acceptor=torch.tensor([0.001]),
        probe_donor=torch.tensor([0.001]),
    )

    prob_routed, _ = result.routed_tracks(double_val_floor=0.01, double_val_ratio=0.1)

    assert torch.equal(prob_routed[:, 0, 0], torch.tensor([1.0, 0.0]))


def test_routing_applies_same_masks_to_psi():
    result = _result(pangolin_psi=torch.tensor([[0.1, 0.2, 0.3, 0.4]]))

    prob_routed, psi_routed = result.routed_tracks(
        double_val_floor=0.05,
        double_val_ratio=0.2,
    )

    assert torch.equal(prob_routed[0, 0] > 0, psi_routed[0, 0] > 0)
    assert torch.equal(prob_routed[1, 0] > 0, psi_routed[1, 0] > 0)
    assert torch.equal(psi_routed[0, 0], torch.tensor([0.1, 0.0, 0.3, 0.4]))
    assert torch.equal(psi_routed[1, 0], torch.tensor([0.0, 0.2, 0.3, 0.0]))


def test_friendly_accessor_returns_acceptor_then_donor():
    result = _result()

    brain = result.brain_P

    assert brain.shape == (2, 4)
    # Row 0 acceptor, row 1 donor — same order as routed_tracks/CLI/VCF.
    assert torch.allclose(brain[0], torch.tensor([0.1, 0.0, 0.3, 0.4]))
    assert torch.allclose(brain[1], torch.tensor([0.0, 0.2, 0.3, 0.0]))


def test_all_tissue_average_requires_all_tissues():
    with pytest.raises(AssertionError, match="needs all"):
        _ = _result().all_tissue_average_P


def test_missing_psi_accessor_explains_how_to_enable_it():
    with pytest.raises(AssertionError, match="PSI was not computed"):
        _ = _result().brain_PSI


def test_p_accessor_is_unavailable_in_psi_only_results():
    with pytest.raises(AssertionError, match="psi-only mode"):
        _ = _result(metadata={"psi_only": True}).brain_P


def test_raw_concatenates_available_tracks():
    result = _result(pangolin_psi=torch.tensor([[0.1, 0.2, 0.3, 0.4]]))

    raw = result.raw

    assert raw.shape == (5, 4)
    assert torch.equal(raw[0], result.pangolin_prob[0])
    assert torch.equal(raw[1], result.pangolin_psi[0])
    assert torch.equal(raw[4], result.probe_donor)

def test_output_unscaled_values():
    result = _result(metadata={"output_unscaled_values": True})

    prob_routed, psi_routed = result.routed_tracks(
        double_val_floor=0.05,
        double_val_ratio=0.2,
    )
    
    # Values should remain unscaled (e.g. 0.14), and baseline should be 0.05
    assert torch.allclose(prob_routed[0, 0], torch.tensor([0.14, 0.05, 0.32, 0.41]))
    assert torch.allclose(prob_routed[1, 0], torch.tensor([0.05, 0.23, 0.32, 0.05]))
