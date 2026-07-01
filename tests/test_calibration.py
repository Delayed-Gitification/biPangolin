import pytest
import torch

from bipangolin import CALIBRATION_SEQ


pytestmark = pytest.mark.integration


def test_calibration_donor_acceptor(integration_runner):
    """Calibration sequence should peak at donor=69 and acceptor=163."""
    result = integration_runner.score_sequence(CALIBRATION_SEQ)
    don_pos = int(result.probe_donor.argmax())
    acc_pos = int(result.probe_acceptor.argmax())
    assert abs(don_pos - 69) <= 2, f"donor peak at {don_pos}, expected 69"
    assert abs(acc_pos - 163) <= 2, f"acceptor peak at {acc_pos}, expected 163"
    assert result.probe_donor.max() > 0.5
    assert result.probe_acceptor.max() > 0.5


def test_result_shapes(integration_runner):
    """Result tensor shapes match input length."""
    result = integration_runner.score_sequence(CALIBRATION_SEQ)
    L = len(CALIBRATION_SEQ)
    assert result.probe_donor.shape == (L,)
    assert result.probe_acceptor.shape == (L,)
    assert result.probe_none.shape == (L,)
    assert result.pangolin_prob.shape == (1, L)
    assert result.tissues == ("brain",)
    # probabilities sum to 1
    total = result.probe_none + result.probe_acceptor + result.probe_donor
    assert torch.allclose(total, torch.ones(L), atol=1e-4)


def test_long_sequence_tiling(integration_runner):
    """Long sequence routes through tiling and produces correct length output."""
    long_seq = CALIBRATION_SEQ * 200
    result = integration_runner.score_long_sequence(long_seq)
    assert len(result) == len(long_seq)
    assert result.metadata["tiled"] is True
    # Probabilities should still sum to 1 everywhere
    total = result.probe_none + result.probe_acceptor + result.probe_donor
    assert torch.allclose(total, torch.ones(len(long_seq)), atol=1e-3)


def test_friendly_accessor_on_real_result(integration_runner):
    result = integration_runner.score_sequence(CALIBRATION_SEQ)

    brain = result.brain_P

    assert brain.shape == (2, len(CALIBRATION_SEQ))


def test_unscaled_routed_values_are_original_pangolin_values(integration_runner_unscaled):
    """With scaling disabled, routing must preserve Pangolin's P values exactly."""
    result = integration_runner_unscaled.score_sequence(CALIBRATION_SEQ)
    prob_routed, psi_routed = result.routed_tracks()
    acc_mask, don_mask = result._routing_masks()

    assert psi_routed is None
    assert result.metadata["output_unscaled_values"] is True
    assert torch.allclose(prob_routed[0, :, acc_mask], result.pangolin_prob[:, acc_mask])
    assert torch.allclose(prob_routed[1, :, don_mask], result.pangolin_prob[:, don_mask])
    assert torch.all(prob_routed[0, :, ~acc_mask] == 0.05)
    assert torch.all(prob_routed[1, :, ~don_mask] == 0.05)
