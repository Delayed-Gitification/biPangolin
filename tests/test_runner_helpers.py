from __future__ import annotations

import pytest
import torch

from bipangolin.runner import BiPangolinRunner, make_probe, parse_probe_layers


def test_parse_probe_layers_accepts_string_and_sequence():
    assert parse_probe_layers("skip+resblock_15") == ["skip", "resblock_15"]
    assert parse_probe_layers(("resblock_1", "skip")) == ["resblock_1", "skip"]


def test_parse_probe_layers_rejects_unknown_layer():
    with pytest.raises(ValueError, match="probe_layer"):
        parse_probe_layers("skip+not_a_layer")


def test_make_probe_without_hidden_layer_outputs_three_classes():
    probe = make_probe(kernel_size=1, hidden_dim=None, in_channels=32)

    out = probe(torch.zeros(2, 32, 5))

    assert out.shape == (2, 3, 5)


def test_make_probe_with_hidden_layer_outputs_three_classes():
    probe = make_probe(kernel_size=3, hidden_dim=7, in_channels=36)

    out = probe(torch.zeros(2, 36, 5))

    assert out.shape == (2, 3, 5)


def test_runner_rejects_window_with_no_usable_output_before_loading_weights():
    with pytest.raises(ValueError, match="window_len must exceed"):
        BiPangolinRunner(window_len=10_000)


def test_runner_rejects_invalid_model_count_before_loading_weights():
    with pytest.raises(ValueError, match="n_models_per_tissue"):
        BiPangolinRunner(n_models_per_tissue=4)


def test_limit_per_tissue_keeps_lowest_folds_in_order():
    candidates = [
        ("final.1.0.3.v2", 0),
        ("final.2.0.3.v2", 0),
        ("final.3.0.3.v2", 0),
        ("final.1.2.3.v2", 1),
        ("final.2.2.3.v2", 1),
    ]

    assert BiPangolinRunner._limit_per_tissue(candidates, 2) == [
        ("final.1.0.3.v2", 0),
        ("final.2.0.3.v2", 0),
        ("final.1.2.3.v2", 1),
        ("final.2.2.3.v2", 1),
    ]
