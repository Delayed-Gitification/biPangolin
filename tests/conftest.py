"""Shared pytest fixtures for biPangolin."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def integration_runner():
    """Real runner using local/default cached weights, skipped when unavailable."""
    from bipangolin import BiPangolinRunner

    models = os.environ.get("BIPANGOLIN_TEST_PANGOLIN")
    probes = os.environ.get("BIPANGOLIN_TEST_PROBES")
    device = os.environ.get("BIPANGOLIN_TEST_DEVICE", "cpu")
    try:
        return BiPangolinRunner(
            pangolin_model_dir=models,
            probe_dir=probes,
            device=device,
            tissue="brain",
            n_models_per_tissue=1,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        pytest.skip(f"Pangolin weights not available for integration tests: {e}")


@pytest.fixture(scope="session")
def integration_runner_unscaled():
    """Real runner that preserves raw Pangolin P values in routed output."""
    from bipangolin import BiPangolinRunner

    models = os.environ.get("BIPANGOLIN_TEST_PANGOLIN")
    probes = os.environ.get("BIPANGOLIN_TEST_PROBES")
    device = os.environ.get("BIPANGOLIN_TEST_DEVICE", "cpu")
    try:
        return BiPangolinRunner(
            pangolin_model_dir=models,
            probe_dir=probes,
            device=device,
            tissue="brain",
            n_models_per_tissue=1,
            output_unscaled_values=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        pytest.skip(f"Pangolin weights not available for integration tests: {e}")
