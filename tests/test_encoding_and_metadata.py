from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import bipangolin
from bipangolin import one_hot_encode


def test_version_matches_package_metadata():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE).group(1)
    assert bipangolin.__version__ == version
    assert bipangolin.__version__ == "0.5.0"


def test_one_hot_encode_accepts_case_rna_and_n():
    encoded = one_hot_encode("aCuGN")

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],  # A
            [0, 1, 0, 0, 0],  # C
            [0, 0, 0, 1, 0],  # G
            [0, 0, 1, 0, 0],  # U -> T
        ],
        dtype=torch.float32,
    )
    assert torch.equal(encoded, expected)
    assert encoded.is_contiguous()


@pytest.mark.parametrize("seq", ["ACGTX", "AC GT", "AC-GT", "ACRGT", "123"])
def test_one_hot_encode_rejects_unexpected_bases(seq):
    with pytest.raises(ValueError, match="unexpected base"):
        one_hot_encode(seq)


def test_one_hot_encode_empty_sequence_has_valid_shape():
    assert one_hot_encode("").shape == (4, 0)
