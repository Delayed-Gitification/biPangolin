from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from bipangolin import _weights


def test_default_cache_dir_respects_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BIPANGOLIN_CACHE", str(tmp_path / "cache"))

    assert _weights._default_cache_dir() == tmp_path / "cache"


def test_count_v2_files_recurses(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "final.1.0.3.v2").write_text("x")
    (nested / "final.2.7.3.v2").write_text("x")
    (nested / "not-a-weight.txt").write_text("x")

    assert _weights._count_v2_files(tmp_path) == 2


def test_flatten_singleton_subdir_moves_weight_files_up(tmp_path: Path):
    inner = tmp_path / "pangolin_models_v24"
    inner.mkdir()
    (inner / "final.1.0.3.v2").write_text("weight")

    _weights._flatten_singleton_subdir(tmp_path)

    assert (tmp_path / "final.1.0.3.v2").read_text() == "weight"
    assert not inner.exists()


def test_flatten_singleton_subdir_leaves_flat_directory_alone(tmp_path: Path):
    weight = tmp_path / "final.1.0.3.v2"
    weight.write_text("weight")

    _weights._flatten_singleton_subdir(tmp_path)

    assert weight.exists()


def test_verify_sha256_accepts_matching_hash(tmp_path: Path):
    archive = tmp_path / "weights.tar.gz"
    archive.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()

    _weights._verify_sha256(archive, digest)


def test_verify_sha256_rejects_mismatch(tmp_path: Path):
    archive = tmp_path / "weights.tar.gz"
    archive.write_bytes(b"hello")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _weights._verify_sha256(archive, "0" * 64)


def test_verify_sha256_placeholder_warns_without_raising(tmp_path: Path, capsys):
    archive = tmp_path / "weights.tar.gz"
    archive.write_bytes(b"hello")

    _weights._verify_sha256(archive, "REPLACE_WITH_ACTUAL_SHA256_BEFORE_PUBLISHING")

    assert "NOT being integrity-checked" in capsys.readouterr().err


class LegacyTarFile(tarfile.TarFile):
    def extractall(self, path=".", members=None, *, numeric_owner=False, filter=None):
        if filter is not None:
            raise TypeError("filter is not supported")
        return super().extractall(path, members=members, numeric_owner=numeric_owner)


def _legacy_tar_with_file(name: str, data: bytes = b"weight") -> LegacyTarFile:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    raw.seek(0)
    return LegacyTarFile.open(fileobj=raw, mode="r")


def _legacy_tar_with_symlink(name: str, linkname: str) -> LegacyTarFile:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        info = tarfile.TarInfo(name)
        info.type = tarfile.SYMTYPE
        info.linkname = linkname
        tf.addfile(info)
    raw.seek(0)
    return LegacyTarFile.open(fileobj=raw, mode="r")


def test_safe_extractall_falls_back_for_older_python(tmp_path: Path):
    with _legacy_tar_with_file("final.1.0.3.v2") as tf:
        _weights._safe_extractall(tf, tmp_path)

    assert (tmp_path / "final.1.0.3.v2").read_bytes() == b"weight"


def test_safe_extractall_rejects_traversal_on_older_python(tmp_path: Path):
    with _legacy_tar_with_file("../escape.txt") as tf:
        with pytest.raises(RuntimeError, match="unsafe tar member"):
            _weights._safe_extractall(tf, tmp_path)


def test_safe_extractall_rejects_links_on_older_python(tmp_path: Path):
    with _legacy_tar_with_symlink("models-link", "/tmp/models") as tf:
        with pytest.raises(RuntimeError, match="unsafe tar member"):
            _weights._safe_extractall(tf, tmp_path)
