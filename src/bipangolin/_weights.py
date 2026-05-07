"""Locate or download model weights.

Pangolin weights are too large to bundle in the wheel (~50 MB). On first use
we download them from a GitHub release of the bipangolin repo to a cache
directory, then reuse that on subsequent runs.

Probe weights ARE bundled inside the package (they're tiny — ~1.3 MB total).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

# ---- USER-CONFIGURABLE CONSTANTS ----
# These point at YOUR github release. Update before publishing.
PANGOLIN_WEIGHTS_URL = (
    "https://github.com/USERNAME/bipangolin/releases/download/v0.1.0/"
    "pangolin_models_v2.tar.gz"
)
PANGOLIN_WEIGHTS_SHA256 = "REPLACE_WITH_ACTUAL_SHA256_BEFORE_PUBLISHING"
# -------------------------------------


def _default_cache_dir() -> Path:
    """Where to store auto-downloaded weights."""
    if "BIPANGOLIN_CACHE" in os.environ:
        return Path(os.environ["BIPANGOLIN_CACHE"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "bipangolin"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "bipangolin" / "Cache"
    # XDG on Linux
    xdg = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(xdg) / "bipangolin"


def _verify_sha256(path: Path, expected: str) -> None:
    if expected.startswith("REPLACE_"):
        return  # dev placeholder, skip
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path.name}: got {got}, expected {expected}")


def _download(url: str, dest: Path) -> None:
    print(f"biPangolin: downloading {url}\n  -> {dest}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp)
    tmp.rename(dest)


def resolve_pangolin_weights(cache_dir: Optional[Path] = None) -> Path:
    """Return path to a directory containing Pangolin .v2 weight files.

    Downloads + extracts on first use; cached thereafter.
    """
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    weights_dir = cache_dir / "pangolin_models"
    if weights_dir.exists() and any(weights_dir.glob("final.*.v2")):
        return weights_dir

    archive = cache_dir / "pangolin_models_v2.tar.gz"
    if not archive.exists():
        _download(PANGOLIN_WEIGHTS_URL, archive)
    _verify_sha256(archive, PANGOLIN_WEIGHTS_SHA256)

    print(f"biPangolin: extracting weights to {weights_dir}", file=sys.stderr)
    weights_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(weights_dir, filter="data")
    return weights_dir


def resolve_probe_weights() -> Path:
    """Return path to the bundled probe weights directory inside the package."""
    here = Path(__file__).parent
    probes = here / "data" / "probes"
    if not probes.exists():
        raise FileNotFoundError(
            f"Bundled probes not found at {probes}. "
            "Was the package installed correctly?")
    return probes
