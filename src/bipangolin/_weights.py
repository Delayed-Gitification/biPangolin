"""Locate or download model weights.

Pangolin weights are too large to bundle in the wheel (~60 MB). On first use
we download them from a GitHub release of the bipangolin repo to a cache
directory, then reuse that on subsequent runs.

Probe weights ARE bundled inside the package (they're tiny — ~1.3 MB total).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

# ---- USER-CONFIGURABLE CONSTANTS ----
# These point at YOUR github release. Update before publishing.
PANGOLIN_WEIGHTS_URL = (
    "https://github.com/Delayed-Gitification/biPangolin/releases/download/v0.3.0/pangolin_models_v24.tar.gz"
)
PANGOLIN_WEIGHTS_SHA256 = "REPLACE_WITH_ACTUAL_SHA256_BEFORE_PUBLISHING"
# Expected number of .v2 files in a complete extracted tarball: 3 folds × 8 tissue/head combos.
PANGOLIN_EXPECTED_FILES = 24
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
    try:
        urlretrieve(url, tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download Pangolin weights from {url}. "
            f"Check the URL is reachable and the release exists.\n  {e}"
        ) from e
    tmp.rename(dest)


def _count_v2_files(directory: Path) -> int:
    """Count .v2 weight files anywhere under `directory`."""
    return sum(1 for _ in directory.rglob("final.*.v2"))


def _flatten_singleton_subdir(directory: Path) -> None:
    """If the directory contains nothing but a single subdirectory (which holds
    the actual .v2 files), move that subdir's contents up one level and remove it.

    Handles tarballs that were built with a top-level wrapper directory
    (e.g. `tar -czf foo.tar.gz pangolin_models_v24/`).
    """
    if any(directory.glob("final.*.v2")):
        return  # already flat
    entries = [e for e in directory.iterdir() if not e.name.startswith(".")]
    subdirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    if len(subdirs) == 1 and not files:
        inner = subdirs[0]
        for child in list(inner.iterdir()):
            child.rename(directory / child.name)
        inner.rmdir()


def resolve_pangolin_weights(cache_dir: Optional[Path] = None,
                              force_refresh: bool = False) -> Path:
    """Return path to a directory containing Pangolin .v2 weight files.

    Downloads + extracts on first use; cached thereafter. Detects partial /
    out-of-date caches by file count and re-extracts (or re-downloads) when
    needed. Set the BIPANGOLIN_FORCE_REFRESH=1 env var, pass
    force_refresh=True, or `rm -rf` both `{cache}/pangolin_models/` and
    `{cache}/pangolin_models_v2.tar.gz` to force a fresh download.
    """
    if os.environ.get("BIPANGOLIN_FORCE_REFRESH"):
        force_refresh = True

    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    weights_dir = cache_dir / "pangolin_models"
    archive = cache_dir / "pangolin_models_v2.tar.gz"

    # Cache hit: extracted dir exists and has the expected number of weight files.
    if not force_refresh and weights_dir.exists():
        n = _count_v2_files(weights_dir)
        if n >= PANGOLIN_EXPECTED_FILES:
            # If everything is buried in a subdir, flatten before returning.
            _flatten_singleton_subdir(weights_dir)
            return weights_dir
        if n > 0:
            print(
                f"biPangolin: cached weights at {weights_dir} have {n} files, "
                f"expected {PANGOLIN_EXPECTED_FILES}. Re-extracting.",
                file=sys.stderr,
            )
            shutil.rmtree(weights_dir)
    elif force_refresh and weights_dir.exists():
        shutil.rmtree(weights_dir)

    # Download tarball if missing (or forced).
    if force_refresh and archive.exists():
        archive.unlink()
    if not archive.exists():
        print("biPangolin: no Pangolin weights cached, downloading...", file=sys.stderr)
        _download(PANGOLIN_WEIGHTS_URL, archive)
    _verify_sha256(archive, PANGOLIN_WEIGHTS_SHA256)

    # Extract, then sanity-check the file count. If extraction yields too few
    # files the archive itself is stale/incomplete (e.g. a cached copy of an
    # older 12-file release). Re-download it ONCE and re-extract before giving
    # up, so the cache self-heals instead of trapping the user in a permanent
    # "re-extracting -> still too few -> error" loop.
    def _extract() -> int:
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
        print(f"biPangolin: extracting weights to {weights_dir}", file=sys.stderr)
        weights_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(weights_dir, filter="data")
        _flatten_singleton_subdir(weights_dir)
        return _count_v2_files(weights_dir)

    n = _extract()
    if n < PANGOLIN_EXPECTED_FILES:
        print(
            f"biPangolin: cached archive {archive.name} yielded {n} .v2 files, "
            f"expected {PANGOLIN_EXPECTED_FILES} — it is stale/incomplete. "
            f"Re-downloading once.",
            file=sys.stderr,
        )
        archive.unlink(missing_ok=True)
        _download(PANGOLIN_WEIGHTS_URL, archive)
        _verify_sha256(archive, PANGOLIN_WEIGHTS_SHA256)
        n = _extract()

    # Final sanity check.
    if n < PANGOLIN_EXPECTED_FILES:
        raise RuntimeError(
            f"Extracted {n} .v2 files from {archive.name} into {weights_dir}, "
            f"expected at least {PANGOLIN_EXPECTED_FILES}. The freshly downloaded "
            f"tarball is incomplete. Try `rm -rf {cache_dir}` and re-running, or "
            f"update PANGOLIN_WEIGHTS_URL in {__file__} to point at a complete "
            f"release (the configured URL is {PANGOLIN_WEIGHTS_URL})."
        )
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
