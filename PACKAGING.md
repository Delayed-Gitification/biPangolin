# Packaging biPangolin for PyPI

This guide walks through publishing biPangolin to PyPI with bundled probe
weights and auto-downloaded Pangolin weights.

## Step 0 — One-time GitHub repo setup

Create a public repo, e.g. `github.com/wilkino/bipangolin`. Push this whole
package directory there.

## Step 1 — Vendor the Pangolin model architecture

```bash
cd src/bipangolin/
curl -O https://raw.githubusercontent.com/tkzeng/Pangolin/main/pangolin/model.py
# Verify the file looks right
head -20 model.py
```

Add this header to the top of `model.py`:

```python
# Vendored from https://github.com/tkzeng/Pangolin (MIT License)
# Copyright (c) Tony Zeng, Yang I. Li, et al.
```

This means **users no longer need** `pip install git+https://github.com/tkzeng/Pangolin.git`
— removing the most fragile dependency.

## Step 2 — Bundle probe weights

Copy your 24 trained probes into `src/bipangolin/data/probes/`:

```bash
mkdir -p src/bipangolin/data/probes
cp /path/to/bipangolin_probes/*.pt src/bipangolin/data/probes/
ls src/bipangolin/data/probes/ | wc -l   # should be 24
```

Total size should be ~1.3 MB — well within PyPI's per-file limit. The
`pyproject.toml` already includes a `force-include` directive to ensure
they ship with the wheel.

## Step 3 — Build a tarball of the Pangolin weights

```bash
cd /path/to/Pangolin/pangolin/models/
tar czf pangolin_models_v24.tar.gz final.[1-3].[0246].3.v2 final.[1-3].[1357].3.v2
ls -lh pangolin_models_v24.tar.gz
```

Include all 24 Pangolin v2 files: 12 P-tuned files plus 12 PSI-tuned files.
The default P-only workflow uses the even-indexed files; `--psi` / `--psi-only`
also need the odd-indexed PSI-tuned files.

Get the SHA-256:

```bash
sha256sum pangolin_models_v24.tar.gz
```

Update `src/bipangolin/_weights.py`:
- Replace `USERNAME` in `PANGOLIN_WEIGHTS_URL` with your GitHub username
- Replace `REPLACE_WITH_ACTUAL_SHA256_BEFORE_PUBLISHING` with the real hash

## Step 4 — Create a GitHub Release

```bash
git tag v0.4.0
git push origin v0.4.0
```

Then on github.com:
1. Go to your repo → Releases → Draft a new release
2. Tag: `v0.4.0`, title: "biPangolin v0.4.0"
3. Upload `pangolin_models_v24.tar.gz` as a release asset
4. Publish release

The download URL will be:
```
https://github.com/USERNAME/bipangolin/releases/download/v0.4.0/pangolin_models_v24.tar.gz
```

This must match what's in `_weights.py`.

## Step 5 — Test locally

```bash
pip install -e .

# Should work (uses bundled probes + downloads Pangolin weights)
bipangolin selftest

# Or in Python
python -c "from bipangolin import selftest; selftest()"
```

Expected output:
```
biPangolin: 12 model+probe pairs ready on cuda
  donor   peak: pos=  69 (expected 69)  P=0.998
  acceptor peak: pos= 163 (expected 163) P=0.997
```

## Step 6 — Build and check the wheel

```bash
pip install build twine
python -m build
ls dist/   # should see bipangolin-0.4.0-py3-none-any.whl and .tar.gz

# Inspect the wheel contents to confirm probes are included
unzip -l dist/bipangolin-0.4.0-py3-none-any.whl | grep probes
# Should list all 24 .pt files

# Check metadata
twine check dist/*
```

## Step 7 — Test in a clean environment

```bash
python -m venv /tmp/test_install
source /tmp/test_install/bin/activate
pip install dist/bipangolin-0.4.0-py3-none-any.whl
bipangolin selftest
deactivate
```

If selftest passes, you're ready to publish.

## Step 8 — Upload to PyPI

First-time setup: register at pypi.org and create an API token.

Test upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ bipangolin
```

Then real PyPI:

```bash
twine upload dist/*
```

Done. Anyone can now `pip install bipangolin`.

## Releasing a new version

```bash
# 1. Bump version in pyproject.toml AND src/bipangolin/__init__.py
# 2. If you retrained probes, replace src/bipangolin/data/probes/*.pt
# 3. If you changed Pangolin weights, build new tarball + new GitHub release
#    with new tag, update _weights.py URL + SHA
# 4. Build and upload
git tag v0.4.0
git push origin v0.4.0
rm -rf dist/
python -m build
twine upload dist/*
```

## Notes on the design

**Probe weights bundled, Pangolin weights downloaded.** Probes are tiny
(~1.3 MB, 24 files), Pangolin weights are large (~50 MB). PyPI allows wheels
up to 100 MB but it's bad practice to ship large model weights — slower
installs, wasted bandwidth for users who don't run inference, and PyPI
storage isn't designed for binary blobs. GitHub Releases is the right place.

**Cache directory respects platform conventions.** Linux uses XDG
(`$XDG_CACHE_HOME/bipangolin` or `~/.cache/bipangolin`), macOS uses
`~/Library/Caches/bipangolin`, Windows uses `%LOCALAPPDATA%\bipangolin\Cache`.
Override with `BIPANGOLIN_CACHE` env var if needed.

**Vendoring Pangolin's `model.py` is fine because:**
1. It's MIT-licensed (verify by checking their LICENSE file).
2. It's small (~200 lines), unlikely to change frequently.
3. It removes a fragile `git+https://...` dependency from your install.
4. You include the original copyright notice as required by MIT.
