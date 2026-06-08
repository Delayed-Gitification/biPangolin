# biPangolin

**Per-base splice donor/acceptor predictions from Pangolin.**

[Pangolin](https://github.com/tkzeng/Pangolin) is a state-of-the-art splice
prediction model, but it collapses splice **donor** and **acceptor** signals
into a single tissue-specific "splice usage" track — it never tells you *which
kind* of splice site a position is. biPangolin recovers that distinction.

It does so without retraining Pangolin. Small **probes** are trained on the
frozen Pangolin network's internal representations to classify each position as
`none` / `acceptor` / `donor`. The probe decides the splice-site *identity*;
Pangolin still provides the *score*. The result is a SpliceAI-style pair of
donor and acceptor tracks, with Pangolin's tissue-specific values intact.

---

## Common ways to use biPangolin

**On the command line** — score a sequence, a genomic region, or a whole VCF:

```bash
bipangolin score-seq    sequence.fa --out predictions          # → bedGraph tracks
bipangolin score-region hg38.fa chr19 13200000 13300000 --out region
bipangolin score-vcf    input.vcf output.vcf --fasta hg38.fa   # → annotated VCF
```

**As a Python library** — get a result object and read the routed donor/acceptor
tracks by name:

```python
from bipangolin import BiPangolinRunner

runner = BiPangolinRunner()                  # all tissues
result = runner.score_sequence("ACGT" * 500)

result.brain_P                # (2, L): acceptor on row 0, donor on row 1
result.all_tissue_average_P   # (2, L): mean over all tissues
```

Each `result.<tissue>_P` (and `_PSI`, if you ran PSI models) returns a `(2, L)`
tensor — acceptor row 0, donor row 1 (the same order used everywhere in biPangolin). Valid tissues are `heart`, `liver`,
`brain`, `testis`, plus `all_tissue_average` (only when all four tissues were
run). Asking for something that wasn't computed — PSI when you didn't load the
PSI models, or a tissue you didn't score — raises a clear error telling you how
to enable it.

**Common modes** (mix and match). Each is a CLI flag and a `BiPangolinRunner`
argument:

| You want… | CLI flag | Runner argument |
|-----------|----------|-----------------|
| Donor/acceptor probability tracks (the default) | *(nothing extra)* | *(nothing extra)* |
| Splice-site usage (PSI) too | `--psi` | `use_psi_models=True` |
| Only usage (PSI), faster | `--psi-only` | `use_psi_models=True` + `score_sequence(seq, psi_only=True)` |
| A single tissue | `--tissue brain` | `tissue="brain"` |
| Faster, lower-cost scoring | `--n-models-per-tissue 1` | `n_models_per_tissue=1` |
| The raw probe outputs as well | `--raw-probes` | *(always on the result: `result.probe_acceptor` / `result.probe_donor`)* |

---
Note, when using PSI-only, acceptor vs donor routing may differ as PSI probes are used for routing (rather than P probes). Strong splice sites should be consistent in essentially all cases.

## Installation

```bash
git clone https://github.com/Delayed-Gitification/biPangolin
cd biPangolin
pip install -e .
```

This pulls in everything needed, including FASTA reading (`pyfastx`) for the
genomic-region and VCF workflows. Scoring runs on GPU, Apple Silicon, or CPU,
selected automatically.

### Model weights and caching

The biPangolin probe weights ship inside the package, but the Pangolin model
weights (~60 MB) are too large to bundle, so on first use biPangolin downloads
and extracts them to a local cache. Subsequent runs reuse that cache.

By default the cache lives in the OS cache directory:

| OS | Default location |
|----|------------------|
| Linux | `$XDG_CACHE_HOME/bipangolin` (usually `~/.cache/bipangolin`) |
| macOS | `~/Library/Caches/bipangolin` |
| Windows | `%LOCALAPPDATA%\bipangolin\Cache` |

On HPC nodes with small home-directory quotas, or air-gapped compute nodes,
this matters:

```bash
# Redirect the cache to scratch / a shared project space.
export BIPANGOLIN_CACHE=/scratch/$USER/bipangolin

# Pre-download weights on a login node, then run offline. Either populate the
# cache above and copy it to the compute node, or point the CLI/runner straight
# at an unpacked weights directory:
bipangolin score-seq seq.fa --models /shared/pangolin_models --probes /shared/probes
```

`--models` / `--probes` (CLI) and `pangolin_model_dir` / `probe_dir`
(`BiPangolinRunner`) bypass the download entirely — useful when the compute node
has no internet access. Other env vars: `BIPANGOLIN_FORCE_REFRESH=1` forces a
fresh download.

---

## Quick start

### Command line

```bash
# sanity check against a built-in calibration sequence
bipangolin selftest

# score a sequence (string or FASTA) and print the top donor/acceptor sites
bipangolin score-seq ACGTACGT...

# write per-tissue donor/acceptor bedGraph tracks
bipangolin score-seq sequence.fa --out predictions
```

### Python

```python
from bipangolin import BiPangolinRunner

runner = BiPangolinRunner(tissue="brain")   # auto-downloads weights; one tissue
result = runner.score_sequence("ACGT" * 500)

result.brain_P                              # (2, L): acceptor row 0, donor row 1
```

Long sequences are handled transparently — call `runner.score_long_sequence`,
or just use the `bipangolin` CLI / `score_region`, which dispatch automatically.

---

## Python: runner options

You configure biPangolin once, when you build the `BiPangolinRunner`. The
defaults are tuned for the common case — all four tissues, the full 3-fold
ensemble, P (probability) tracks, automatic device selection — so
`BiPangolinRunner()` with no arguments is a sensible starting point.

```python
from bipangolin import BiPangolinRunner

# Defaults: tissue="all_tissues", n_models_per_tissue=3 (full ensemble),
# use_psi_models=False (P only), device="auto" (GPU / Apple Silicon / CPU).
runner = BiPangolinRunner()
```

The settings you are most likely to change:

```python
# A single tissue (faster, and the per-tissue accessors return one track set).
runner = BiPangolinRunner(tissue="brain")     # heart / liver / brain / testis

# Faster, lower-cost scoring: use fewer of the 3 folds per tissue.
runner = BiPangolinRunner(n_models_per_tissue=1)   # or 2; default is 3

# Also compute PSI (splice-site usage), not just P. Roughly doubles inference.
runner = BiPangolinRunner(use_psi_models=True)
result = runner.score_sequence(seq)
result.brain_PSI            # now available; otherwise PSI accessors error

# Force a specific device instead of auto-detection.
runner = BiPangolinRunner(device="cpu")       # or "cuda", "mps"

# Lower the per-tile input length if a machine runs out of memory.
runner = BiPangolinRunner(window_len=20000)   # default 50000
```

`window_len` is the per-tile input length; longer sequences are split into tiles
of this size, so it caps peak memory regardless of how long the input is. The
default `50000` is comfortable on a GPU or a modern 16 GB+ machine. On
memory-constrained CPU nodes, dropping to `10000`–`20000` cuts the peak roughly
in proportion at the cost of more (cheap) forward passes. It must stay above
`2 × 5000` (the model's receptive-field crop); below that no usable output
remains.

Once built, a runner can score as many sequences as you like:

```python
runner = BiPangolinRunner(tissue="brain", use_psi_models=True)

result = runner.score_sequence("ACGT" * 500)         # any length; auto-tiled
region = runner.score_region("hg38.fa", "chr19", 13_200_000, 13_300_000)

result.brain_P              # (2, L) routed P,  acceptor row 0, donor row 1
result.brain_PSI            # (2, L) routed PSI
```

Routing is decided at read time, so you can re-route the same result with
different sensitivity without re-scoring:

```python
result = runner.score_sequence(seq)

# Defaults: floor 0.01, ratio 0.1 (when a position is assigned to both
# donor and acceptor columns).
prob_routed, psi_routed = result.routed_tracks()                  # (2, n_tissues, L)
prob_routed, _          = result.routed_tracks(double_val_floor=0.05,
                                                double_val_ratio=0.2)
```

---

## What you get: routed tracks

biPangolin's primary output is a **routed** pair of tracks. At every position:

- the **value** is always Pangolin's metric (P(spliced), and optionally PSI);
- the **column** it lands in (acceptor or donor) is chosen by the probe;
- the other column is `0`.

So a clean donor site reads `acceptor=0, donor=0.97`.

The routing rule, using the probe acceptor/donor probabilities at each position:

| Case | Result |
|------|--------|
| one class clearly higher | value goes to that **single** column |
| genuinely ambiguous (`min ≥ floor` and `min/max ≥ ratio`) | value goes to **both** columns |

Defaults are `floor = 0.01`, `ratio = 0.1`, tunable with `--double-val-floor` /
`--double-val-ratio`. The same single routing decision is applied to both P and
PSI, so a position's donor/acceptor identity is consistent across metrics.

### Raw probe values

The probe's own `none` / `acceptor` / `donor` probabilities are **hidden by
default**. You can still get them:

```bash
bipangolin score-seq sequence.fa --out predictions --raw-probes
# → PREFIX.probe.acceptor.bg, PREFIX.probe.donor.bg
```

```python
result.probe_acceptor   # (L,) P(acceptor)
result.probe_donor      # (L,) P(donor)
result.probe_none       # (L,) P(no splice site)
```

We **recommend reporting the routed Pangolin values rather than the probe
probabilities themselves.** The probe scores are actually very good — this is
not about accuracy. The point is consistency: the headline numbers stay
Pangolin's, and the probe is used only to decide donor vs. acceptor. That keeps
results directly comparable to plain Pangolin and avoids introducing a second
score that users have to reason about. Hence the probe values are off by default.

---

## Command-line reference

### `score-seq` / `score-region`

```bash
bipangolin score-seq    <sequence|FASTA> [options]
bipangolin score-region <fasta> <chrom> <start> <end> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--out PREFIX` | — | Write per-tissue donor/acceptor bedGraphs (see below) |
| `--tissue` | `all_tissues` | `heart` / `liver` / `brain` / `testis` / `all_tissues` |
| `--psi` | off | Also emit routed **PSI** (usage) tracks |
| `--psi-only` | off | Emit **only** routed PSI; skip the P-tuned models |
| `--raw-probes` | off | Additionally write the raw probe acceptor/donor tracks |
| `--double-val-floor` | `0.01` | Floor for the "both columns" rule |
| `--double-val-ratio` | `0.1` | min/max ratio for the "both columns" rule |
| `--n-models-per-tissue` | `3` | Folds per tissue to ensemble: `3` (full), or `2` / `1` for faster scoring |
| `--top` | `10` | Number of top sites to print |
| `--models DIR` | *(auto-download)* | Use Pangolin weights from this directory instead of the cache (offline use) |
| `--probes DIR` | *(bundled)* | Use probe weights from this directory instead of the ones shipped with the package |

`--psi` and `--psi-only` are mutually exclusive.

### `score-vcf`

```bash
bipangolin score-vcf <in.vcf[.gz]> <out.vcf> --fasta <ref.fa> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--fasta` | *required* | Reference FASTA matching the VCF build |
| `--tissue` | `all_tissues` | Restrict to one tissue, or use the full ensemble |
| `--distance` | `50` | Report the max delta within ± this many nt of the variant |
| `--n-models-per-tissue` | `3` | Fewer folds per tissue for faster scoring |
| `--no-progress` | off | Disable the progress bar |

(`score-vcf` also accepts `--models` / `--probes` for offline weights, as above.)

#### The `biPangolin=` INFO field

Each ALT allele is annotated with a `biPangolin=` INFO field, in a SpliceAI-style
pipe-delimited format:

```text
ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL[|TISSUE:DS_GAIN:DS_LOSS:DP_GAIN:DP_LOSS ...]
```

| Field | Meaning |
|-------|---------|
| `ALT` | the alternate allele this annotation is for |
| `DS_AG` / `DS_AL` | delta score for **a**cceptor **g**ain / **l**oss (probe-based, tissue-agnostic) |
| `DS_DG` / `DS_DL` | delta score for **d**onor **g**ain / **l**oss |
| `DP_AG` / `DP_AL` / `DP_DG` / `DP_DL` | position (relative to the variant, in nt) of each of those max deltas |
| `TISSUE:...` | per-tissue Pangolin P(spliced) gain/loss deltas and their positions |

Delta scores are formatted to 3 decimals; positions are signed integers
(negative = upstream of the variant). With multiple ALT alleles the INFO value
holds a comma-separated list, one annotation per allele.

**How many tissue blocks?** The `DS_*` / `DP_*` core fields are always
tissue-agnostic (probe-based) and appear once. The trailing `TISSUE:...` blocks
are *one per tissue actually loaded* — so `--tissue all_tissues` (the default)
appends **four** blocks (`heart`, `liver`, `brain`, `testis`), while
`--tissue brain` appends a single `brain:` block. biPangolin does **not** emit
an averaged tissue block; if you want one tissue in the VCF, pass `--tissue`.
Restricting the tissue is the way to keep the INFO field short for downstream
parsers.

#### VCF Annotation Limits

`score-vcf` is designed as a lightweight, memory-efficient streaming parser with zero external dependencies (it does not require `pysam`). To achieve this, it has a few specific constraints:
- **Automatic Gzip:** It natively reads `.vcf` or `.vcf.gz` inputs. The output will automatically be compressed with gzip if the output filename ends in `.gz`.
- **Structural / Symbolic Variants:** It expects a clean VCF containing only standard sequence alleles. Symbolic alleles (like `<DEL>` or `<DUP>`) are skipped silently and annotated with null values (`.|.|.|.`).
- **Performance:** For high-throughput annotation on massive cohorts, batching is not currently implemented. We recommend pre-filtering your VCF with `bcftools` to contain only variants of interest before scoring.

### bedGraph output layout

`--out PREFIX` writes one acceptor and one donor file per tissue and metric:

```text
PREFIX.<tissue>.prob.acceptor.bg     # donor/acceptor P(spliced)
PREFIX.<tissue>.prob.donor.bg
PREFIX.<tissue>.psi.acceptor.bg      # only with --psi / --psi-only
PREFIX.<tissue>.psi.donor.bg
PREFIX.probe.acceptor.bg             # only with --raw-probes (tissue-agnostic)
PREFIX.probe.donor.bg
```

Each `.bg` is a standard 4-column bedGraph (`chrom  start  end  value`), ready
to load in IGV or a genome browser.

---

## Tissue selection

By default biPangolin uses all four tissues. Restrict to one:

```python
runner = BiPangolinRunner(tissue="brain")
```

```bash
bipangolin score-seq sequence.fa --tissue brain --out brain_preds
```

Valid tissues: `heart`, `liver`, `brain`, `testis`.

---

## The result object

The friendly accessors are the easiest entry point — each is a `(2, L)` tensor
with acceptor on row 0 and donor on row 1 (consistent with routed_tracks, the CLI bedGraphs, and the VCF deltas):

```python
result = runner.score_sequence(seq)

result.brain_P                # (2, L) for brain
result.liver_PSI              # (2, L), PSI metric (needs PSI models)
result.all_tissue_average_P   # (2, L), mean over all tissues
```

The full set of attributes:

```python
result = runner.score_sequence(seq)

result.probe_none       # P(no splice site)      (L,)
result.probe_acceptor   # P(acceptor)            (L,)
result.probe_donor      # P(donor)               (L,)
result.pangolin_prob    # Pangolin P(spliced)    (n_tissues, L)
result.pangolin_psi     # Pangolin PSI           (n_tissues, L)  or None
result.tissues          # tuple of tissue names
result.metadata         # dict of run metadata
```

```python
prob_routed, psi_routed = result.routed_tracks()
# each: (2, n_tissues, L), channel 0 = acceptor, 1 = donor
# psi_routed is None unless the runner loaded PSI models (--psi / use_psi_models)
```

`pangolin_psi` (and `psi_routed`) are populated only when the runner is built
with `use_psi_models=True`, the library equivalent of `--psi`.

---

## How it works

1. **Frozen Pangolin trunk.** The Pangolin network is loaded with its weights
   frozen; long sequences are tiled automatically.
2. **Probes.** Lightweight convolutional probes read the trunk's internal
   activations and output a `none` / `acceptor` / `donor` distribution per
   position, ensembled across folds per tissue.
3. **Calibration.** Probe probabilities are recalibrated with an empirical
   correction, applied automatically by default.
4. **Routing.** Pangolin's tissue-specific scores are routed into acceptor and
   donor tracks using the probe identity.

### Models and probes loaded

Pangolin is a single multi-output network, but the released weights ship as
**three independently fine-tuned versions of each output** (an ensemble of
folds). biPangolin uses those folds directly: with the default full ensemble it
loads up to **24 models** during inference — three folds for each of the four
tissues' P-tuned outputs (12), plus, when `use_psi_models=True` / `--psi`, three
folds for each tissue's PSI-tuned output (another 12). Each loaded model carries
its **own bespoke probe**, trained on that specific model's internal
activations, and the per-position `none`/`acceptor`/`donor` predictions are
averaged across folds per tissue.

This is why `--n-models-per-tissue` / `n_models_per_tissue` (1, 2, or 3) trades
robustness for speed - fewer models = less inference cost.

---

## Citation & license

biPangolin builds on Pangolin (Zeng & Li, 2022). The vendored Pangolin model
code is included under the MIT License (© Tony Zeng, Yang I. Li); see `LICENSE`.

If you use biPangolin, please cite both biPangolin and the original Pangolin
paper.
