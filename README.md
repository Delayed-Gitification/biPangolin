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

result.brain_P                # (2, L): donor on row 0, acceptor on row 1
result.all_tissue_average_P   # (2, L): mean over all tissues
```

Each `result.<tissue>_P` (and `_PSI`, if you ran PSI models) returns a `(2, L)`
tensor — donor row 0, acceptor row 1. Valid tissues are `heart`, `liver`,
`brain`, `testis`, plus `all_tissue_average` (only when all four tissues were
run). Asking for something that wasn't computed — PSI when you didn't load the
PSI models, or a tissue you didn't score — raises a clear error telling you how
to enable it.

**Common modes** (mix and match on any scoring command):

| You want… | Use |
|-----------|-----|
| Donor/acceptor probability tracks (the default) | *(nothing extra)* |
| Splice-site usage (PSI) too | `--psi` |
| Only usage (PSI), faster | `--psi-only` |
| A single tissue | `--tissue brain` |
| Faster, lower-cost scoring | `--n-models-per-tissue 1` |
| The raw probe outputs as well | `--raw-probes` |

---

## Installation

```bash
git clone https://github.com/Delayed-Gitification/biPangolin
cd biPangolin
pip install -e .
```

For FASTA / genomic-region / VCF workflows, add the FASTA extra:

```bash
pip install -e ".[fasta]"
```

On first use biPangolin downloads the Pangolin model weights to a local cache.
The biPangolin probe weights ship with the package, so nothing else is required.
Scoring runs on GPU, Apple Silicon, or CPU, selected automatically.

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

result.brain_P                              # (2, L): donor row 0, acceptor row 1
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

Once built, a runner can score as many sequences as you like:

```python
runner = BiPangolinRunner(tissue="brain", use_psi_models=True)

result = runner.score_sequence("ACGT" * 500)         # any length; auto-tiled
region = runner.score_region("hg38.fa", "chr19", 13_200_000, 13_300_000)

result.brain_P              # (2, L) routed P,  donor row 0, acceptor row 1
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

Each ALT allele is annotated with a `biPangolin=` INFO field.

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
with donor on row 0 and acceptor on row 1:

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
