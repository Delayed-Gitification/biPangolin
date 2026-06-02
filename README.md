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

**Hardware:** runs on CUDA, Apple Silicon (MPS), or CPU — selected
automatically. The MPS backend is health-checked at startup and falls back to
CPU if it mishandles Pangolin's high-dilation convolutions.

---

## Quick start

### Command line

```bash
# sanity check against a built-in calibration sequence
bipangolin selftest

# score a sequence (string or FASTA) and print the top donor/acceptor sites
bipangolin score-seq ACGTACGT...

# write per-tissue routed bedGraph tracks
bipangolin score-seq sequence.fa --out predictions

# score a genomic region from a reference FASTA
bipangolin score-region hg38.fa chr19 13200000 13300000 --out region

# annotate a VCF with splice-altering variant effects
bipangolin score-vcf input.vcf output.vcf --fasta hg38.fa
```

### Python

```python
from bipangolin import BiPangolinRunner

runner = BiPangolinRunner()                 # auto-downloads weights
result = runner.score_sequence("ACGT" * 500)

# Recommended output: Pangolin scores routed into acceptor/donor tracks.
prob_routed, _ = result.routed_tracks()     # (2, n_tissues, L); ch0 acceptor, ch1 donor
acceptor_track = prob_routed[0]             # (n_tissues, L)
donor_track    = prob_routed[1]
```

Long sequences are handled transparently — call `runner.score_long_sequence`,
or just use the `bipangolin` CLI / `score_region`, which dispatch automatically.

---

## What you get: routed tracks

biPangolin's primary output is a **routed** pair of tracks. At every position:

- the **value** is always Pangolin's metric (P(spliced), and optionally PSI);
- the **column** it lands in (acceptor or donor) is chosen by the probe;
- the other column is exactly `0`.

So a clean donor site reads `acceptor=0, donor=0.97`, and most intronic
positions read two near-zero values, one of which is a hard zero.

### The routing rule

For each position, using the (corrected) probe acceptor/donor probabilities:

| Case | Result |
|------|--------|
| one class clearly higher | value goes to that **single** column (`argmax`) |
| genuinely ambiguous — `min ≥ floor` **and** `min/max ≥ ratio` | value goes to **both** columns |

Defaults: `floor = 0.01`, `ratio = 0.1`. The floor rejects two near-zero values
(e.g. `0.001 / 0.001` → `argmax`, not "both"); the ratio rejects a clear winner
with a marginal loser (e.g. `donor 0.99 / acceptor 0.02` → donor only, since
`0.02 / 0.99 < 0.1`). Tune with `--double-val-floor` / `--double-val-ratio`.

### One identity, one router

A base has a single splice identity — it either *is* a donor site or *is* an
acceptor site; that is a property of the sequence, not of which Pangolin head
you read. biPangolin therefore computes **one** routing decision per position
and applies it to **both** P and PSI. It deliberately does *not* route PSI by a
separate probe, which could otherwise send the same base's P to the donor column
and its PSI to the acceptor column. (The lone exception is `--psi-only`, where
the P-tuned models are never run and routing falls back to the PSI-side probes.)

### How folds are combined

Each tissue has three trained model+probe folds. Their probability vectors are
**averaged before** the routing decision is made — never hard-voted. A confident
fold dominates; an uncertain fold contributes a near-flat vector that barely
moves the mean; a genuinely split case lands in the "both" bucket via the ratio
test. No special disagreement handling is needed.

---

## Command-line reference

### `score-seq` / `score-region`

```bash
bipangolin score-seq   <sequence|FASTA> [options]
bipangolin score-region <fasta> <chrom> <start> <end> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--out PREFIX` | — | Write per-tissue routed bedGraphs (see below) |
| `--tissue` | `all_tissues` | `heart` / `liver` / `brain` / `testis` / `all_tissues` |
| `--psi` | off | Also load PSI-tuned models and emit routed **PSI** tracks (≈2× inference) |
| `--psi-only` | off | Emit **only** routed PSI; skip the P-tuned models. Routes via the PSI-side probes |
| `--raw-probes` | off | Additionally write the raw probe acceptor/donor tracks |
| `--double-val-floor` | `0.01` | Floor for the "both columns" rule |
| `--double-val-ratio` | `0.1` | min/max ratio for the "both columns" rule |
| `--n-models-per-tissue` | `3` | Folds per tissue to ensemble: `3` (full), or `2` / `1` for faster scoring |
| `--four-track-per-tissue-out FILE.npy` | — | Write the `(4, n_tissues, L)` matrix (implies `--psi`) |
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
| `--n-models-per-tissue` | `3` | Fast mode: `2` / `1` fewer folds per tissue |
| `--no-progress` | off | Disable the progress bar |

VCFs are streamed in a single pass (constant memory regardless of file size) and
each ALT allele is annotated with a `biPangolin=` INFO field.

### bedGraph output layout

`--out PREFIX` writes one acceptor and one donor file per tissue and metric:

```text
PREFIX.<tissue>.prob.acceptor.bg     # routed P(spliced)
PREFIX.<tissue>.prob.donor.bg
PREFIX.<tissue>.psi.acceptor.bg      # only with --psi / --psi-only
PREFIX.<tissue>.psi.donor.bg
PREFIX.probe.acceptor.bg             # only with --raw-probes (tissue-agnostic)
PREFIX.probe.donor.bg
```

Each `.bg` is a standard 4-column bedGraph (`chrom  start  end  value`), ready
to load in IGV or a genome browser.

---

## Fast modes

The full model is a 3-fold ensemble per tissue. For quicker, slightly less
robust scoring, drop to 2 or 1 fold per tissue:

```bash
bipangolin score-vcf in.vcf out.vcf --fasta hg38.fa --n-models-per-tissue 1
```

Inference cost scales roughly linearly with the fold count. Routing and tiling
are unaffected — predictions remain seam-free regardless of the fold count.

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

```python
result = runner.score_sequence(seq)

result.probe_none       # P(no splice site)      (L,)
result.probe_acceptor   # P(acceptor)            (L,)   — corrected
result.probe_donor      # P(donor)               (L,)   — corrected
result.pangolin_prob    # Pangolin P(spliced)    (n_tissues, L)
result.pangolin_psi     # Pangolin PSI           (n_tissues, L)  or None
result.tissues          # tuple of tissue names
result.metadata         # dict: length, tiling info, etc.
```

`pangolin_psi` is populated only when the runner was built with
`use_psi_models=True` (reading the PSI channel off a P-tuned model gives a
misleading side-output, so biPangolin refuses to do it silently).

### Routed tracks

```python
prob_routed, psi_routed = result.routed_tracks(
    double_val_floor=0.01, double_val_ratio=0.1)
# each: (2, n_tissues, L), channel 0 = acceptor, 1 = donor
# psi_routed is None unless the runner loaded PSI models
```

### Four-track matrix

```python
matrix = result.four_track_per_tissue()   # (4, n_tissues, L)
```

Channel order:

```text
0: donor PSI
1: donor P(spliced)
2: acceptor PSI
3: acceptor P(spliced)
```

This uses **exactly the same routing rule** as `routed_tracks` — the only
difference is the 4-channel `(PSI, P) × (donor, acceptor)` layout. Write it from
the CLI with `--four-track-per-tissue-out matrix.npy`.

---

## How it works

1. **Frozen Pangolin trunk.** The Pangolin convolutional network is loaded with
   its weights frozen. Because the model internally crops its receptive-field
   radius (5000 bp) from each side, every emitted position has full ±5000 bp of
   real context — so long sequences are tiled with **no overlap and no
   blending**, and the result is identical to scoring the whole sequence at once.
2. **Probes.** Lightweight convolutional probes read the trunk's internal
   activations and output a 3-way `none` / `acceptor` / `donor` distribution per
   position. Three folds per tissue are averaged.
3. **None-class correction.** Probe probabilities are recalibrated with an
   empirical correction (`optimal_correction.json`) that down-weights the
   dominant `none` class, applied automatically by default.
4. **Routing.** Pangolin's tissue-specific P (and optionally PSI) is routed into
   acceptor/donor tracks using the probe identity and the floor+ratio rule above.

---

## Citation & license

biPangolin builds on Pangolin (Zeng & Li, 2022). The vendored Pangolin model
code is included under the MIT License (© Tony Zeng, Yang I. Li); see `LICENSE`.

If you use biPangolin, please cite both biPangolin and the original Pangolin
paper.
