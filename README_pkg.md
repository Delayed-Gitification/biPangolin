# biPangolin

Per-base donor/acceptor splice site predictions from frozen Pangolin
representations.

Pangolin (Zeng & Li, *Genome Biology* 2022) collapses splice donor and
acceptor predictions into a single "splice usage" track. **biPangolin**
trains small probes on Pangolin's internal representations to recover the
donor/acceptor distinction, achieving >99% accuracy on held-out
chromosomes.

## Install

```bash
pip install bipangolin
```

For FASTA region scoring, also install:

```bash
pip install bipangolin[fasta]
```

The first time you score a sequence, biPangolin will download the Pangolin
model weights (~50 MB) to `~/.cache/bipangolin/` (or your platform's cache
directory). The probe weights are bundled with the package.

## Quick start

```python
from bipangolin import BiPangolinRunner

runner = BiPangolinRunner()
result = runner.score_sequence("ACGT" * 500)

print(result.probe_donor.argmax())     # most likely donor position
print(result.probe_acceptor.argmax())  # most likely acceptor position
```

For long sequences:

```python
result = runner.score_long_sequence(my_50kb_sequence)
```

For genomic regions from a FASTA file:

```python
result = runner.score_region("hg38.fa", "chr19", 13_200_000, 13_300_000)
```

## Variant effect prediction

Score the splicing impact of variants in a VCF file:

```python
from bipangolin import BiPangolinRunner

# Tissue-mixed ensemble (default)
runner = BiPangolinRunner()
n = runner.score_vcf("input.vcf", "output.vcf",
                     fasta_path="hg38.fa", distance=50)

# Brain-only predictions
runner = BiPangolinRunner(tissue="brain")
runner.score_vcf("input.vcf", "output.brain.vcf", fasta_path="hg38.fa")
```

Or score a single variant:

```python
score = runner.score_variant("hg38.fa", "chr19", pos=13207859,
                              ref="A", alt="G", distance=50)
print(f"Acceptor gain: {score.ds_ag:.3f} at offset {score.dp_ag}")
print(f"Donor loss:    {score.ds_dl:.3f} at offset {score.dp_dl}")
print(f"Brain P(spliced) gain: {score.pangolin_per_tissue['brain']['ds_gain']:.3f}")
```

Output VCFs get a `biPangolin=` INFO field per ALT allele:

```
biPangolin=G|0.123|0.000|0.456|0.789|10|0|-5|20|brain:0.500:0.100:5:-2
           ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL|TISSUE:DS_GAIN:DS_LOSS:DP_GAIN:DP_LOSS
```

Where `DS_AG`/`DS_DG` are acceptor/donor gain probabilities, `DS_AL`/`DS_DL`
are losses, and `DP_*` are positions relative to the variant. The
trailing `TISSUE:...` blocks give per-tissue Pangolin P(spliced) deltas
and only appear if multiple tissues were loaded.

CLI:

```bash
bipangolin score-vcf input.vcf output.vcf --fasta hg38.fa --tissue brain --distance 50
```

## Result format

```python
result.probe_none      # (L,) tensor — P(not a splice site)
result.probe_acceptor  # (L,) tensor — P(acceptor)
result.probe_donor     # (L,) tensor — P(donor)
result.pangolin_prob   # (n_tissues, L) — Pangolin's P(spliced) per tissue
result.pangolin_psi    # (n_tissues, L) — Pangolin's PSI per tissue
result.tissues         # tuple of tissue names
```

## Command-line

```bash
bipangolin selftest                                  # sanity check
bipangolin score-seq ACGTACGT...                     # short sequence
bipangolin score-seq myseq.fa --out predictions      # FASTA in, bedGraph out
bipangolin score-region hg38.fa chr19 13200000 13300000 --out region
```

## How it works

For each of Pangolin's 12 ensemble model files, we train a small probe
(~13k parameters) on activations from layers `skip`, `resblock_15`,
`resblock_11`, `resblock_7`, `resblock_3`, `resblock_1` plus the raw
input nucleotide. The probe outputs per-position softmax over
{none, acceptor, donor}.

Training labels are derived from GENCODE GTF annotations on forward-strand
transcripts, with TSS/TTS positions correctly excluded (these appear as
exon boundaries in the GTF but aren't real splice sites).

## Citation

If you use biPangolin in your research, please cite:

```
[citation pending]
```

And the original Pangolin paper:

```
Zeng, T. and Li, Y.I., 2022. Predicting RNA splicing from DNA sequence
using Pangolin. Genome biology, 23(1), p.103.
```

## License

MIT. The vendored Pangolin model architecture is also MIT-licensed; see
`src/bipangolin/model.py` for the upstream copyright notice.
