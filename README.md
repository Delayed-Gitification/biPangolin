# biPangolin

biPangolin runs Pangolin splicing predictions and adds explicit
`None` / `Acceptor` / `Donor` probe outputs, giving Pangolin a SpliceAI-like
splice-site classification track while preserving Pangolin's tissue-specific
outputs.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Delayed-Gitification/biPangolin
cd biPangolin
pip install -e .
```

For FASTA, genomic region, or VCF workflows, install the optional FASTA
dependency:

```bash
pip install -e ".[fasta]"
```

On first use, biPangolin downloads the Pangolin model weights to your local
cache. The biPangolin probe weights are bundled with the package.

## Python Quick Start

```python
from bipangolin import BiPangolinRunner

runner = BiPangolinRunner()
result = runner.score_sequence("ACGT" * 500)

print(result.probe_acceptor)  # P(acceptor) at each position
print(result.probe_donor)     # P(donor) at each position
print(result.pangolin_prob)   # Pangolin tissue-specific P(spliced)
print(result.pangolin_psi)    # Pangolin tissue-specific PSI
```

## Command Line

Run the sanity check:

```bash
bipangolin selftest
```

Score a sequence:

```bash
bipangolin score-seq ACGTACGTACGT
```

Score a FASTA sequence and write bedGraph tracks:

```bash
bipangolin score-seq sequence.fa --out predictions
```

Score a genomic region:

```bash
bipangolin score-region hg38.fa chr19 13200000 13300000 --out region
```

Annotate a VCF:

```bash
bipangolin score-vcf input.vcf output.vcf --fasta hg38.fa
```

## Tissue Selection

By default, biPangolin uses all available tissues. To restrict Pangolin outputs
to one tissue:

```python
runner = BiPangolinRunner(tissue="brain")
```

or:

```bash
bipangolin score-vcf input.vcf output.vcf --fasta hg38.fa --tissue brain
```

Valid tissues are `heart`, `liver`, `brain`, and `testis`.

## Output

The result object contains:

```python
result.probe_none      # P(no splice site), shape (L,)
result.probe_acceptor  # P(acceptor), shape (L,)
result.probe_donor     # P(donor), shape (L,)
result.pangolin_prob   # Pangolin P(spliced), shape (n_tissues, L)
result.pangolin_psi    # Pangolin PSI, shape (n_tissues, L)
result.tissues         # names of tissue rows
```

Probe probabilities use the empirical correction from `optimal_correction.json`
by default.

For a tissue-specific 4-track matrix:

```python
matrix = result.four_track_per_tissue()
print(matrix.shape)  # (4, n_tissues, L)
```

The channel order is:

```text
0: donor PSI
1: donor P(spliced)
2: acceptor PSI
3: acceptor P(spliced)
```

At each position, Pangolin values are routed into donor or acceptor channels
using the biPangolin probe argmax. Positions classified as `None` remain zero.
The same matrix can be written from the command line:

```bash
bipangolin score-seq sequence.fa --four-track-per-tissue-out prediction_matrix.npy
```
