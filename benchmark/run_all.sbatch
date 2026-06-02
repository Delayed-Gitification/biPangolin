#!/bin/bash
#SBATCH --job-name=bipangolin_bench
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bench_all.%j.out
#SBATCH --error=logs/bench_all.%j.err

# Full biPangolin benchmark pipeline in a single job, from a clean slate:
#
#     [1] score        (GPU)  bench_score.py        -> bench_scores/{chrom}.parquet
#     [2] metrics      (CPU)  bench_metrics.py      -> bench_metrics/summary.csv
#     [3] correlations (CPU)  bench_correlations.py -> bench_metrics/correlations.npz
#
# Scoring uses the (new) BiPangolinRunner directly — the same code path the
# `bipangolin` CLI runs internally — and stores raw per-position probe
# (none/acc/don) + per-tissue Pangolin P and PSI columns, which the metrics and
# correlation stages consume. (Note: none = 1 - acc - don, so even a
# CLI/bedGraph-only workflow can reconstruct it; we store it directly here.)
#
# Submit from the repo root:
#     mkdir -p logs && sbatch benchmark/run_all.sbatch

set -euo pipefail

# --- config (edit these for your data / run) ---------------------------------
DATA=/camp/home/wilkino/home/POSTDOC/software/biPangolin/data
FASTA="$DATA/GRCh38.primary_assembly.genome.fa"
GTF="$DATA/gencode.v47.basic.annotation.gtf"
CHROMS="chr1 chr9"                 # held-out TEST_CHROMS, matches train_probes.py
SCORES_DIR=bench_scores
METRICS_DIR=bench_metrics
# Generate the full superset so every downstream analysis (PR curves,
# per-tissue PSI, correlations) has what it needs without re-scoring. Drop
# these for a faster, P-only run.
EXTRA_SCORE_FLAGS="--use-psi-models --per-tissue-probes"
# -----------------------------------------------------------------------------

# --- conda env ---------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bipangolin
# Conda's libstdc++ must come before the system one (TF / numpy need GLIBCXX_3.4.29).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- info --------------------------------------------------------------------
echo "host:   $(hostname)"
echo "date:   $(date)"
echo "cwd:    $(pwd)"
echo "python: $(which python)"
echo "chroms: $CHROMS"
echo "score flags: $EXTRA_SCORE_FLAGS"
nvidia-smi || true

# --- clean slate -------------------------------------------------------------
echo "== cleaning previous outputs =="
rm -rf "$SCORES_DIR" "$METRICS_DIR"
mkdir -p "$SCORES_DIR" "$METRICS_DIR"

# --- 1. score (GPU, slow) ----------------------------------------------------
echo "== [1/3] scoring -> $SCORES_DIR =="
python benchmark/bench_score.py \
    --fasta "$FASTA" \
    --gtf   "$GTF" \
    --out   "$SCORES_DIR" \
    --chroms $CHROMS \
    --device cuda \
    $EXTRA_SCORE_FLAGS

# --- 2. metrics (CPU) --------------------------------------------------------
echo "== [2/3] metrics -> $METRICS_DIR/summary.csv =="
python benchmark/bench_metrics.py \
    --scores "$SCORES_DIR" \
    --out    "$METRICS_DIR/summary.csv" \
    --chroms $CHROMS

# --- 3. correlations (CPU) ---------------------------------------------------
echo "== [3/3] correlations -> $METRICS_DIR/correlations.npz =="
python benchmark/bench_correlations.py \
    --scores "$SCORES_DIR" \
    --out    "$METRICS_DIR/correlations.npz" \
    --chroms $CHROMS

echo "done: $(date)"
echo "outputs:"
echo "  $SCORES_DIR/<chrom>.parquet"
echo "  $METRICS_DIR/summary.csv"
echo "  $METRICS_DIR/correlations.npz"
