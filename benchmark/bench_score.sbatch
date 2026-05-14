#!/bin/bash
#SBATCH --job-name=bench_score
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bench_score.%j.out
#SBATCH --error=logs/bench_score.%j.err

# Run from the biPangolin project root.
# Submit with:
#   mkdir -p logs && sbatch benchmark/bench_score.sbatch

set -euo pipefail

# --- conda env ---------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bipangolin

# Conda's libstdc++ must come before the system one (TF / numpy need GLIBCXX_3.4.29).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- info --------------------------------------------------------------------
echo "host:      $(hostname)"
echo "date:      $(date)"
echo "cwd:       $(pwd)"
echo "python:    $(which python)"
nvidia-smi || true

# --- run ---------------------------------------------------------------------
DATA=/camp/home/wilkino/home/POSTDOC/software/biPangolin/data

python benchmark/bench_score.py \
    --fasta "$DATA/GRCh38.primary_assembly.genome.fa" \
    --gtf   "$DATA/gencode.v47.basic.annotation.gtf" \
    --out   bench_scores/ \
    --device cuda

echo "done: $(date)"
