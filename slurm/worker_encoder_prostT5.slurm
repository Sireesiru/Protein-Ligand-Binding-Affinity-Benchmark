#!/bin/bash
#SBATCH -A gen006
#SBATCH -J prostt5_embed
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 04:00:00
#SBATCH -p extended
#SBATCH -o logs/prostt5_%j.out
#SBATCH -e logs/prostt5_%j.err

# ------------------------
# Setup
# ------------------------
mkdir -p logs
mkdir -p prostt5_embeddings

module load miniforge3
conda activate protgpt2

# Prevent user-site conflicts
export PYTHONNOUSERSITE=1

# Optional: better GPU memory handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Starting ProstT5 embedding job"
echo "========================================"

# ------------------------
# Run script
# ------------------------
python3 worker_encoder_prostT5.py \
    --csv pdbbind_canonical_affinities.csv \
    --col seq \
    --outdir prostt5_embeddings \
    --batch_size 8

echo "========================================"
echo "Job complete!"
echo "========================================"