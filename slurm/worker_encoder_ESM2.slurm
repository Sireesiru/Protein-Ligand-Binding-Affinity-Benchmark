#!/bin/bash
#SBATCH -A gen006
#SBATCH -J esm2_embeddings
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 04:00:00
#SBATCH -p extended
#SBATCH -o logs/esm2_%j.out
#SBATCH -e logs/esm2_%j.err

# Ensure directories exist
mkdir -p logs
mkdir -p esm2_embeddings

# Environment setup
module load miniforge3
conda activate protgpt2

# Safety flag
export PYTHONNOUSERSITE=1

echo "========================================"
echo "Starting ESM2 embedding job"
echo "========================================"

# Run ESM2 embedding script
python3 worker_encoder_ESM2.py \
    --csv pdbbind_canonical_affinities.csv \
    --col seq \
    --outdir esm2_embeddings \
    --batch_size 8 \
    --method mean

echo "========================================"
echo "Job complete!"
echo "Embeddings saved in esm2_embeddings/"
echo "========================================"