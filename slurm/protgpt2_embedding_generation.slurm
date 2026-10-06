#!/bin/bash
#SBATCH -A gen006
#SBATCH -J protgpt2_linear
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 04:00:00
#SBATCH -p extended      # <--- Using the partition you have access to
#SBATCH -o logs/extract_%j.out
#SBATCH -e logs/extract_%j.err

# Ensure directories exist
mkdir -p logs
mkdir -p protgpt2_embeddings

# Environment setup
module load miniforge3
conda activate protgpt2

# The "Safety" flag for Frontier
export PYTHONNOUSERSITE=1

echo "Starting Linear Extraction on partition: extended"
echo "Processing pdbbind_canonical_affinities.csv (19,118 rows)"

# Run your working python script
python3 extract_embeddings_new.py

# Move output to the designated folder
mv pdbbind_protgpt2_embeddings_FINAL.npy protgpt2_embeddings/
mv pdbbind_protgpt2_metadata_FINAL.csv protgpt2_embeddings/

echo "========================================"
echo "Job complete! Verification CSV and NPY are in protgpt2_embeddings/"
echo "========================================"