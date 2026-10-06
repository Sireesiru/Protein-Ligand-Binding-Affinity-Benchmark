#!/bin/bash
#SBATCH -A gen006
#SBATCH -J progen2_embed
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH --array=0-3
#SBATCH -o logs/progen2_%A_%a.out
#SBATCH -e logs/progen2_%A_%a.err

mkdir -p logs
mkdir -p progen2_embeddings

module load miniforge3
conda activate protgpt2

export PYTHONNOUSERSITE=1

echo "========================================"
echo "Starting ProGen2 chunk $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

python3 worker_decoder_progen2.py \
    --csv pdbbind_canonical_affinities.csv \
    --col seq \
    --outdir ./progen2_embeddings \
    --gpu 0 \
    --chunk $SLURM_ARRAY_TASK_ID \
    --num_chunks 4

echo "========================================"
echo "Finished chunk $SLURM_ARRAY_TASK_ID"
echo "Time: $(date)"
echo "========================================"