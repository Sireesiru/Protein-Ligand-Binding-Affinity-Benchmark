#!/bin/bash
#SBATCH -A GEN006
#SBATCH -J affinity_repeat
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --array=0-49%5
#SBATCH -o logs/repeated_seed/slurm_%A_%a.out
#SBATCH -e logs/repeated_seed/slurm_%A_%a.err

# ============================================================
# Frontier repeated-seed MLP experiment
#
# 50 total tasks:
#   5 protein representations
#   x
#   10 seeds
#
# Maximum 5 jobs running simultaneously.
# ============================================================

cd /lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb/XGB

mkdir -p logs/repeated_seed

# --- CRITICAL ORNL AMD GPU MODULES ---
module purge
module load rocm

# --- RAY / AMD GPU FIX ---
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# --- ENVIRONMENT SETUP ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate protgpt2

# --- CPU THREAD CONTROL ---
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "=================================================="
echo "SLURM JOB ID       : ${SLURM_JOB_ID}"
echo "SLURM ARRAY JOB ID : ${SLURM_ARRAY_JOB_ID}"
echo "ARRAY TASK ID      : ${SLURM_ARRAY_TASK_ID}"
echo "NODE                : $(hostname)"
echo "CONDA ENV           : ${CONDA_DEFAULT_ENV}"
echo "PYTHON              : $(which python)"
echo "START               : $(date)"
echo "=================================================="

echo "Visible GPUs:"
rocm-smi

python repeated_seed_mlp.py \
    --task_id ${SLURM_ARRAY_TASK_ID}

status=$?

echo "=================================================="
echo "END                 : $(date)"
echo "EXIT STATUS         : ${status}"
echo "=================================================="

exit ${status}