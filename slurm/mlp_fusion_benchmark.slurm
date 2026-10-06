#!/bin/bash
#SBATCH -A GEN006
#SBATCH -J mlp_fusion
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --array=0-19%5
#SBATCH -o logs/fusion_%A_%a.out
#SBATCH -e logs/fusion_%A_%a.err

# --- CRITICAL ORNL AMD GPU MODULES ---
module purge
module load rocm

cd /lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb/XGB

mkdir -p logs
mkdir -p fusion_results

module purge
module load rocm

export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

## ENVIRONMENT SETUP ##########
source $(conda info --base)/etc/profile.d/conda.sh
conda activate protgpt2

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

TASK=${SLURM_ARRAY_TASK_ID}

# 4 strategies for each representation
MODEL_ID=$(( TASK / 4 ))
STRATEGY_ID=$(( TASK % 4 ))

echo "=========================================="
echo "ARRAY TASK  = ${TASK}"
echo "MODEL ID    = ${MODEL_ID}"
echo "STRATEGY ID = ${STRATEGY_ID}"
echo "=========================================="

python mlp_fusion_benchmark.py \
    --task_id ${MODEL_ID} \
    --strategy ${STRATEGY_ID}