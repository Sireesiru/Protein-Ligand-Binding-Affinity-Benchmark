#!/bin/bash
#SBATCH -A GEN006
#SBATCH -J mlp_shap
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --array=0-4%5
#SBATCH -o logs/mlp_shap_%A_%a.out
#SBATCH -e logs/mlp_shap_%A_%a.err

# --- CRITICAL ORNL AMD GPU MODULES ---
module purge
module load rocm

# --- RAY AMD GPU FIX ---
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# --- ENVIRONMENT SETUP ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate protgpt2

cd /lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb/XGB

mkdir -p logs
mkdir -p mlp_shap

export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Visible GPUs:"
rocm-smi

python mlp_modality_shap.py --task_id ${SLURM_ARRAY_TASK_ID}
#python FS_ray_tuning_modified.py --task_id $SLURM_ARRAY_TASK_ID
#python FS_ray_tuning.py --task_id $SLURM_ARRAY_TASK_ID