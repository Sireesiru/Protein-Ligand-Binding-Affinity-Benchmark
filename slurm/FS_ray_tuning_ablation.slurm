#!/bin/bash
#SBATCH -A GEN006
#SBATCH -J miniffinity_ablation
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH -o logs/%x_%j_%a.out
#SBATCH -e logs/%x_%j_%a.err

# --- CRITICAL ORNL AMD GPU MODULES ---
module purge
module load rocm
module load miniforge3

# --- RAY AMD GPU FIX ---
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# --- ENVIRONMENT SETUP ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate protgpt2

# --- WORKING DIRECTORY ---
cd /lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb/XGB || exit 1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "============================================================"
echo "Affinity Ablation Study"
echo "Job ID       : $SLURM_JOB_ID"
echo "Array Task   : $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Python       : $(which python)"
echo "Conda env    : $CONDA_DEFAULT_ENV"
echo "Working dir  : $(pwd)"
echo "Start time   : $(date)"
echo "============================================================"

echo "Visible GPUs:"
rocm-smi

# ============================================================
# TASKS
# ============================================================

case $SLURM_ARRAY_TASK_ID in

    0)
        echo "Running ESM2 -- PROTEIN ONLY"
        python FS_ray_tuning_ablation.py --task_id 0 --mode protein
        ;;

    1)
        echo "Running ProstT5 -- PROTEIN ONLY"
        python FS_ray_tuning_ablation.py --task_id 1 --mode protein
        ;;

    2)
        echo "Running ProGen2 -- PROTEIN ONLY"
        python FS_ray_tuning_ablation.py --task_id 2 --mode protein
        ;;

    3)
        echo "Running Boltz -- PROTEIN ONLY"
        python FS_ray_tuning_ablation.py --task_id 3 --mode protein
        ;;

    4)
        echo "Running ProtGPT2 -- PROTEIN ONLY"
        python FS_ray_tuning_ablation.py --task_id 4 --mode protein
        ;;

    5)
        echo "Running TxGemma -- LIGAND ONLY"
        python FS_ray_tuning_ablation.py --task_id 0 --mode ligand
        ;;

    *)
        echo "ERROR: Unknown array task $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

STATUS=$?

echo "============================================================"
echo "Finished task : $SLURM_ARRAY_TASK_ID"
echo "Exit status   : $STATUS"
echo "End time      : $(date)"
echo "============================================================"

exit $STATUS