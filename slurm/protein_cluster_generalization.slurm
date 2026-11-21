#!/bin/bash
#SBATCH -A GEN006
#SBATCH -J cluster30_gen
#SBATCH -o cluster30_gen_%A_%a.out
#SBATCH -e cluster30_gen_%A_%a.err

#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8

#SBATCH --array=0-4

# Frontier: one GCD per array task
#SBATCH --gpus-per-task=1


# ============================================================
# ENVIRONMENT
# ============================================================

source /ccs/home/sireesiru/.conda/etc/profile.d/conda.sh
conda activate /ccs/home/sireesiru/.conda/envs/protgpt2


# ============================================================
# WORKING DIRECTORY
# ============================================================

cd /lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb/XGB/protein_clustering || exit 1

# ============================================================
# FRONTIER / ROCm GPU
# ============================================================

export OMP_NUM_THREADS=8

echo "============================================================"
echo "Protein-cluster generalization benchmark"
echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Array task   : $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "============================================================"

echo
echo "Checking PyTorch GPU access..."

python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated GPU.")

if torch.cuda.device_count() < 1:
    raise RuntimeError("No GPU visible to PyTorch.")

print("GPU:", torch.cuda.get_device_name(0))
PY

if [ $? -ne 0 ]; then
    echo "GPU check FAILED."
    exit 1
fi


# ============================================================
# RUN
#
# task_id:
# 0 = ESM2
# 1 = ProstT5
# 2 = ProGen2
# 3 = Boltz
# 4 = ProtGPT2
# ============================================================

echo
echo "Starting representation task ${SLURM_ARRAY_TASK_ID}..."

python protein_cluster_generalization.py \
    --task_id ${SLURM_ARRAY_TASK_ID}

STATUS=$?

echo
echo "============================================================"
echo "Finished: $(date)"
echo "Exit code: $STATUS"
echo "============================================================"

exit $STATUS
