#!/bin/bash
#SBATCH -A GEN006
#SBATCH -J emb
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source ~/.bashrc
conda init
conda activate protgpt2

cd /lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb/XGB

#python pipeline_shap_xgb.py --embedding $1
python revised_pipeline_shap_xgb.py --embedding $1
