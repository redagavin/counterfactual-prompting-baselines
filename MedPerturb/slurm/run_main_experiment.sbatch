#!/bin/bash
#SBATCH --job-name=main_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/main_experiment_%A_%a.out
#SBATCH --error=logs/main_experiment_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for main experiment evaluation with data-parallel array jobs
# ABOUTME: 4 array tasks, each with 1 H200 GPU, 12h time limit

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASET=${2:-"data_with_baselines.csv"}

echo "======================================"
echo "Main Experiment Evaluation"
echo "======================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Started at: $(date)"
echo ""

nvidia-smi

source ~/.bashrc
conda activate cot

MEDPERTURB_DIR="${SLURM_SUBMIT_DIR}"
cd "${MEDPERTURB_DIR}"

mkdir -p logs
mkdir -p checkpoints/main

python code/main_evaluate.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --output_dir "results" \
    --checkpoint_dir "checkpoints/main" \
    --checkpoint_freq 5

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================"

exit ${EXIT_CODE}
