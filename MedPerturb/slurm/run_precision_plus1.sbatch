#!/bin/bash
#SBATCH --job-name=precision_plus1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/precision_plus1_%A_%a.out
#SBATCH --error=logs/precision_plus1_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for precision +1 evaluation with age +1 negative control
# ABOUTME: 4 data-parallel array tasks with Llama-3-8B

MEDPERTURB_DIR="${SLURM_SUBMIT_DIR}"

echo "========================================"
echo "Precision +1: Age +1 Evaluation"
echo "========================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "Start time: $(date)"
echo "Node: $(hostname)"

nvidia-smi

source ~/.bashrc
conda activate cot

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}

echo "Model: ${MODEL}"
echo ""

cd ${MEDPERTURB_DIR}

python code/precision_check_evaluate.py \
    --model "${MODEL}" \
    --dataset "data_with_baselines.csv" \
    --age_swap "results/precision_plus1_age_plus1.json" \
    --baselines "results/precision_check_baselines.json" \
    --output_dir "results" \
    --checkpoint_dir "checkpoints/precision_plus1" \
    --scenario "precision_plus1"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
