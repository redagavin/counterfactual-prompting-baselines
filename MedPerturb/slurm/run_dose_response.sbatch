#!/bin/bash
#SBATCH --job-name=dose_response
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/dose_response_%A_%a.out
#SBATCH --error=logs/dose_response_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for dose-response evaluation
# ABOUTME: 4 data-parallel array tasks with configurable model

MEDPERTURB_DIR="${SLURM_SUBMIT_DIR}"

echo "========================================"
echo "Dose-Response Evaluation"
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
MODEL_SHORT=$(echo "${MODEL}" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "Model: ${MODEL}"
echo "Model short: ${MODEL_SHORT}"
echo ""

cd "${MEDPERTURB_DIR}"

python code/dose_response_evaluate.py \
    --model "${MODEL}" \
    --paraphrases "results/dose_response_paraphrases.json" \
    --dataset "data.csv" \
    --output "results/dose_response_eval_${MODEL_SHORT}_${SLURM_ARRAY_TASK_ID}.json" \
    --checkpoint_dir "checkpoints/dose_response" \
    --checkpoint_freq 5

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
