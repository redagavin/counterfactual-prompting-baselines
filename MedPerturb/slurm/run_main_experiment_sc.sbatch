#!/bin/bash
#SBATCH --job-name=sc_main_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/sc_main_eval_%A_%a.out
#SBATCH --error=logs/sc_main_eval_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SC main-experiment evaluation: 4-way data-parallel array job
# ABOUTME: Each task gets 1 H200, processes ~25 cases; merge shards afterward

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
BATCH_SIZE=${2:-4}
TEMPERATURE=${3:-0.7}
MEDPERTURB_DIR="${SLURM_SUBMIT_DIR}"

source ~/.bashrc
conda activate cot
cd "${MEDPERTURB_DIR}"
mkdir -p results checkpoints/main_experiment_sc logs

echo "========================================"
echo "SC Eval: Array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"
echo "Model: ${MODEL}, Temperature: ${TEMPERATURE}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

python code/main_evaluate_sc.py \
    --model "${MODEL}" \
    --dataset data_with_baselines.csv \
    --output_dir results \
    --checkpoint_dir checkpoints/main_experiment_sc \
    --n_samples 10 \
    --batch_size ${BATCH_SIZE} \
    --temperature ${TEMPERATURE} \
    --max_new_tokens 512 \
    --rng_seed 42

EXIT_CODE=$?
echo "Completed at: $(date), exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
