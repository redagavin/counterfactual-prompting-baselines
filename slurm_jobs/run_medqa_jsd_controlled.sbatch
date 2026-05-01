#!/bin/bash
#SBATCH --job-name=medqa_jsd_ctrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/medqa_jsd_ctrl_%A_%a.out
#SBATCH --error=logs/medqa_jsd_ctrl_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for MedQA JSD Controlled Perturbation Analysis
# ABOUTME: 4 data-parallel array tasks, each with 1 H200 GPU

cd "${SLURM_SUBMIT_DIR}"

echo "========================================"
echo "MedQA JSD Controlled Analysis"
echo "========================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: 4"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

nvidia-smi

echo ""
echo "Running controlled perturbation analysis..."
echo ""

# Activate conda environment
source ~/.bashrc
conda activate cot

# Run analysis
python src/medqa_jsd_controlled_analysis.py \
    --model deepseek_r1_0528 \
    --checkpoint_freq 10

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
