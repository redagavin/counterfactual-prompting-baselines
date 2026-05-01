#!/bin/bash
#SBATCH --job-name=simulation_v2
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/simulation_%j.out
#SBATCH --error=logs/simulation_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for Monte Carlo power simulation v2 (empirically grounded)
# ABOUTME: CPU-only job, iterates over 6 (question, model) conditions

echo "========================================"
echo "Power Simulation v2 (Empirically Grounded)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/.bashrc
conda activate cot

MEDPERTURB_DIR="${SLURM_SUBMIT_DIR}"
cd "${MEDPERTURB_DIR}"

mkdir -p results/simulation_v2

python -u code/simulation.py \
    --n-simulations 1000 \
    --n-bootstrap 1000 \
    --sigma-pert-max 3.0 \
    --sigma-pert-step 0.1 \
    --sigma-values 0.0 0.25 0.5 1.0 \
    --seed 42 \
    --n-workers 16 \
    --output-dir results/simulation_v2

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
