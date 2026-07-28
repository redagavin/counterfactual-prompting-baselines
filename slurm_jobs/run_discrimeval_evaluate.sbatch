#!/bin/bash
# ABOUTME: DiscrimEval log-odds evaluation SLURM job (optionally a data-parallel array)
# ABOUTME: Args: $1 model name, $2 paraphrases JSON path
#SBATCH --job-name=discrimeval_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=logs/discrimeval_eval_%A_%a.out
#SBATCH --error=logs/discrimeval_eval_%A_%a.err
#SBATCH --signal=INT@300

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
PARAPHRASES=${2:-"results/discrimeval/paraphrases.json"}
source ~/.bashrc
conda activate cot
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/discrimeval checkpoints/discrimeval logs
python scripts/run_discrimeval_evaluate.py \
    --model "${MODEL}" --paraphrases "${PARAPHRASES}" \
    --output_dir results/discrimeval --checkpoint_dir checkpoints/discrimeval
echo "exit $?"
