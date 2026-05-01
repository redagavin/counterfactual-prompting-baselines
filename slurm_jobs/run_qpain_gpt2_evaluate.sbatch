#!/bin/bash
# ABOUTME: SLURM script for GPT-2 Large Q-Pain evaluation with few-shot prompting
# ABOUTME: Runs GPT-2 Large (774M) on 50 vignettes with checkpointing
#SBATCH --job-name=qpain_gpt2_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qpain_gpt2_eval_%j.out
#SBATCH --error=logs/qpain_gpt2_eval_%j.err
#SBATCH --signal=INT@300

source ~/.bashrc
conda activate cot
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs checkpoints/qpain results/qpain

PARAPHRASE_BLACK=${PARAPHRASE_BLACK:-"results/qpain/qpain_paraphrases_black_gpt2.json"}
PARAPHRASE_ASIAN=${PARAPHRASE_ASIAN:-"results/qpain/qpain_paraphrases_asian_gpt2.json"}

python -c "
import sys
sys.path.insert(0, 'src')
import os
from qpain_data import prepare_dataset
from qpain_gpt2_evaluate import Gpt2QpainEvaluator, load_closed_data, run_gpt2_evaluation

para_black = '${PARAPHRASE_BLACK}' if os.path.exists('${PARAPHRASE_BLACK}') else None
para_asian = '${PARAPHRASE_ASIAN}' if os.path.exists('${PARAPHRASE_ASIAN}') else None

df = prepare_dataset(seed=42, paraphrase_path=para_black, asian_paraphrase_path=para_asian)
samples = df.to_dict('records')
closed_data = load_closed_data()

evaluator = Gpt2QpainEvaluator('gpt2-large')
run_gpt2_evaluation(
    evaluator, samples, closed_data,
    output_path='results/qpain/qpain_eval_gpt2_large.json',
    checkpoint_path='checkpoints/qpain/qpain_eval_gpt2_large.pkl',
    checkpoint_freq=5,
)
"
