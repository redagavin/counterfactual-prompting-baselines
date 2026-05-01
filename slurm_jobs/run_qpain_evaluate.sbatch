#!/bin/bash
# ABOUTME: SLURM script for Q-Pain race bias evaluation with Qwen3 models
# ABOUTME: Runs model evaluation on 50 vignettes with checkpointing and paraphrase loading
#SBATCH --job-name=qpain_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qpain_eval_%j.out
#SBATCH --error=logs/qpain_eval_%j.err
#SBATCH --signal=INT@300

source ~/.bashrc
conda activate cot
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs checkpoints/qpain results/qpain

MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=${MODEL_SHORT:-"qwen3_8b"}
PARAPHRASE_PATH=${PARAPHRASE_PATH:-"results/qpain/qpain_paraphrases.json"}
ASIAN_PARAPHRASE_PATH=${ASIAN_PARAPHRASE_PATH:-"results/qpain/qpain_paraphrases_asian.json"}

python -c "
import sys
sys.path.insert(0, 'src')
from qpain_data import prepare_dataset
from qpain_evaluate import QpainEvaluator, run_evaluation

import os
paraphrase_path = '${PARAPHRASE_PATH}'
if not os.path.exists(paraphrase_path):
    print(f'WARNING: paraphrase file not found at {paraphrase_path}, skipping')
    paraphrase_path = None

asian_paraphrase_path = '${ASIAN_PARAPHRASE_PATH}'
if not os.path.exists(asian_paraphrase_path):
    print(f'WARNING: Asian paraphrase file not found at {asian_paraphrase_path}, skipping')
    asian_paraphrase_path = None

df = prepare_dataset(seed=42, paraphrase_path=paraphrase_path, asian_paraphrase_path=asian_paraphrase_path)
samples = df.to_dict('records')

evaluator = QpainEvaluator('${MODEL}')
run_evaluation(
    evaluator, samples,
    output_path='results/qpain/qpain_eval_${MODEL_SHORT}.json',
    checkpoint_path='checkpoints/qpain/qpain_eval_${MODEL_SHORT}.pkl',
    checkpoint_freq=5,
)
"
