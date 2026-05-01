#!/bin/bash
#SBATCH --job-name=qpain_replication
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/qpain_replication_%j.out
#SBATCH --error=logs/qpain_replication_%j.err
#SBATCH --signal=INT@300

source activate cot
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"

python -c "
from qpain_replication_data import load_context_data, prepare_dataset, compute_comparisons
from qpain_replication_evaluate import Gpt2ReplicationEvaluator, run_evaluation
import json

dataset = prepare_dataset()
comparisons = compute_comparisons()
closed_data = load_context_data()

# Load paraphrases if available
paraphrases = None
para_path = 'results/qpain_replication/paraphrases.json'
import os
if os.path.exists(para_path):
    with open(para_path) as f:
        paraphrases = json.load(f)

evaluator = Gpt2ReplicationEvaluator()
run_evaluation(
    evaluator, dataset, comparisons, closed_data,
    output_path='results/qpain_replication/eval_results.json',
    checkpoint_path='checkpoints/qpain_replication/eval.pkl',
    paraphrases=paraphrases,
)
"
