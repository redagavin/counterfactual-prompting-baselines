#!/bin/bash
# ABOUTME: Orchestration script for Q-Pain replication experiment
# ABOUTME: Runs paraphrase generation, GPT-2 evaluation, and analysis sequentially
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${PROJ_DIR}/src"

# All srun jobs need conda activated. Use persistent runner script.
RUNNER="$(cd "$(dirname "$0")" && pwd)/srun_runner.sh"

RESULTS_DIR="results/qpain_replication"
CKPT_DIR="checkpoints/qpain_replication"
mkdir -p "$RESULTS_DIR" "$CKPT_DIR" logs

PARA_FILE="$RESULTS_DIR/paraphrases.json"
EVAL_FILE="$RESULTS_DIR/eval_results.json"
ANALYSIS_FILE="$RESULTS_DIR/analysis.xlsx"

# --- Stage 1: Paraphrase Generation ---
if [ ! -f "$PARA_FILE" ]; then
    echo "=== Stage 1: Generating calibrated paraphrases ==="
    srun --partition=frink --time=12:00:00 --mem=32G --cpus-per-task=4 \
         --signal=INT@300 --export=ALL \
         --output=logs/qpain_replication_para_%j.out \
         --error=logs/qpain_replication_para_%j.err \
         "$RUNNER" "
import asyncio
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from qpain_replication_data import prepare_dataset, compute_comparisons
from qpain_replication_paraphrase import compute_paraphrase_targets, generate_all_paraphrases

tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
dataset = prepare_dataset()
comparisons = compute_comparisons()
targets = compute_paraphrase_targets(dataset, comparisons, tokenizer)
client = AsyncOpenAI()
asyncio.run(generate_all_paraphrases(
    targets, tokenizer, client, '$PARA_FILE',
    max_concurrent=50, checkpoint_freq=10,
))
"
    echo "Paraphrases complete: $PARA_FILE"
else
    echo "Paraphrases already exist: $PARA_FILE"
fi

# --- Stage 2: GPT-2 Evaluation ---
if [ ! -f "$EVAL_FILE" ]; then
    echo "=== Stage 2: Running GPT-2 evaluation ==="
    srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G \
         --time=2:00:00 --signal=INT@300 --export=ALL \
         --output=logs/qpain_replication_%j.out \
         --error=logs/qpain_replication_%j.err \
         "$RUNNER" "
from qpain_replication_data import load_context_data, prepare_dataset, compute_comparisons
from qpain_replication_evaluate import Gpt2ReplicationEvaluator, run_evaluation
import json, os
dataset = prepare_dataset()
comparisons = compute_comparisons()
closed_data = load_context_data()
paraphrases = None
if os.path.exists('$PARA_FILE'):
    with open('$PARA_FILE') as f:
        paraphrases = json.load(f)
evaluator = Gpt2ReplicationEvaluator()
run_evaluation(evaluator, dataset, comparisons, closed_data,
    output_path='$EVAL_FILE',
    checkpoint_path='$CKPT_DIR/eval.pkl',
    paraphrases=paraphrases)
"
    echo "Evaluation complete: $EVAL_FILE"
else
    echo "Evaluation already exists: $EVAL_FILE"
fi

# --- Stage 3: Analysis ---
if [ ! -f "$ANALYSIS_FILE" ]; then
    echo "=== Stage 3: Running analysis ==="
    srun --partition=frink --time=00:10:00 --mem=8G \
         --signal=INT@300 --export=ALL \
         --output=logs/qpain_replication_analysis_%j.out \
         --error=logs/qpain_replication_analysis_%j.err \
         "$RUNNER" "
import json
from qpain_replication_data import compute_comparisons
from qpain_replication_analysis import run_all_comparisons, export_to_excel

with open('$EVAL_FILE') as f:
    eval_results = json.load(f)
comparisons = compute_comparisons()
results = run_all_comparisons(eval_results, comparisons)
export_to_excel(results, '$ANALYSIS_FILE')
print('Analysis complete: $ANALYSIS_FILE')
"
else
    echo "Analysis already exists: $ANALYSIS_FILE"
fi

echo "=== All stages complete ==="
