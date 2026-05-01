#!/bin/bash
# ABOUTME: Orchestrates the full Q-Pain race bias pipeline (3 models x 2 comparisons)
# ABOUTME: Runs paraphrase generation, evaluation, and analysis for Qwen3-8B, Qwen3-32B, GPT-2 Large

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p results/qpain checkpoints/qpain logs

# All srun jobs need conda activated. Use persistent runner script.
RUNNER="$(cd "$(dirname "$0")" && pwd)/srun_runner.sh"

# Step 1: Generate 4 paraphrase sets (Black/Asian x Qwen3/GPT2 tokenizers)
echo "=== Step 1: Generating calibrated paraphrases ==="

generate_paraphrases() {
    local tokenizer_name="$1"
    local comparison="$2"  # "black" or "asian"
    local output_path="$3"

    if [ -f "$output_path" ]; then
        echo "Paraphrases already exist at $output_path, skipping"
        return
    fi

    local rename_snippet=""
    if [ "$comparison" = "asian" ]; then
        rename_snippet="df = df.drop(columns=['black_text']).rename(columns={'asian_text': 'black_text'})"
    fi

    srun --partition=frink --time=01:00:00 --mem=32G --cpus-per-task=4 \
        --job-name=qpain_para_${comparison}_${tokenizer_name##*/} \
        "$RUNNER" "
import sys; sys.path.insert(0, 'src')
import asyncio
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from qpain_data import prepare_dataset
from qpain_paraphrase import generate_all_paraphrases

tokenizer = AutoTokenizer.from_pretrained('${tokenizer_name}')
client = AsyncOpenAI()
df = prepare_dataset(seed=42)
${rename_snippet}

asyncio.run(generate_all_paraphrases(
    df, '${output_path}',
    tokenizer, client, max_concurrent=50, checkpoint_freq=10,
))
"
}

generate_paraphrases "Qwen/Qwen3-8B" "black" "results/qpain/qpain_paraphrases.json"
generate_paraphrases "Qwen/Qwen3-8B" "asian" "results/qpain/qpain_paraphrases_asian.json"
generate_paraphrases "gpt2-large" "black" "results/qpain/qpain_paraphrases_black_gpt2.json"
generate_paraphrases "gpt2-large" "asian" "results/qpain/qpain_paraphrases_asian_gpt2.json"

# Step 2: Run 3 evaluation jobs in parallel
echo "=== Step 2: Running evaluation jobs ==="

QPAIN_EVAL_SNIPPET='
import sys; sys.path.insert(0, "src")
import os
from qpain_data import prepare_dataset
from qpain_evaluate import QpainEvaluator, run_evaluation

paraphrase_path = "results/qpain/qpain_paraphrases.json"
if not os.path.exists(paraphrase_path): paraphrase_path = None
asian_paraphrase_path = "results/qpain/qpain_paraphrases_asian.json"
if not os.path.exists(asian_paraphrase_path): asian_paraphrase_path = None

df = prepare_dataset(seed=42, paraphrase_path=paraphrase_path, asian_paraphrase_path=asian_paraphrase_path)
samples = df.to_dict("records")
'

# Qwen3-8B (background) — fits on A100
srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=160G --time=01:00:00 \
    --job-name=qpain_eval_qwen3_8b \
    "$RUNNER" "${QPAIN_EVAL_SNIPPET}
evaluator = QpainEvaluator('Qwen/Qwen3-8B')
run_evaluation(evaluator, samples,
    output_path='results/qpain/qpain_eval_qwen3_8b.json',
    checkpoint_path='checkpoints/qpain/qpain_eval_qwen3_8b.pkl',
    checkpoint_freq=5)
" &
PID_8B=$!

# Qwen3-32B (background)
srun --partition=gpu --gres=gpu:h200:1 --cpus-per-task=8 --mem=160G --time=01:00:00 \
    --job-name=qpain_eval_qwen3_32b \
    "$RUNNER" "${QPAIN_EVAL_SNIPPET}
evaluator = QpainEvaluator('Qwen/Qwen3-32B')
run_evaluation(evaluator, samples,
    output_path='results/qpain/qpain_eval_qwen3_32b.json',
    checkpoint_path='checkpoints/qpain/qpain_eval_qwen3_32b.pkl',
    checkpoint_freq=5)
" &
PID_32B=$!

# GPT-2 Large (background) — fits on A100
srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
    --job-name=qpain_eval_gpt2_large \
    "$RUNNER" "
import sys; sys.path.insert(0, 'src')
import os
from qpain_data import prepare_dataset
from qpain_gpt2_evaluate import Gpt2QpainEvaluator, load_closed_data, run_gpt2_evaluation

para_black = 'results/qpain/qpain_paraphrases_black_gpt2.json'
if not os.path.exists(para_black): para_black = None
para_asian = 'results/qpain/qpain_paraphrases_asian_gpt2.json'
if not os.path.exists(para_asian): para_asian = None

df = prepare_dataset(seed=42, paraphrase_path=para_black, asian_paraphrase_path=para_asian)
samples = df.to_dict('records')
closed_data = load_closed_data()
evaluator = Gpt2QpainEvaluator('gpt2-large')
run_gpt2_evaluation(evaluator, samples, closed_data,
    output_path='results/qpain/qpain_eval_gpt2_large.json',
    checkpoint_path='checkpoints/qpain/qpain_eval_gpt2_large.pkl',
    checkpoint_freq=5)
" &
PID_GPT2=$!

echo "Waiting for all 3 eval jobs to finish..."
wait $PID_8B $PID_32B $PID_GPT2
echo "All evaluations complete."

# Step 3: Run analysis (6 combinations: 3 models x 2 comparisons)
echo "=== Step 3: Running analysis ==="

run_analysis() {
    local model="$1"
    local comparison="$2"
    local eval_path="results/qpain/qpain_eval_${model}.json"
    local output_path="results/qpain/qpain_analysis_${model}_${comparison}.xlsx"

    srun --partition=frink --time=00:10:00 --mem=8G \
        --job-name=qpain_analysis_${model}_${comparison} \
        "$RUNNER" "
import sys; sys.path.insert(0, 'src')
from qpain_analysis import load_eval_results, run_qpain_analysis
results = load_eval_results('${eval_path}')
run_qpain_analysis(results, '${output_path}', comparison='${comparison}')
"
}

run_analysis "qwen3_8b" "black"
run_analysis "qwen3_8b" "asian"
run_analysis "qwen3_32b" "black"
run_analysis "qwen3_32b" "asian"
run_analysis "gpt2_large" "black"
run_analysis "gpt2_large" "asian"

echo "=== All done ==="
