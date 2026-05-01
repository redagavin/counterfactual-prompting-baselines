#!/bin/bash
# ABOUTME: Orchestration script for bias-in-bios experiment
# ABOUTME: Generates paraphrases, runs evaluation for both models and splits, then analysis
set -eo pipefail

MAX_ITERATIONS=50
TOTAL_GPUS=4
CHECK_INTERVAL=300

MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-32B")
SPLITS=("train" "test")

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs checkpoints/bib results/bib

# STEP 1: Generate paraphrases (CPU, both splits)
for SPLIT in "${SPLITS[@]}"; do
    PARA_FILE="results/bib/bib_paraphrases_${SPLIT}.json"
    if [ -f "$PARA_FILE" ]; then
        echo "Paraphrases already exist for $SPLIT"
    else
        echo "Generating paraphrases for $SPLIT..."
        PARA_JOB=$(sbatch slurm_jobs/run_bib_paraphrase.sbatch "$SPLIT" | awk '{print $NF}')
        echo "Submitted paraphrase job $PARA_JOB for $SPLIT, waiting..."
        while squeue -j "$PARA_JOB" -h 2>/dev/null | grep -q .; do
            sleep 60
        done
        if [ ! -f "$PARA_FILE" ]; then
            echo "ERROR: Paraphrase file not created for $SPLIT"
            exit 1
        fi
        echo "Paraphrases complete for $SPLIT"
    fi
done

# STEP 2: GPU evaluation
check_completion() {
    local model_short=$1
    local split=$2
    local count=$(ls -1 results/bib/bib_eval_${split}_${model_short}_gpu*_of_${TOTAL_GPUS}.json 2>/dev/null | wc -l)
    [ "$count" -eq $TOTAL_GPUS ]
}

merge_results() {
    local model_short=$1
    local split=$2
    python -c "
import json, glob, sys
sys.path.insert(0, 'src')
files = sorted(glob.glob('results/bib/bib_eval_${split}_${model_short}_gpu*_of_${TOTAL_GPUS}.json'))
all_results = []
seen_ids = set()
for f in files:
    with open(f) as fh:
        for r in json.load(fh):
            if r['bio_id'] not in seen_ids:
                all_results.append(r)
                seen_ids.add(r['bio_id'])
with open('results/bib/bib_eval_${split}_${model_short}_merged.json', 'w') as fh:
    json.dump(all_results, fh, indent=2)
print(f'Merged {len(all_results)} results for ${split}/${model_short}')
"
}

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | awk -F/ '{print $NF}' | tr '[:upper:]-' '[:lower:]_')
    for SPLIT in "${SPLITS[@]}"; do
        if check_completion "$MODEL_SHORT" "$SPLIT"; then
            echo "Already complete: $SPLIT/$MODEL_SHORT"
            merge_results "$MODEL_SHORT" "$SPLIT"
            continue
        fi
        echo "Starting evaluation: $SPLIT/$MODEL_SHORT"
        ITERATION=1
        while [ $ITERATION -le $MAX_ITERATIONS ]; do
            if check_completion "$MODEL_SHORT" "$SPLIT"; then
                merge_results "$MODEL_SHORT" "$SPLIT"
                break
            fi
            RUNNING=$(squeue -u $USER -n bib_eval -h 2>/dev/null | wc -l)
            if [ "$RUNNING" -gt 0 ]; then
                sleep $CHECK_INTERVAL
            else
                sbatch slurm_jobs/run_bib_evaluate.sbatch "$MODEL" "$SPLIT"
                sleep 60
                sleep $CHECK_INTERVAL
            fi
            ITERATION=$((ITERATION + 1))
        done

        # Verify completion after retry loop — fail if not done
        if ! check_completion "$MODEL_SHORT" "$SPLIT"; then
            echo "ERROR: evaluation did not complete for $SPLIT/$MODEL_SHORT after $MAX_ITERATIONS iterations"
            exit 1
        fi
    done

    echo "Running analysis for $MODEL_SHORT"
    srun --partition=short --mem=8G --cpus-per-task=2 --time=1:00:00 \
        python -c "
import sys
sys.path.insert(0, 'src')
from bib_analysis import run_full_analysis
run_full_analysis(
    'results/bib/bib_eval_train_${MODEL_SHORT}_merged.json',
    'results/bib/bib_eval_test_${MODEL_SHORT}_merged.json',
    'results/bib/bib_analysis_${MODEL_SHORT}.xlsx',
)
"
done

echo "All evaluations and analyses complete."
