#!/bin/bash
# ABOUTME: Automated dose-response pipeline with auto-relaunch
# ABOUTME: Runs paraphrase generation, model evaluation, and analysis

set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDPERTURB_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${MEDPERTURB_DIR}"

TOTAL_GPUS=4
CHECK_INTERVAL=300
JOB_NAME="dose_response"
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "======================================"
echo "Dose-Response Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results logs checkpoints/dose_response

# ==========================================
# STEP 1: Generate Paraphrases
# ==========================================
echo "======================================"
echo "STEP 1: Generate Paraphrases"
echo "======================================"

SAMPLE_SIZE_ARG=""
if [ "$TEST_MODE" = true ]; then
    SAMPLE_SIZE_ARG="--sample_size 5"
fi

GENERATE_CMD="source ~/.bashrc && conda activate cot && cd ${MEDPERTURB_DIR} && \
    python code/generate_dose_response_paraphrases.py \
        --dataset data.csv \
        --output results/dose_response_paraphrases.json \
        ${SAMPLE_SIZE_ARG}"

if [ -f "results/dose_response_paraphrases.json" ]; then
    PARA_COUNT=$(python -c "import json; print(len(json.load(open('results/dose_response_paraphrases.json'))))")
    EXPECTED=500
    if [ "$PARA_COUNT" -ge "$EXPECTED" ]; then
        echo "Paraphrases complete (${PARA_COUNT}/${EXPECTED}) — skipping"
    else
        echo "Paraphrases incomplete (${PARA_COUNT}/${EXPECTED}) — resuming"
        srun --partition=frink --cpus-per-task=4 --mem=16G --time=4:00:00 \
            bash -c "${GENERATE_CMD}"
    fi
else
    srun --partition=frink --cpus-per-task=4 --mem=16G --time=4:00:00 \
        bash -c "${GENERATE_CMD}"
fi

# ==========================================
# STEP 2: Run Evaluation
# ==========================================
echo "======================================"
echo "STEP 2: Model Evaluation"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    srun --partition=frink --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && cd ${MEDPERTURB_DIR} && \
        python code/dose_response_evaluate.py \
            --model '${MODEL}' \
            --paraphrases results/dose_response_paraphrases.json \
            --dataset data.csv \
            --output results/dose_response_eval_test.json \
            --checkpoint_dir checkpoints/dose_response \
            --checkpoint_freq 1 \
            --gpu_id 0 \
            --total_gpus 1 \
            --sample_size 5"
else
    check_completion() {
        COMPLETE_COUNT=$(ls -1 checkpoints/dose_response/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        [ "$COMPLETE_COUNT" -eq $TOTAL_GPUS ]
    }

    check_jobs_running() {
        RUNNING_JOBS=$(squeue -u $USER -n ${JOB_NAME} -h 2>/dev/null | wc -l)
        [ "$RUNNING_JOBS" -gt 0 ]
    }

    ITERATION=1
    while true; do
        echo "[$(date)] Iteration ${ITERATION}"

        if check_completion; then
            echo "All ${TOTAL_GPUS}/${TOTAL_GPUS} GPUs complete!"
            break
        fi

        COMPLETE_COUNT=$(ls -1 checkpoints/dose_response/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

        if check_jobs_running; then
            squeue -u $USER -n ${JOB_NAME}
            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep $CHECK_INTERVAL
        else
            echo "Launching jobs..."
            SUBMITTED=$(sbatch slurm/run_dose_response.sbatch "${MODEL}" | awk '{print $NF}')
            echo "Submitted job: ${SUBMITTED}"
            sleep 60

            if check_jobs_running; then
                echo "Jobs started"
                squeue -u $USER -n ${JOB_NAME}
            else
                echo "Warning: Jobs may not have started"
            fi

            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep $CHECK_INTERVAL
        fi

        ITERATION=$((ITERATION + 1))
    done
fi

# ==========================================
# STEP 3: Merge Results
# ==========================================
echo "======================================"
echo "STEP 3: Merge Results"
echo "======================================"

MERGED_OUTPUT="results/dose_response_evaluation_${MODEL_SHORT}.json"

if [ "$TEST_MODE" = true ]; then
    cp results/dose_response_eval_test.json "${MERGED_OUTPUT}"
    echo "Test mode: copied to ${MERGED_OUTPUT}"
else
    export MODEL_SHORT
    python << 'MERGE_EOF'
import json
import glob
import os
import shutil
import tempfile

model_short = os.environ.get('MODEL_SHORT', 'llama_3.1_8b_instruct')
result_files = glob.glob(f'results/dose_response_eval_{model_short}_*.json')
result_files = [f for f in result_files if 'test' not in f]

print(f"Found {len(result_files)} result files")

seen_ids = set()
merged = []
for f in sorted(result_files):
    with open(f, 'r') as fp:
        data = json.load(fp)
    for item in data:
        cid = item['context_id']
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(item)

output = f'results/dose_response_evaluation_{model_short}.json'
with tempfile.NamedTemporaryFile('w', delete=False, dir='results', suffix='.tmp') as f:
    json.dump(merged, f, indent=2)
    temp_path = f.name
shutil.move(temp_path, output)

print(f"Merged {len(merged)} unique results to {output}")
MERGE_EOF
fi

# ==========================================
# STEP 4: Run Analysis
# ==========================================
echo "======================================"
echo "STEP 4: Analysis & Plots"
echo "======================================"

srun --partition=frink --cpus-per-task=4 --mem=16G --time=1:00:00 \
    bash -c "source ~/.bashrc && conda activate cot && cd ${MEDPERTURB_DIR} && \
    python case_studies/dose_response_analysis.py \
        --evaluation '${MERGED_OUTPUT}' \
        --output results/dose_response_analysis.xlsx \
        --plot_prefix results/dose_response"

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Paraphrases: results/dose_response_paraphrases.json"
echo "Evaluation: ${MERGED_OUTPUT}"
echo "Analysis: results/dose_response_analysis.xlsx"
echo "Plots: results/dose_response_flip_rate.png, results/dose_response_mi.png"
echo "Completed at: $(date)"
echo "======================================"
