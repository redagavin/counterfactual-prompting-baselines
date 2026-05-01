#!/bin/bash
# ABOUTME: Automated precision +1 age pipeline with auto-relaunch
# ABOUTME: Runs age +1 evaluation with model-tagged shards, merge, and analysis

set -eo pipefail

MAX_ITERATIONS=150

# Defaults
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
JOB_NAME="precision_plus1"
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "======================================"
echo "Precision +1 Age Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results logs checkpoints/precision_plus1

# ==========================================
# STEP 1: Generate Age +1 JSON
# ==========================================
echo "======================================"
echo "STEP 1: Generate Age +1 JSON"
echo "======================================"

AGE_PLUS1_PATH="results/precision_plus1_age_plus1.json"

if [ -f "${AGE_PLUS1_PATH}" ]; then
    ENTRY_COUNT=$(python -c "import json; print(len(json.load(open('${AGE_PLUS1_PATH}'))))")
    echo "Age +1 file exists with ${ENTRY_COUNT} entries, skipping generation"
else
    srun --partition=frink --cpus-per-task=4 --mem=16G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && cd ${MEDPERTURB_DIR} && python code/precision_check_age_plus1.py \
            --dataset data_with_baselines.csv \
            --output '${AGE_PLUS1_PATH}' \
            --model '${MODEL}'"

    ENTRY_COUNT=$(python -c "import json; print(len(json.load(open('${AGE_PLUS1_PATH}'))))")
    echo "Age +1 generation complete: ${ENTRY_COUNT} entries"
fi

echo ""

# ==========================================
# STEP 1.5: Generate Calibrated Baselines
# ==========================================
echo "======================================"
echo "STEP 1.5: Generate Calibrated Baselines"
echo "======================================"

BASELINES_PATH="results/precision_check_baselines.json"

if [ -f "${BASELINES_PATH}" ]; then
    BASELINE_COUNT=$(python -c "import json; print(len(json.load(open('${BASELINES_PATH}'))))")
    echo "Baselines file exists with ${BASELINE_COUNT} entries, skipping generation"
else
    srun --partition=frink --cpus-per-task=4 --mem=16G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && cd ${MEDPERTURB_DIR} && python code/precision_check_baselines.py \
            --age_swap '${AGE_PLUS1_PATH}' \
            --dataset data_with_baselines.csv \
            --output '${BASELINES_PATH}' \
            --model '${MODEL}'"

    BASELINE_COUNT=$(python -c "import json; print(len(json.load(open('${BASELINES_PATH}'))))")
    echo "Baselines generation complete: ${BASELINE_COUNT} entries"
fi

echo ""

# ==========================================
# STEP 2: Run Evaluation
# ==========================================
echo "======================================"
echo "STEP 2: Age +1 Evaluation"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    srun --partition=gpu --gres=gpu:h200:1 --cpus-per-task=8 --mem=160G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/precision_check_evaluate.py \
            --model ${MODEL} \
            --dataset data_with_baselines.csv \
            --scenario precision_plus1 \
            --age_swap results/precision_plus1_age_plus1.json \
            --baselines results/precision_check_baselines.json \
            --output_dir results \
            --checkpoint_dir checkpoints/precision_plus1 \
            --gpu_id 0 \
            --total_gpus 1 \
            --sample_size 5"
else
    # Production: sbatch with auto-relaunch
    check_completion() {
        COMPLETE_COUNT=$(ls -1 checkpoints/precision_plus1/precision_plus1_eval_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        [ "$COMPLETE_COUNT" -eq $TOTAL_GPUS ]
    }

    check_jobs_running() {
        RUNNING_JOBS=$(squeue -u $USER -n ${JOB_NAME} -h 2>/dev/null | wc -l)
        [ "$RUNNING_JOBS" -gt 0 ]
    }

    ITERATION=1
    while [ $ITERATION -le $MAX_ITERATIONS ]; do
        echo "[$(date)] Iteration ${ITERATION}"

        if check_completion; then
            echo "All ${TOTAL_GPUS}/${TOTAL_GPUS} GPUs complete!"
            break
        fi

        COMPLETE_COUNT=$(ls -1 checkpoints/precision_plus1/precision_plus1_eval_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l) || true
        echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

        if check_jobs_running; then
            squeue -u $USER -n ${JOB_NAME}
            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep $CHECK_INTERVAL
        else
            echo "Launching jobs..."
            SUBMITTED=$(sbatch slurm/run_precision_plus1.sbatch "${MODEL}" | awk '{print $NF}')
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

    if ! check_completion; then
        echo "FATAL: exceeded ${MAX_ITERATIONS} iterations without completion"
        exit 1
    fi
fi

# ==========================================
# STEP 3: Merge Results
# ==========================================
echo "======================================"
echo "STEP 3: Merge Results"
echo "======================================"

MERGED_OUTPUT="results/precision_plus1_evaluation_${MODEL_SHORT}.json"

if [ "$TEST_MODE" = true ]; then
    TEST_FILE=$(ls -1 results/precision_plus1_eval_${MODEL_SHORT}_gpu0_of_1.json 2>/dev/null | head -1)
    if [ -n "${TEST_FILE}" ]; then
        cp "${TEST_FILE}" "${MERGED_OUTPUT}"
        echo "Test mode: copied to ${MERGED_OUTPUT}"
    else
        echo "Warning: No test result file found"
    fi
else
    python << EOF
import json
import glob

result_files = glob.glob('results/precision_plus1_eval_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}.json')

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

output = '${MERGED_OUTPUT}'
with open(output, 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} unique results to {output}")
EOF
fi

# ==========================================
# STEP 4: Run Analysis
# ==========================================
echo "======================================"
echo "STEP 4: Run Analysis"
echo "======================================"

ANALYSIS_OUTPUT="results/precision_plus1_analysis_${MODEL_SHORT}.xlsx"

python case_studies/precision_check_analysis.py \
    --evaluation "${MERGED_OUTPUT}" \
    --output "${ANALYSIS_OUTPUT}"

echo "Analysis complete: ${ANALYSIS_OUTPUT}"

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Evaluation: ${MERGED_OUTPUT}"
echo "Analysis: ${ANALYSIS_OUTPUT}"
echo "Completed at: $(date)"
echo "======================================"
