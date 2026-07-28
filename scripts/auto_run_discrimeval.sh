#!/bin/bash
# ABOUTME: Auto-run DiscrimEval: paraphrase (if missing) -> submit eval array -> merge shards -> analyze
# ABOUTME: Args: $1 model name (default 8B), $2 TOTAL_GPUS (1 for 8B, 4 for 70B)
set -eo pipefail
MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
TOTAL_GPUS=${2:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"
source ~/.bashrc
conda activate cot

MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
PARAPHRASES="results/discrimeval/paraphrases.json"
MERGED="results/discrimeval/discrimeval_eval_${MODEL_SHORT}.json"
XLSX="results/discrimeval/discrimeval_analysis_${MODEL_SHORT}.xlsx"

# 1. Paraphrases (model-independent; generate once)
if [ ! -f "${PARAPHRASES}" ]; then
    echo "Generating paraphrases..."
    python scripts/run_discrimeval_paraphrase.py --output "${PARAPHRASES}"
fi

# 2. Submit eval as a SLURM array of TOTAL_GPUS tasks; wait for completion
JOB=$(sbatch --array=0-$((TOTAL_GPUS - 1)) slurm_jobs/run_discrimeval_evaluate.sbatch "${MODEL}" "${PARAPHRASES}" | awk '{print $NF}')
echo "Submitted eval array job ${JOB} (${TOTAL_GPUS} task(s)); waiting..."
while squeue -j "${JOB}" -h 2>/dev/null | grep -q .; do sleep 60; done

# 2b. Verify every expected shard was produced; abort loudly if any task failed.
MISSING=0
for g in $(seq 0 $((TOTAL_GPUS - 1))); do
    SHARD="results/discrimeval/discrimeval_eval_${MODEL_SHORT}_gpu${g}_of_${TOTAL_GPUS}.json"
    if [ ! -f "${SHARD}" ]; then
        echo "FATAL: missing shard ${SHARD} — eval task ${g} did not complete." >&2
        MISSING=1
    fi
done
if [ "${MISSING}" -ne 0 ]; then
    echo "FATAL: not all ${TOTAL_GPUS} eval shards present; aborting before merge/analysis." >&2
    exit 1
fi

# 3. Merge shards, deduping by decision_question_id
python3 - "${MODEL_SHORT}" "${TOTAL_GPUS}" "${MERGED}" <<'PYEOF'
import glob, json, os, sys
short, total, merged_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
shards = sorted(glob.glob(f"results/discrimeval/discrimeval_eval_{short}_gpu*_of_{total}.json"))
seen, merged = set(), []
for f in shards:
    with open(f) as fh:
        data = json.load(fh)
    for item in data:
        cid = item["decision_question_id"]
        if cid not in seen:
            seen.add(cid); merged.append(item)
if not merged:
    sys.stderr.write("FATAL: zero scenarios after merge; aborting.\n")
    sys.exit(1)
with open(merged_path, "w") as fh:
    json.dump(merged, fh, indent=2)
print(f"Merged {len(merged)} scenarios from {len(shards)} shard(s) -> {merged_path}")
PYEOF

# 4. Analyze -> xlsx
python3 -c "import sys; sys.path.insert(0, 'src'); from discrimeval_analysis import run_analysis; run_analysis('${MERGED}', '${XLSX}')"
echo "Done: ${XLSX}"
