#!/bin/bash
#SBATCH --job-name=bib_paraphrase
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bib_paraphrase_%j.out
#SBATCH --error=logs/bib_paraphrase_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for generating calibrated paraphrases for bias-in-bios bios
# ABOUTME: CPU-only job using AsyncOpenAI with GPT-5.2 and Qwen3-8B tokenizer

SPLIT=${1:-"train"}

echo "========================================"
echo "Bias-in-Bios Calibrated Paraphrase Generation"
echo "Split: ${SPLIT}"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/.bashrc
conda activate cot
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/bib

python -c "
import sys, asyncio
sys.path.insert(0, 'src')
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from bib_data import prepare_split
from bib_paraphrase import generate_all_paraphrases

split = '${SPLIT}'

async def main():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    client = AsyncOpenAI()
    balanced_df = prepare_split(split)
    output_path = f'results/bib/bib_paraphrases_{split}.json'

    await generate_all_paraphrases(
        balanced_df=balanced_df,
        output_path=output_path,
        tokenizer=tokenizer,
        openai_client=client,
        max_concurrent=300,
        checkpoint_freq=50,
    )

asyncio.run(main())
"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
