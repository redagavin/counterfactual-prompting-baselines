#!/bin/bash
# ABOUTME: SLURM script for Qwen3 logit extraction on bias-in-bios balanced set
# ABOUTME: Array job sharding across 4 GPUs with checkpointing and paraphrase loading
#SBATCH --job-name=bib_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/bib_eval_%A_%a.out
#SBATCH --error=logs/bib_eval_%A_%a.err
#SBATCH --signal=INT@300

MODEL=${1:-"Qwen/Qwen3-8B"}
SPLIT=${2:-"train"}

source ~/.bashrc
conda activate cot
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs checkpoints/bib results/bib

python -c "
import sys, os
sys.path.insert(0, 'src')
from bib_data import prepare_split
from bib_evaluate import BibEvaluator, run_evaluation

gpu_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
total_gpus = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

model_short = '${MODEL}'.split('/')[-1].lower().replace('-', '_')
split = '${SPLIT}'

paraphrase_path = f'results/bib/bib_paraphrases_{split}.json'
if not os.path.exists(paraphrase_path):
    print(f'WARNING: paraphrase file not found at {paraphrase_path}, skipping paraphrase evaluation')
    paraphrase_path = None

df = prepare_split(split, seed=42, paraphrase_path=paraphrase_path)
samples = df.to_dict('records')
shard = samples[gpu_id::total_gpus]

print(f'GPU {gpu_id}/{total_gpus}: evaluating {len(shard)}/{len(samples)} samples')

evaluator = BibEvaluator('${MODEL}')
run_evaluation(
    evaluator, shard,
    output_path=f'results/bib/bib_eval_{split}_{model_short}_gpu{gpu_id}_of_{total_gpus}.json',
    checkpoint_path=f'checkpoints/bib/bib_eval_{split}_{model_short}_gpu{gpu_id}_of_{total_gpus}.pkl',
    checkpoint_freq=20,
)
"
