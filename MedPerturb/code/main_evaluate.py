# ABOUTME: Main experiment evaluation for all text variants per sample
# ABOUTME: Evaluates originals, perturbations, calibrated baselines, and neutral baseline

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

from eval_utils import (
    model_short_name, detect_slurm, shard_samples,
    save_checkpoint, load_checkpoint, mark_complete,
    result_path, checkpoint_path,
)
from evaluate_models import ModelEvaluator


NEUTRAL_SENTENCE = "A family member came with me to the appointment."

# Mapping from dataset_id to variant name
VARIANT_NAMES = {
    1: "original",
    2: "gender_swap",
    3: "gender_remove",
    4: "uncertain",
    5: "colorful",
    6: "gender_swap_baseline",
    7: "gender_remove_baseline",
    8: "uncertain_baseline",
    9: "colorful_baseline",
}


def load_main_experiment_data(dataset_path):
    """Load and group data by context_id with all text variants.

    Returns list of sample dicts, each containing:
    - context_id: int
    - original_text, gender_swap_text, ..., colorful_baseline_text: str
    """
    df = pd.read_csv(dataset_path)
    df = df[df["dataset"] != "conversational"]

    samples = []
    for context_id, group in df.groupby("context_id"):
        id_to_text = {}
        for _, row in group.iterrows():
            did = int(row["dataset_id"])
            if did in VARIANT_NAMES:
                id_to_text[did] = row["clinical_context"]

        # Require all 9 dataset_ids
        if len(id_to_text) < 9:
            continue

        sample = {"context_id": context_id}
        for did, name in VARIANT_NAMES.items():
            sample[f"{name}_text"] = id_to_text[did]
        samples.append(sample)

    return sorted(samples, key=lambda s: s["context_id"])


def evaluate_main_experiment_sample(evaluator, sample):
    """Evaluate all text variants for one sample.

    Returns dict with context_id and result keys:
    - {variant}_{QUESTION} for originals and perturbations
    - {perturbation}_baseline_{QUESTION} for calibrated baselines
    - neutral_{QUESTION} for neutral baseline
    """
    result = {"context_id": sample["context_id"]}

    # All named variants from CSV
    for did, variant_name in VARIANT_NAMES.items():
        text = sample[f"{variant_name}_text"]
        triage_results = evaluator.evaluate_triage(text)
        for question_type, qresult in triage_results.items():
            result[f"{variant_name}_{question_type}"] = qresult

    # Neutral baseline (prepend to original)
    neutral_text = NEUTRAL_SENTENCE + " " + sample["original_text"]
    neutral_results = evaluator.evaluate_triage(neutral_text)
    for question_type, qresult in neutral_results.items():
        result[f"neutral_{question_type}"] = qresult

    return result


def main():
    parser = argparse.ArgumentParser(description="Main experiment evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data_with_baselines.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/main")
    parser.add_argument("--checkpoint_freq", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()

    ms = model_short_name(args.model)
    slurm_gpu, slurm_total = detect_slurm()
    if slurm_gpu is not None:
        args.gpu_id = slurm_gpu
        args.total_gpus = slurm_total
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0

    ckpt_path = checkpoint_path(args.checkpoint_dir, "main", ms, args.gpu_id, args.total_gpus)
    res_path = result_path(args.output_dir, "main", ms, args.gpu_id, args.total_gpus)

    # Load data
    samples = load_main_experiment_data(args.dataset)
    samples = shard_samples(samples, args.gpu_id, args.total_gpus)
    if args.sample_size:
        samples = samples[:args.sample_size]
    print(f"GPU {args.gpu_id}: {len(samples)} samples to evaluate")

    # Load checkpoint
    results, completed = load_checkpoint(ckpt_path)

    # Initialize model
    evaluator = ModelEvaluator(args.model)

    # Evaluate
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    new_count = 0
    for sample in tqdm(samples, desc=f"GPU {args.gpu_id}"):
        if sample["context_id"] in completed:
            continue
        result = evaluate_main_experiment_sample(evaluator, sample)
        results.append(result)
        completed.add(sample["context_id"])
        new_count += 1
        if new_count % args.checkpoint_freq == 0:
            save_checkpoint(ckpt_path, results, completed)

    save_checkpoint(ckpt_path, results, completed)

    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    mark_complete(args.checkpoint_dir, "main", ms, args.gpu_id, args.total_gpus)
    print(f"GPU {args.gpu_id}: Complete. {len(results)} results saved to {res_path}")


if __name__ == "__main__":
    main()
