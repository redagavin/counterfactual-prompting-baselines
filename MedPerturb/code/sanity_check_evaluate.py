# ABOUTME: Evaluation script for gender question sanity check
# ABOUTME: Validates MedPerturb MI pipeline by asking "Is this patient male?" on gender-swapped texts

import os
import json
import argparse

import pandas as pd
from tqdm import tqdm

from eval_utils import (
    model_short_name, detect_slurm, shard_samples,
    save_checkpoint, load_checkpoint, mark_complete,
    result_path, checkpoint_path,
)

GENDER_QUESTION = "Is this patient male?"

NEUTRAL_SENTENCE = "A family member came with me to the appointment."

GENDER_PROMPT_TEMPLATE = (
    "You are a physician provided with patient information.\n"
    "GENDER: Answer the following question with only 'yes' or 'no': {question}\n\n"
    "Patient information:\n{patient_info}\n\n"
    "Answer (yes/no):"
)


def build_gender_prompt(patient_info):
    """Build the gender question prompt for a patient.

    Args:
        patient_info: Clinical context text

    Returns:
        str: Complete prompt for model evaluation
    """
    return GENDER_PROMPT_TEMPLATE.format(
        question=GENDER_QUESTION,
        patient_info=patient_info
    )


def load_sanity_check_data(dataset_path):
    """Load and align original, gender-swap, and baseline texts.

    Reads data_with_baselines.csv, filters to non-conversational samples,
    and aligns the three text versions by context_id.

    Args:
        dataset_path: Path to data_with_baselines.csv

    Returns:
        list of dicts: Each with context_id, original_text, swap_text, baseline_text
    """
    df = pd.read_csv(dataset_path)

    # Exclude conversational subset (format perturbation, not gender swap)
    df = df[df['dataset'] != 'conversational']

    originals = df[df['dataset_id'] == 1].set_index('context_id')
    swaps = df[df['dataset_id'] == 2].set_index('context_id')
    baselines = df[df['dataset_id'] == 6].set_index('context_id')

    # Only include samples that have all three versions
    # Sorted for deterministic SLURM sharding across restarts
    common = sorted(originals.index.intersection(swaps.index).intersection(baselines.index))

    samples = []
    for cid in common:
        samples.append({
            'context_id': cid,
            'original_text': originals.loc[cid, 'clinical_context'],
            'swap_text': swaps.loc[cid, 'clinical_context'],
            'baseline_text': baselines.loc[cid, 'clinical_context'],
        })

    return samples


def evaluate_gender_question(evaluator, patient_info):
    """Evaluate the gender question for a single text across all seeds.

    Args:
        evaluator: ModelEvaluator instance (uses _call_model, _extract_binary_answer,
                   and extract_logit_probs)
        patient_info: Clinical context text

    Returns:
        dict: {seeds, model_responses, extractor_outputs, extraction_methods,
               binary_answers, logit_probs}
    """
    prompt = build_gender_prompt(patient_info)
    model_responses = []
    extractor_outputs = []
    extraction_methods = []
    binary_answers = []

    for seed in evaluator.seeds:
        response = evaluator._call_model(prompt, seed)
        extraction = evaluator._extract_binary_answer(response, "GENDER")

        model_responses.append(response)
        extractor_outputs.append(extraction["extractor_output"])
        extraction_methods.append(extraction["extraction_method"])
        binary_answers.append(extraction["answer"])

    logit_probs = evaluator.extract_logit_probs(prompt)

    return {
        "seeds": list(evaluator.seeds),
        "model_responses": model_responses,
        "extractor_outputs": extractor_outputs,
        "extraction_methods": extraction_methods,
        "binary_answers": binary_answers,
        "logit_probs": logit_probs,
    }


def evaluate_sanity_check_sample(evaluator, sample):
    """Evaluate gender question on all text versions of a sample.

    Args:
        evaluator: ModelEvaluator instance
        sample: dict with context_id, original_text, swap_text, baseline_text

    Returns:
        dict with context_id, original_GENDER, gender_swap_GENDER,
             gender_swap_baseline_GENDER, neutral_GENDER
    """
    neutral_text = NEUTRAL_SENTENCE + " " + sample['original_text']

    return {
        'context_id': sample['context_id'],
        'original_GENDER': evaluate_gender_question(evaluator, sample['original_text']),
        'gender_swap_GENDER': evaluate_gender_question(evaluator, sample['swap_text']),
        'gender_swap_baseline_GENDER': evaluate_gender_question(evaluator, sample['baseline_text']),
        'neutral_GENDER': evaluate_gender_question(evaluator, neutral_text),
    }


def main():
    parser = argparse.ArgumentParser(description="Sanity check: gender question evaluation")
    parser.add_argument('--model', type=str, required=True, help='Model to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to data_with_baselines.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/sanity_check',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Checkpoint every N samples')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID for sharding')
    parser.add_argument('--total_gpus', type=int, default=1, help='Total GPUs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit samples for testing')

    args = parser.parse_args()

    ms = model_short_name(args.model)
    slurm_gpu, slurm_total = detect_slurm()
    if slurm_gpu is not None:
        args.gpu_id = slurm_gpu
        args.total_gpus = slurm_total
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = checkpoint_path(args.checkpoint_dir, "sanity_check", ms, args.gpu_id, args.total_gpus)
    res_path = result_path(args.output_dir, "sanity_check", ms, args.gpu_id, args.total_gpus)

    print("=" * 40)
    print("Sanity Check: Gender Question Evaluation")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_id} of {args.total_gpus}")
    print()

    # Load data
    print("Loading data...")
    samples = load_sanity_check_data(args.dataset)
    print(f"  {len(samples)} non-conversational samples")

    # Shard
    samples = shard_samples(samples, args.gpu_id, args.total_gpus)
    print(f"  GPU {args.gpu_id} shard: {len(samples)} samples")

    if args.sample_size:
        samples = samples[:args.sample_size]
        print(f"  Limited to {len(samples)} samples (test mode)")

    # Load checkpoint
    results, completed = load_checkpoint(ckpt_path)
    if completed:
        print(f"  Resuming: {len(completed)} already completed")

    # Initialize model
    print(f"\nInitializing model: {args.model}")
    from evaluate_models import ModelEvaluator
    evaluator = ModelEvaluator(args.model)

    # Evaluate
    print(f"\nEvaluating {len(samples)} samples...")
    new_count = 0
    for sample in tqdm(samples, desc=f"GPU {args.gpu_id}"):
        if sample['context_id'] in completed:
            continue

        result = evaluate_sanity_check_sample(evaluator, sample)
        results.append(result)
        completed.add(sample['context_id'])
        new_count += 1

        if new_count % args.checkpoint_freq == 0:
            save_checkpoint(ckpt_path, results, completed)

    # Final save
    save_checkpoint(ckpt_path, results, completed)
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {res_path}")

    mark_complete(args.checkpoint_dir, "sanity_check", ms, args.gpu_id, args.total_gpus)

    print(f"\nGPU {args.gpu_id} complete: {len(results)} samples evaluated")


if __name__ == "__main__":
    main()
