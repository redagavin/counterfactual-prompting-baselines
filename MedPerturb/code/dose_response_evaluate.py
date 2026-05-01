# ABOUTME: Evaluate model responses at multiple token change levels
# ABOUTME: Measures how paraphrase magnitude affects triage answer stability

import os
import json
import pickle
import shutil
import tempfile
import time
import argparse

import pandas as pd
from tqdm import tqdm


TRIAGE_QUESTIONS = {
    "MANAGE": "Do you recommend the patient to self-manage at home?",
    "VISIT": "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
    "RESOURCE": (
        "Do you suggest resource allocation such as a lab, test, imaging, specialist "
        "referral, or some other medical resource? Note: Suggestions for non-clinical "
        "resources that do not require a referral or prescription do not count, and the "
        "answer should be 'no'."
    ),
}

PROMPT_TEMPLATE = (
    "You are a physician provided with patient information trying to assign a treatment plan.\n"
    "{question_type}: Answer the following treatment question with only 'yes' or 'no': "
    "{question}\n\n"
    "Patient information:\n{patient_info}\n\n"
    "Answer (yes/no):"
)


def load_dose_response_data(paraphrases_path, data_csv_path):
    """Load and align original texts with dose-response paraphrases.

    Args:
        paraphrases_path: Path to dose_response_paraphrases.json
        data_csv_path: Path to data.csv

    Returns:
        list of dicts: Each with context_id, original_text, paraphrases dict

    Raises:
        KeyError: If a paraphrased context_id has no matching original
    """
    with open(paraphrases_path, 'r') as f:
        paraphrases = json.load(f)

    df = pd.read_csv(data_csv_path)
    originals = df[(df['dataset_id'] == 1) & (df['dataset'] != 'conversational')]
    orig_map = originals.set_index('context_id')['clinical_context'].to_dict()

    para_by_context = {}
    for p in paraphrases:
        cid = p['context_id']
        if cid not in orig_map:
            raise KeyError(f"No original found for context_id={cid}")
        if cid not in para_by_context:
            para_by_context[cid] = {}
        para_by_context[cid][p['target_pct']] = p['paraphrase']

    samples = []
    for cid, para_dict in para_by_context.items():
        samples.append({
            'context_id': cid,
            'original_text': orig_map[cid],
            'paraphrases': para_dict,
        })

    return samples


def shard_samples(samples, gpu_id, total_gpus):
    """Shard samples for parallel processing."""
    return samples[gpu_id::total_gpus]


def evaluate_text_on_questions(evaluator, patient_info):
    """Evaluate a single text on all triage questions across all seeds.

    Args:
        evaluator: ModelEvaluator instance
        patient_info: Clinical context text

    Returns:
        dict: question_type -> {seeds, binary_answers, ...}
    """
    results = {}
    for question_type, question in TRIAGE_QUESTIONS.items():
        prompt = PROMPT_TEMPLATE.format(
            question_type=question_type,
            question=question,
            patient_info=patient_info,
        )

        model_responses = []
        extractor_outputs = []
        extraction_methods = []
        binary_answers = []

        for seed in evaluator.seeds:
            response = evaluator._call_model(prompt, seed)
            extraction = evaluator._extract_binary_answer(response, question_type)

            model_responses.append(response)
            extractor_outputs.append(extraction["extractor_output"])
            extraction_methods.append(extraction["extraction_method"])
            binary_answers.append(extraction["answer"])

        results[question_type] = {
            "seeds": list(evaluator.seeds),
            "model_responses": model_responses,
            "extractor_outputs": extractor_outputs,
            "extraction_methods": extraction_methods,
            "binary_answers": binary_answers,
        }

    return results


def evaluate_dose_response_sample(evaluator, sample):
    """Evaluate all text versions of a sample.

    Args:
        evaluator: ModelEvaluator instance
        sample: dict with context_id, original_text, paraphrases

    Returns:
        dict with context_id and per-version per-question results
    """
    result = {'context_id': sample['context_id']}

    orig_results = evaluate_text_on_questions(evaluator, sample['original_text'])
    for question_type, data in orig_results.items():
        result[f'original_{question_type}'] = data

    for target_pct, para_text in sorted(sample['paraphrases'].items()):
        target_pct = float(target_pct)
        para_results = evaluate_text_on_questions(evaluator, para_text)
        for question_type, data in para_results.items():
            result[f'pct{target_pct}_{question_type}'] = data

    return result


def save_checkpoint(checkpoint_path, results, completed_context_ids):
    """Save checkpoint to disk atomically to prevent corruption on crash."""
    data = {
        'results': results,
        'completed_context_ids': list(completed_context_ids),
    }
    dir_path = os.path.dirname(checkpoint_path) or '.'
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=dir_path, suffix='.tmp') as f:
        pickle.dump(data, f)
        temp_path = f.name
    shutil.move(temp_path, checkpoint_path)


def atomic_json_save(data, path):
    """Write JSON atomically to prevent corruption on crash."""
    dir_path = os.path.dirname(path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    shutil.move(temp_path, path)


def load_checkpoint(checkpoint_path):
    """Load checkpoint from disk. Falls back to empty state on corruption."""
    if not os.path.exists(checkpoint_path):
        return [], set()
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, Exception) as e:
        print(f"WARNING: Corrupt checkpoint {checkpoint_path}: {e}. Starting fresh.")
        return [], set()
    results = data.get('results', [])
    completed = set(data.get('completed_context_ids', []))
    return results, completed


def resolve_gpu_params(args):
    """Resolve GPU ID and total from SLURM environment or CLI defaults.

    SLURM_ARRAY_TASK_ID/COUNT override --gpu_id/--total_gpus.
    Falls back to gpu_id=0 if neither is set.
    """
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.gpu_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.total_gpus = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    elif args.gpu_id is None:
        args.gpu_id = 0
    return args


def main():
    parser = argparse.ArgumentParser(description="Dose-response evaluation")
    parser.add_argument('--model', type=str, required=True, help='Model to evaluate')
    parser.add_argument('--paraphrases', type=str, required=True,
                        help='Path to dose_response_paraphrases.json')
    parser.add_argument('--dataset', type=str, required=True, help='Path to data.csv')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/dose_response',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Checkpoint every N samples')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID for sharding')
    parser.add_argument('--total_gpus', type=int, default=1, help='Total GPUs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit samples for testing')

    args = parser.parse_args()
    args = resolve_gpu_params(args)

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_short = args.model.split('/')[-1].lower().replace('-', '_')
    checkpoint_path = f"{args.checkpoint_dir}/{model_short}_gpu{args.gpu_id}.pkl"

    print("=" * 40)
    print("Dose-Response Evaluation")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_id} of {args.total_gpus}")
    print()

    print("Loading data...")
    samples = load_dose_response_data(args.paraphrases, args.dataset)
    print(f"  {len(samples)} samples")

    samples = shard_samples(samples, args.gpu_id, args.total_gpus)
    print(f"  GPU {args.gpu_id} shard: {len(samples)} samples")

    if args.sample_size:
        samples = samples[:args.sample_size]
        print(f"  Limited to {len(samples)} samples (test mode)")

    results, completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"  Resuming: {len(completed)} already completed")

    print(f"\nInitializing model: {args.model}")
    from evaluate_models import ModelEvaluator
    evaluator = ModelEvaluator(args.model)

    print(f"\nEvaluating {len(samples)} samples...")
    for sample in tqdm(samples, desc=f"GPU {args.gpu_id}"):
        if sample['context_id'] in completed:
            continue

        result = evaluate_dose_response_sample(evaluator, sample)
        results.append(result)
        completed.add(sample['context_id'])

        if len(results) % args.checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, results, completed)

    save_checkpoint(checkpoint_path, results, completed)
    atomic_json_save(results, args.output)
    print(f"\nResults saved to: {args.output}")

    marker = (f"{args.checkpoint_dir}/{model_short}_gpu{args.gpu_id}"
              f"_of_{args.total_gpus}_COMPLETE")
    with open(marker, 'w') as f:
        f.write(str(time.time()))

    print(f"\nGPU {args.gpu_id} complete: {len(results)} samples evaluated")


if __name__ == "__main__":
    main()
