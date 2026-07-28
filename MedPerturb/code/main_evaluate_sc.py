# ABOUTME: Self-consistency evaluation orchestrator with batched inference
# ABOUTME: Runs N i.i.d. samples per (case, condition, task) at configurable temperature

import argparse
import json
import os
import sys
import tempfile
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_models import ModelEvaluator
from main_evaluate import VARIANT_NAMES, load_main_experiment_data
from eval_utils import detect_slurm, shard_samples


# ---- Prompt template ----
# DUPLICATED from evaluate_models.py:276-295. If the existing template
# changes, update both. (We don't refactor the existing function to keep
# this work purely additive.)

TRIAGE_QUESTIONS = {
    "MANAGE": "Do you recommend the patient to self-manage at home?",
    "VISIT": "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
    "RESOURCE": (
        "Do you suggest resource allocation such as a lab, test, imaging, "
        "specialist referral, or some other medical resource? Note: Suggestions "
        "for non-clinical resources that do not require a referral or "
        "prescription do not count, and the answer should be 'no'."
    ),
}


def build_triage_prompt(patient_info: str, question_type: str) -> str:
    """Construct the user-content prompt for one (patient_info, task) pair.

    DUPLICATED from evaluate_models.py:290-295. Keep in sync if the existing
    template changes."""
    question = TRIAGE_QUESTIONS[question_type]
    return (
        "You are a physician provided with patient information trying to assign a treatment plan.\n"
        f"{question_type}: Answer the following treatment question with only 'yes' or 'no': {question}\n\n"
        f"Patient information:\n{patient_info}\n\n"
        "Answer (yes/no):"
    )


SC_CONDITIONS = ['original', 'gender_swap', 'gender_remove', 'uncertain', 'colorful']
SC_TASKS = ['MANAGE', 'VISIT', 'RESOURCE']


def setup_batched_tokenizer(tokenizer):
    """Configure tokenizer for batched left-padded inference. One-time."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def batched_sample(model, tokenizer, prompts, n_samples=10, max_new_tokens=512,
                    temperature=0.7):
    """Batched generation: multiple prompts x n_samples per generate() call.

    Args:
        prompts: list of raw user-content strings. Chat-template formatting
                 happens INSIDE this function.
        n_samples: number of i.i.d. samples per prompt (=10 for SC).
    Returns:
        list of len(prompts) lists, each containing n_samples decoded strings.
    """
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
    B = inputs.input_ids.shape[0]
    prompt_len = inputs.input_ids.shape[1]

    from transformers import GenerationConfig, LogitsProcessorList, LogitsProcessor
    import copy

    class _SafeLogitsProcessor(LogitsProcessor):
        """Sanitize logit values that FP8 quantization can corrupt.
        Without this, multinomial sampling crashes with 'probability tensor
        contains inf, nan or element < 0' on FP8-quantized 70B models.
        Tracks how many tokens are affected for scientific-validity auditing."""
        def __init__(self):
            self.n_nan_replaced = 0
            self.n_inf_replaced = 0
            self.n_clamped = 0
            self.n_total = 0

        def __call__(self, input_ids, scores):
            self.n_total += scores.numel()
            nan_mask = torch.isnan(scores)
            inf_mask = torch.isinf(scores)
            self.n_nan_replaced += nan_mask.sum().item()
            self.n_inf_replaced += inf_mask.sum().item()
            scores = torch.where(nan_mask, torch.full_like(scores, -100.0), scores)
            scores = torch.where(inf_mask & (scores > 0), torch.full_like(scores, 100.0), scores)
            scores = torch.where(inf_mask & (scores < 0), torch.full_like(scores, -100.0), scores)
            clamp_mask = (scores < -100) | (scores > 100)
            self.n_clamped += clamp_mask.sum().item()
            scores = scores.clamp(min=-100, max=100)
            return scores

    gen_config = copy.deepcopy(model.generation_config)
    gen_config.update(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if n_samples > 1:
        gen_config.update(num_return_sequences=n_samples)
    logits_proc = _SafeLogitsProcessor()
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=gen_config,
        logits_processor=LogitsProcessorList([logits_proc]),
    )
    if logits_proc.n_nan_replaced > 0 or logits_proc.n_inf_replaced > 0 or logits_proc.n_clamped > 0:
        print(f"  Logits sanitized: {logits_proc.n_nan_replaced} NaN→-100, "
              f"{logits_proc.n_inf_replaced} inf→±100, "
              f"{logits_proc.n_clamped} clamped "
              f"(of {logits_proc.n_total} total logits)", file=sys.stderr)

    generated = outputs[:, prompt_len:]
    flat = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [flat[b * n_samples:(b + 1) * n_samples] for b in range(B)]


def atomic_write_json(data, path):
    """tempfile + os.replace = POSIX-atomic write within same filesystem."""
    dir_ = os.path.dirname(path) or '.'
    os.makedirs(dir_, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp_path = f.name
    os.replace(tmp_path, path)


SC_EXPECTED_KEYS = frozenset(f'{p}_{t}' for p in SC_CONDITIONS for t in SC_TASKS)


def evaluate_case_sc(evaluator, case_row, n_samples, batch_size, max_new_tokens,
                     temperature=0.7):
    """Run SC evaluation for one case across all conditions and tasks.

    Returns dict with 15 keys (one per condition_task), each containing
    sample_index, model_responses, extractor_outputs, extraction_methods,
    binary_answers, and logit_probs.
    """
    prompts = []
    for pert in SC_CONDITIONS:
        patient_info = case_row[f'{pert}_text']
        for task in SC_TASKS:
            user_content = build_triage_prompt(patient_info, task)
            prompts.append((pert, task, user_content))

    results = {}
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        batch_contents = [p[2] for p in batch]

        # Sequential sampling: call generate with 1 sample at a time, loop n_samples
        # times. Required because num_return_sequences > 1 causes 75% NaN logits
        # under FP8 quantization (70B), producing garbage text.
        per_prompt_samples = [[] for _ in batch]
        for _ in range(n_samples):
            decoded = batched_sample(
                evaluator.model, evaluator.tokenizer,
                batch_contents, n_samples=1, max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for i, one_sample in enumerate(decoded):
                per_prompt_samples[i].append(one_sample[0])

        for (pert, task, user_content), samples in zip(batch, per_prompt_samples):
            extractions = [evaluator._extract_binary_answer(s, task) for s in samples]
            results[f"{pert}_{task}"] = {
                "sample_index": list(range(n_samples)),
                "model_responses": samples,
                "extractor_outputs": [e["extractor_output"] for e in extractions],
                "extraction_methods": [e["extraction_method"] for e in extractions],
                "binary_answers": [e["answer"] for e in extractions],
                "logit_probs": evaluator.extract_logit_probs(user_content),
            }
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-consistency main-experiment evaluation"
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='data_with_baselines.csv')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/main_experiment_sc')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=None)
    return parser.parse_args()


def _shard_paths(output_dir, checkpoint_dir, model_short, gpu_id, total_gpus,
                  temperature=0.7):
    """Build per-shard output/checkpoint paths."""
    temp_suffix = f"_t{temperature:.1f}" if temperature != 0.7 else ""
    tag = f"{model_short}{temp_suffix}_gpu{gpu_id}_of_{total_gpus}"
    return (
        os.path.join(output_dir, f"main_evaluation_sc_{tag}.json"),
        os.path.join(checkpoint_dir, f"partial_sc_{tag}.json"),
    )


def main():
    args = parse_args()

    # Detect SLURM array job
    slurm_gpu, slurm_total = detect_slurm()
    if slurm_gpu is not None:
        args.gpu_id = slurm_gpu
        args.total_gpus = slurm_total
        print(f"Detected SLURM array: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0

    model_short = (args.model.split('/')[-1].lower().replace('-', '_'))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    out_path, ckpt_path = _shard_paths(
        args.output_dir, args.checkpoint_dir,
        model_short, args.gpu_id, args.total_gpus,
        temperature=args.temperature,
    )

    torch.manual_seed(args.rng_seed + args.gpu_id)

    completed_cids = set()
    results = []
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                results = json.load(f)
            completed_cids = {r['context_id'] for r in results}
            print(f"Resuming: {len(completed_cids)} cases already done.")
        except (json.JSONDecodeError, KeyError) as e:
            corrupt = f"{ckpt_path}.corrupt.{int(time.time())}"
            os.rename(ckpt_path, corrupt)
            print(f"WARNING: corrupt checkpoint renamed to {corrupt}: {e}",
                  file=sys.stderr)
            results = []
            completed_cids = set()

    # Validate completeness of resumed cases
    valid_results = []
    for r in results:
        if SC_EXPECTED_KEYS.issubset(set(r.keys())):
            valid_results.append(r)
        else:
            print(f"WARNING: dropping incomplete case {r.get('context_id')}",
                  file=sys.stderr)
    results = valid_results
    completed_cids = {r['context_id'] for r in results}

    samples = load_main_experiment_data(args.dataset)
    samples = shard_samples(samples, args.gpu_id, args.total_gpus)
    if args.sample_size:
        samples = samples[:args.sample_size]
    print(f"GPU {args.gpu_id}: {len(samples)} samples to evaluate")

    evaluator = ModelEvaluator(args.model)
    setup_batched_tokenizer(evaluator.tokenizer)

    for sample in tqdm(samples, desc=f'SC eval GPU {args.gpu_id}'):
        cid = sample['context_id']
        if cid in completed_cids:
            continue
        try:
            case_result = evaluate_case_sc(
                evaluator, sample,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"ERROR on case {cid}: {e}", file=sys.stderr)
            case_result = {'error': str(e)}
        case_result['context_id'] = cid
        case_result['rng_seed'] = args.rng_seed
        case_result['temperature'] = args.temperature
        results.append(case_result)
        atomic_write_json(results, ckpt_path)

    atomic_write_json(results, out_path)
    n_errors = sum(1 for r in results if 'error' in r)
    print(
        f"Wrote {len(results)} cases to {out_path}"
        + (f" ({n_errors} with errors)" if n_errors else "")
    )


if __name__ == "__main__":
    main()
