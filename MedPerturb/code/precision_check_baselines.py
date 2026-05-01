# ABOUTME: Generates calibrated paraphrases targeting each sample's age swap token change %
# ABOUTME: Sequential baseline generation for precision sanity check (~100 samples)

import argparse
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))


def _atomic_save(data, path):
    """Write JSON atomically via tempfile + move to prevent corruption on crash."""
    dir_path = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    shutil.move(temp_path, path)


def generate_precision_baselines(age_swap_path, originals, output_path,
                                 tokenizer=None, openai_client=None,
                                 max_retries=50, tolerance=0.5,
                                 checkpoint_freq=5, _paraphrase_fn=None):
    """Generate calibrated paraphrases targeting each sample's age swap token change %.

    Args:
        age_swap_path: Path to age perturbation JSON (token_change_pct field per sample is the calibration target)
        originals: Dict mapping context_id to original clinical text
        output_path: Path to save baselines JSON
        tokenizer: HuggingFace tokenizer (required unless using _paraphrase_fn)
        openai_client: OpenAI client (required unless using _paraphrase_fn)
        max_retries: Max retries for calibrated paraphrasing
        tolerance: Acceptable deviation from target in percentage points
        checkpoint_freq: Save checkpoint every N samples
        _paraphrase_fn: Optional mock replacing generate_calibrated_paraphrase

    Returns:
        list: Baseline dicts with context_id, dataset, paraphrase, target_pct,
              actual_pct, deviation, retries_used
    """
    if _paraphrase_fn is None:
        from calibrated_paraphrase import generate_calibrated_paraphrase
        paraphrase_fn = generate_calibrated_paraphrase
    else:
        paraphrase_fn = _paraphrase_fn

    with open(age_swap_path, 'r') as f:
        age_swap_data = json.load(f)

    # Resume from existing output
    baselines = []
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            baselines = json.load(f)
            completed_ids = {b['context_id'] for b in baselines}
            print(f"  Resuming: {len(baselines)} baselines already generated")

    # Filter: skip failed extractions and ~0% token change
    eligible = [
        s for s in age_swap_data
        if not s.get('age_extraction_failed', False)
        and s.get('token_change_pct') is not None
        and s['token_change_pct'] > 0
    ]

    newly_completed = 0
    for sample in eligible:
        context_id = sample['context_id']

        if context_id in completed_ids:
            continue

        original_text = originals[context_id]
        target_pct = sample['token_change_pct']

        result = paraphrase_fn(
            original_text,
            target_pct,
            tokenizer,
            openai_client,
            max_retries,
            tolerance,
        )

        baseline = {
            'context_id': context_id,
            'dataset': sample['dataset'],
            'paraphrase': result['paraphrase'],
            'target_pct': target_pct,
            'actual_pct': result['actual_pct'],
            'deviation': result['deviation'],
            'retries_used': result['retries_used'],
        }
        baselines.append(baseline)
        newly_completed += 1

        if newly_completed % checkpoint_freq == 0:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            _atomic_save(baselines, output_path)

    # Final save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    _atomic_save(baselines, output_path)

    return baselines


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibrated paraphrase baselines for precision sanity check"
    )
    parser.add_argument('--age_swap', type=str,
                        default='results/precision_plus1_age_plus1.json',
                        help='Path to age perturbation JSON (token_change_pct field is used as calibration target)')
    parser.add_argument('--dataset', type=str,
                        default='data_with_baselines.csv',
                        help='Path to dataset CSV for loading original texts')
    parser.add_argument('--output', type=str,
                        default='results/precision_check_baselines.json',
                        help='Output JSON path')
    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer')
    parser.add_argument('--max_retries', type=int, default=50,
                        help='Max retries per sample')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='Acceptable deviation from target %%')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Save checkpoint every N samples')
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from openai import OpenAI

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    # Build originals dict from dataset_id=1, non-conversational
    orig_df = df[(df['dataset_id'] == 1) & (df['dataset'] != 'conversational')]
    originals = dict(zip(orig_df['context_id'], orig_df['clinical_context']))
    print(f"  Loaded {len(originals)} original texts")

    openai_client = OpenAI()

    print(f"Loading age swap data: {args.age_swap}")
    print(f"Generating baselines...")

    baselines = generate_precision_baselines(
        age_swap_path=args.age_swap,
        originals=originals,
        output_path=args.output,
        tokenizer=tokenizer,
        openai_client=openai_client,
        max_retries=args.max_retries,
        tolerance=args.tolerance,
        checkpoint_freq=args.checkpoint_freq,
    )

    print(f"\nDone! Generated {len(baselines)} baselines")
    if baselines:
        deviations = [b['deviation'] for b in baselines]
        print(f"Deviation: mean={sum(deviations)/len(deviations):.2f}, "
              f"max={max(deviations):.2f}")
        within_tol = sum(1 for d in deviations if d <= args.tolerance)
        print(f"Within tolerance: {within_tol}/{len(baselines)} "
              f"({100*within_tol/len(baselines):.1f}%)")
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
