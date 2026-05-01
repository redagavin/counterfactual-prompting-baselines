# ABOUTME: Generate calibrated baselines for MedPerturb with correct data flow
# ABOUTME: Uses dataset_id/context_id structure to link perturbations to originals

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import asyncio
import json

import pandas as pd
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm as atqdm

from token_edit_distance import levenshtein_distance
from calibrated_paraphrase import generate_calibrated_paraphrase_async


def compute_token_change_percent(original, perturbed, tokenizer=None, use_words=False):
    """
    Compute token change percentage between original and perturbed text.

    Args:
        original: Original text
        perturbed: Perturbed text
        tokenizer: HuggingFace tokenizer (optional if use_words=True)
        use_words: If True, use word-level comparison (for testing)

    Returns:
        float: Percentage of tokens changed
    """
    if use_words:
        orig_tokens = original.split()
        pert_tokens = perturbed.split()
    else:
        orig_tokens = tokenizer.encode(original, add_special_tokens=False)
        pert_tokens = tokenizer.encode(perturbed, add_special_tokens=False)

    if len(orig_tokens) == 0:
        return 0.0 if len(pert_tokens) == 0 else 100.0

    edit_dist = levenshtein_distance(orig_tokens, pert_tokens)
    return (edit_dist / len(orig_tokens)) * 100


def find_original(df: pd.DataFrame, pert_row: pd.Series) -> pd.Series:
    """
    Find the original row corresponding to a perturbation row.

    Args:
        df: Full dataset DataFrame
        pert_row: A perturbation row (dataset_id in 2,3,4,5)

    Returns:
        pd.Series: The matching original row (dataset_id=1)

    Raises:
        KeyError: If no matching original found
    """
    matches = df[
        (df['dataset'] == pert_row['dataset']) &
        (df['context_id'] == pert_row['context_id']) &
        (df['dataset_id'] == 1)
    ]

    if len(matches) == 0:
        raise KeyError(
            f"No original found for dataset={pert_row['dataset']}, "
            f"context_id={pert_row['context_id']}"
        )

    return matches.iloc[0]


async def generate_baselines_for_perturbations(
    df: pd.DataFrame,
    tokenizer,
    output_path: str,
    openai_client=None,
    max_concurrent: int = 300,
    checkpoint_freq: int = 10,
    max_retries: int = 50,
    tolerance: float = 0.5,
    _paraphrase_fn=None
) -> list:
    """
    Generate calibrated baselines for all perturbation rows.

    For each perturbation (dataset_id 2-5):
    1. Find corresponding original (same dataset + context_id, dataset_id=1)
    2. Compute token change % between original and perturbation
    3. Generate calibrated paraphrase of original matching that %
    4. Store with metadata linking to source perturbation

    Args:
        df: Dataset with originals and perturbations
        tokenizer: HuggingFace tokenizer for token change calculation
        output_path: Path to save baselines JSON
        openai_client: AsyncOpenAI client (creates one if None)
        max_concurrent: Maximum concurrent API requests
        checkpoint_freq: Save checkpoint every N baselines
        max_retries: Max retries for calibrated paraphrasing
        tolerance: Acceptable deviation from target %
        _paraphrase_fn: Optional mock for testing

    Returns:
        list: Baseline dictionaries
    """
    if openai_client is None and _paraphrase_fn is None:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI()

    # Get perturbation rows only (dataset_id 2-5)
    perturbation_rows = df[df['dataset_id'].isin([2, 3, 4, 5])]

    # Load existing baselines for resume (preserves previous work on restart)
    baselines = []
    completed_sources = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            baselines = json.load(f)
            completed_sources = {int(b['source_index']) for b in baselines}
            print(f"  Resuming: {len(baselines)} baselines already generated")
    baselines_lock = asyncio.Lock()
    failed_items = []
    failed_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    completed_lock = asyncio.Lock()

    def atomic_save(data, path):
        """Write JSON atomically to prevent corruption on crash."""
        import tempfile
        import shutil
        dir_path = os.path.dirname(path) or '.'
        with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
            json.dump(data, f, indent=2)
            temp_path = f.name
        shutil.move(temp_path, path)

    async def process_perturbation(pert_row):
        nonlocal completed_count

        # Skip if already completed (resume logic)
        if int(pert_row['Index']) in completed_sources:
            return

        try:
            # Find original
            original = find_original(df, pert_row)

            # Compute token change %
            target_pct = compute_token_change_percent(
                original['clinical_context'],
                pert_row['clinical_context'],
                tokenizer
            )

            # Generate calibrated paraphrase
            async with semaphore:
                if _paraphrase_fn is not None:
                    result = await _paraphrase_fn(
                        original['clinical_context'],
                        target_pct,
                        tokenizer,
                        openai_client,
                        max_retries,
                        tolerance
                    )
                else:
                    result = await generate_calibrated_paraphrase_async(
                        question=original['clinical_context'],
                        target_pct=target_pct,
                        tokenizer=tokenizer,
                        openai_client=openai_client,
                        max_retries=max_retries,
                        tolerance=tolerance
                    )

            # Build baseline entry
            baseline = {
                'source_index': int(pert_row['Index']),
                'original_index': int(original['Index']),
                'dataset': pert_row['dataset'],
                'context_id': pert_row['context_id'],
                'perturbation_type': int(pert_row['dataset_id']),
                'baseline_dataset_id': int(pert_row['dataset_id']) + 4,
                'paraphrase': result['paraphrase'],
                'target_pct': target_pct,
                'actual_pct': result['actual_pct'],
                'deviation': result['deviation'],
                'retries_used': result['retries_used']
            }

            async with baselines_lock:
                baselines.append(baseline)

        except Exception as e:
            # Track failed items instead of crashing
            async with failed_lock:
                failed_items.append({
                    'index': int(pert_row['Index']),
                    'error': str(e)
                })
            print(f"ERROR: Failed to process perturbation {pert_row['Index']}: {e}")
            return

        # Checkpoint (only on success)
        async with completed_lock:
            completed_count += 1
            if completed_count % checkpoint_freq == 0:
                async with baselines_lock:
                    atomic_save(baselines, output_path)

    # Create tasks for all perturbations
    tasks = [process_perturbation(row) for _, row in perturbation_rows.iterrows()]

    # Run with progress bar
    await atqdm.gather(*tasks, desc="Generating baselines")

    # Final save (atomic)
    atomic_save(baselines, output_path)

    # Report failures
    if failed_items:
        print(f"\nWARNING: {len(failed_items)} items failed to process:")
        for item in failed_items[:10]:  # Show first 10
            print(f"  Index {item['index']}: {item['error']}")
        if len(failed_items) > 10:
            print(f"  ... and {len(failed_items) - 10} more")

    return baselines, failed_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate calibrated baselines for MedPerturb perturbations"
    )
    parser.add_argument('--dataset', type=str, default='data.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='results/baselines_v2.json',
                        help='Output JSON path')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer')
    parser.add_argument('--max_concurrent', type=int, default=300,
                        help='Maximum concurrent API requests')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N baselines')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit perturbations to process (for testing)')

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"  Total rows: {len(df)}")
    print(f"  Originals (dataset_id=1): {len(df[df['dataset_id'] == 1])}")
    print(f"  Perturbations (dataset_id 2-5): {len(df[df['dataset_id'].isin([2,3,4,5])])}")

    # Optionally limit for testing
    if args.sample_size:
        # Keep all originals, limit perturbations
        originals = df[df['dataset_id'] == 1]
        perturbations = df[df['dataset_id'].isin([2, 3, 4, 5])].head(args.sample_size)
        df = pd.concat([originals, perturbations])
        print(f"  Limited to {len(perturbations)} perturbations (test mode)")

    print(f"\nGenerating baselines with {args.max_concurrent} concurrent requests...")

    baselines, failed = asyncio.run(generate_baselines_for_perturbations(
        df=df,
        tokenizer=tokenizer,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        checkpoint_freq=args.checkpoint_freq
    ))

    print(f"\nDone! Generated {len(baselines)} baselines")
    if failed:
        print(f"WARNING: {len(failed)} items failed - check logs above")
    print(f"Saved to: {args.output}")
