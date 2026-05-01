# ABOUTME: Generate calibrated paraphrases at multiple token change levels
# ABOUTME: Produces dose-response data showing how token change % affects answer stability

import asyncio
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def load_samples(df):
    """Load non-conversational original samples.

    Args:
        df: DataFrame with data.csv structure

    Returns:
        list of dicts with context_id, dataset, clinical_context
    """
    originals = df[(df['dataset_id'] == 1) & (df['dataset'] != 'conversational')]
    samples = []
    for _, row in originals.iterrows():
        samples.append({
            'context_id': row['context_id'],
            'dataset': row['dataset'],
            'clinical_context': row['clinical_context'],
        })
    return samples


def atomic_save(data, path):
    """Write JSON atomically to prevent corruption on crash."""
    dir_path = os.path.dirname(path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    shutil.move(temp_path, path)


async def generate_paraphrases(
    samples,
    targets,
    tokenizer,
    openai_client,
    output_path=None,
    max_concurrent=300,
    checkpoint_freq=10,
    max_retries=50,
    tolerance=0.5,
    _paraphrase_fn=None,
):
    """Generate calibrated paraphrases at multiple token change levels.

    Args:
        samples: List of dicts with context_id, dataset, clinical_context
        targets: List of target token change percentages
        tokenizer: HuggingFace tokenizer
        openai_client: AsyncOpenAI client
        output_path: Path for JSON output (enables checkpointing/resume)
        max_concurrent: Maximum concurrent API requests
        checkpoint_freq: Save checkpoint every N paraphrases
        max_retries: Max retries per paraphrase
        tolerance: Acceptable deviation from target %
        _paraphrase_fn: Injectable paraphrase function for testing

    Returns:
        list: Paraphrase result dicts
    """
    if _paraphrase_fn is None:
        from calibrated_paraphrase import generate_calibrated_paraphrase_async
        _paraphrase_fn = generate_calibrated_paraphrase_async

    # Resume from checkpoint
    results = []
    completed_keys = set()
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            completed_keys = {(r['context_id'], r['target_pct']) for r in results}
            print(f"  Resuming: {len(results)} paraphrases already generated")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Corrupt checkpoint {output_path}: {e}. Starting fresh.")
            results = []
            completed_keys = set()

    results_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    completed_lock = asyncio.Lock()

    async def process_one(sample, target_pct):
        nonlocal completed_count

        key = (sample['context_id'], target_pct)
        if key in completed_keys:
            return

        async with semaphore:
            result = await _paraphrase_fn(
                sample['clinical_context'],
                target_pct,
                tokenizer,
                openai_client,
                max_retries,
                tolerance,
            )

        entry = {
            'context_id': sample['context_id'],
            'dataset': sample['dataset'],
            'target_pct': target_pct,
            'actual_pct': result['actual_pct'],
            'deviation': result['deviation'],
            'retries_used': result['retries_used'],
            'paraphrase': result['paraphrase'],
        }

        async with results_lock:
            results.append(entry)

        async with completed_lock:
            completed_count += 1
            if output_path and completed_count % checkpoint_freq == 0:
                async with results_lock:
                    atomic_save(results, output_path)

    tasks = []
    for sample in samples:
        for target in targets:
            tasks.append(process_one(sample, target))

    await asyncio.gather(*tasks)

    if output_path:
        atomic_save(results, output_path)

    return results


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from tqdm.asyncio import tqdm as atqdm

    parser = argparse.ArgumentParser(
        description="Generate dose-response paraphrases at multiple token change levels"
    )
    parser.add_argument('--dataset', type=str, default='data.csv',
                        help='Path to data.csv')
    parser.add_argument('--output', type=str,
                        default='results/dose_response_paraphrases.json',
                        help='Output JSON path')
    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer')
    parser.add_argument('--targets', type=str, default='5,10,20,40,60',
                        help='Comma-separated target token change percentages')
    parser.add_argument('--max_concurrent', type=int, default=300,
                        help='Maximum concurrent API requests')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N paraphrases')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit samples for testing')

    args = parser.parse_args()

    targets = [float(t) for t in args.targets.split(',')]

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    samples = load_samples(df)
    print(f"  {len(samples)} non-conversational original samples")

    if args.sample_size:
        samples = samples[:args.sample_size]
        print(f"  Limited to {len(samples)} samples (test mode)")

    print(f"Targets: {targets}")
    print(f"Total paraphrases: {len(samples) * len(targets)}")

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    results = asyncio.run(generate_paraphrases(
        samples=samples,
        targets=targets,
        tokenizer=tokenizer,
        openai_client=client,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        checkpoint_freq=args.checkpoint_freq,
    ))

    print(f"\nDone! Generated {len(results)} paraphrases")
    print(f"Saved to: {args.output}")
