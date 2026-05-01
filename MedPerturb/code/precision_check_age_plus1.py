# ABOUTME: Age +1 perturbation for precision sanity check
# ABOUTME: Increments patient age by 1 to test semantic vs token-level sensitivity

import os
import json
import argparse

from precision_check_age_swap import apply_age_perturbation


def run_age_plus1(dataset_path, tokenizer=None):
    """Apply +1 age perturbation to all original non-conversational samples.

    Args:
        dataset_path: Path to data_with_baselines.csv
        tokenizer: Optional HuggingFace tokenizer for token edit distance

    Returns:
        list of dicts with keys: context_id, dataset, original_text,
            age_swapped_text, original_age, new_age, token_change_pct,
            age_extraction_failed
    """
    return apply_age_perturbation(
        dataset_path, lambda age, context_id: age + 1, tokenizer)


def main():
    parser = argparse.ArgumentParser(
        description="Age +1 perturbation for precision sanity check")
    parser.add_argument('--dataset', type=str, default='data_with_baselines.csv',
                        help='Path to data_with_baselines.csv')
    parser.add_argument('--output', type=str,
                        default='results/precision_plus1_age_plus1.json',
                        help='Output JSON path')
    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer (token edit distance)')
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = run_age_plus1(args.dataset, tokenizer=tokenizer)

    # Print stats
    total = len(results)
    failed = sum(1 for r in results if r['age_extraction_failed'])
    successful = total - failed
    print(f"Total samples: {total}")
    print(f"Successful age +1: {successful}")
    print(f"Failed age extraction: {failed}")

    if successful > 0:
        pcts = [r['token_change_pct'] for r in results
                if r['token_change_pct'] is not None]
        if pcts:
            print(f"Token change %: mean={sum(pcts)/len(pcts):.2f}, "
                  f"min={min(pcts):.2f}, max={max(pcts):.2f}")

    # Save JSON (exclude original_text to save space — can re-join by context_id)
    output_records = []
    for r in results:
        record = {k: v for k, v in r.items() if k != 'original_text'}
        output_records.append(record)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_records, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
