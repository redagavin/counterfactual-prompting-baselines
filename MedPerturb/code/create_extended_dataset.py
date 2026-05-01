# ABOUTME: Create extended dataset by appending baseline rows
# ABOUTME: Baseline rows have dataset_id 6-9 and empty model output columns

import os
import pandas as pd
import json
import argparse


def create_extended_dataset(
    dataset_path: str,
    baselines_path: str,
    output_path: str,
    force: bool = False
) -> pd.DataFrame:
    """
    Create extended dataset by appending baseline rows.

    Args:
        dataset_path: Path to original data.csv
        baselines_path: Path to baselines.json
        output_path: Path to save extended dataset
        force: If True, overwrite even if output has model results

    Returns:
        pd.DataFrame: Extended dataset with baseline rows

    Raises:
        ValueError: If output_path exists with model evaluation results and force=False
    """
    # Check for existing output with model results (prevent accidental data loss)
    if os.path.exists(output_path) and not force:
        existing_df = pd.read_csv(output_path)
        model_cols = [c for c in existing_df.columns if c.startswith('LLAMA')]
        if model_cols:
            has_results = existing_df[model_cols].notna().any().any()
            if has_results:
                raise ValueError(
                    f"{output_path} already contains model evaluation results. "
                    "Re-running would destroy this data. Use force=True to overwrite."
                )

    # Load original dataset
    df = pd.read_csv(dataset_path)

    # Load baselines
    with open(baselines_path, 'r') as f:
        baselines = json.load(f)

    # Get column list from original
    columns = df.columns.tolist()

    # Find the starting index for baselines
    start_index = 800  # Per design spec

    # Create baseline rows
    baseline_rows = []
    for i, b in enumerate(baselines):
        # Find the original row to copy metadata from
        original_row = df[df['Index'] == b['original_index']].iloc[0]

        row = {
            'Index': start_index + i,
            'dataset': b['dataset'],
            'dataset_id': b['baseline_dataset_id'],
            'context_id': b['context_id'],
            'original_patient_gender': original_row['original_patient_gender'],
            'clinical_context': b['paraphrase'],
        }

        # All other columns are empty (model outputs, clinician ratings, etc.)
        for col in columns:
            if col not in row:
                row[col] = None

        baseline_rows.append(row)

    # Create DataFrame from baseline rows
    baselines_df = pd.DataFrame(baseline_rows, columns=columns)

    # Concatenate
    extended_df = pd.concat([df, baselines_df], ignore_index=True)

    # Save
    extended_df.to_csv(output_path, index=False)

    return extended_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create extended dataset with baseline rows"
    )
    parser.add_argument('--dataset', type=str, default='data.csv',
                        help='Path to original dataset')
    parser.add_argument('--baselines', type=str, default='results/baselines_v2.json',
                        help='Path to baselines JSON')
    parser.add_argument('--output', type=str, default='data_with_baselines.csv',
                        help='Output path for extended dataset')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite output even if it has model results')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    print(f"Loading baselines: {args.baselines}")

    result_df = create_extended_dataset(
        args.dataset,
        args.baselines,
        args.output,
        force=args.force
    )

    print(f"\nExtended dataset created:")
    print(f"  Original rows: 800")
    print(f"  Baseline rows: {len(result_df) - 800}")
    print(f"  Total rows: {len(result_df)}")
    print(f"Saved to: {args.output}")
