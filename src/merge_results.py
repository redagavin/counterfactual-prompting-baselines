#!/usr/bin/env python3
"""
ABOUTME: Result merging for parallel GPU execution
ABOUTME: Combines checkpoint files from multiple GPUs into unified results
"""

import pickle
import os
import glob
from pathlib import Path


def merge_parallel_results(model_name, total_gpus, checkpoint_dir, output_dir, suffix=''):
    """
    Merge results from parallel GPU checkpoints into unified output

    Args:
        model_name: Name of model (e.g., 'deepseek_r1_0528')
        total_gpus: Total number of GPUs that ran in parallel
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Directory for merged output
        suffix: Optional suffix for checkpoint files (e.g., '_controlled')

    Returns:
        tuple: (merged_results, merged_logits_data)

    Raises:
        FileNotFoundError: If any GPU checkpoint is missing
    """
    all_results = []
    all_logits_samples = []

    # Load all GPU checkpoints
    for gpu_id in range(total_gpus):
        checkpoint_path = f"{checkpoint_dir}/{model_name}_gpu{gpu_id}_of_{total_gpus}{suffix}.pkl"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        all_results.extend(checkpoint_data.get('results', []))
        all_logits_samples.extend(checkpoint_data.get('logits_data', {}).get('samples', []))

    # Sort by case_idx to restore original order
    all_results.sort(key=lambda x: x.get('case_idx', 0))

    merged_logits_data = {'samples': all_logits_samples}

    # Save merged logits pickle
    logits_dir = f"{output_dir}/medqa_jsd_logits"
    os.makedirs(logits_dir, exist_ok=True)
    logits_path = f"{logits_dir}/medqa_jsd_logits_{model_name}.pkl"

    with open(logits_path, 'wb') as f:
        pickle.dump(merged_logits_data, f)

    print(f"Merged {len(all_results)} results from {total_gpus} GPUs")
    print(f"Saved logits to: {logits_path}")

    return all_results, merged_logits_data
