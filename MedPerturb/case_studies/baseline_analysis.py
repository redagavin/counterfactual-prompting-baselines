# ABOUTME: Statistical analysis comparing perturbation effects vs baselines using 5 metrics
# ABOUTME: Loads JSON evaluation results, applies Bonferroni correction, supports calibrated + neutral baselines

import numpy as np
import pandas as pd
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from metrics import mi, phi, flip_rate
from metrics import jsd, kl

POPULATION_METRICS = {"mi": mi, "phi": phi, "flip_rate": flip_rate}
SAMPLE_METRICS = {"jsd": jsd, "kl": kl}

PERTURBATION_TYPES = ['gender_swap', 'gender_remove', 'uncertain', 'colorful']
TASKS = ['MANAGE', 'VISIT', 'RESOURCE']
BONFERRONI_ALPHA = 0.05
BONFERRONI_N_TESTS = len(PERTURBATION_TYPES) * len(TASKS)  # 12
BONFERRONI_THRESHOLD = BONFERRONI_ALPHA / BONFERRONI_N_TESTS

# Per-metric directional alternative for one-sided test (paper convention)
ONE_SIDED_ALTERNATIVE = {
    "flip_rate": "greater",
    "jsd": "greater",
    "kl": "greater",
    "mi": "less",
    "phi": "less",
}


def majority_vote(responses):
    """Return majority vote from list of 0/1 responses."""
    return 1 if sum(responses) > len(responses) / 2 else 0


def _extract_vectors(eval_results, pert_type, baseline_type, task):
    """Extract aligned binary and probability vectors from evaluation results.

    Args:
        eval_results: List of sample dicts from main_evaluate.py output
        pert_type: Perturbation name (e.g. 'gender_swap')
        baseline_type: 'calibrated' or 'neutral'
        task: Question type (e.g. 'MANAGE')

    Returns:
        tuple: (orig_binary, pert_binary, base_binary,
                orig_probs, pert_probs, base_probs)
    """
    orig_binary = []
    pert_binary = []
    base_binary = []
    orig_probs = []
    pert_probs = []
    base_probs = []

    orig_key = f'original_{task}'
    pert_key = f'{pert_type}_{task}'
    if baseline_type == 'calibrated':
        base_key = f'{pert_type}_baseline_{task}'
    else:
        base_key = f'neutral_{task}'

    for r in eval_results:
        if orig_key not in r or pert_key not in r or base_key not in r:
            continue

        orig_binary.append(majority_vote(r[orig_key]['binary_answers']))
        pert_binary.append(majority_vote(r[pert_key]['binary_answers']))
        base_binary.append(majority_vote(r[base_key]['binary_answers']))

        orig_probs.append(r[orig_key]['logit_probs'])
        pert_probs.append(r[pert_key]['logit_probs'])
        base_probs.append(r[base_key]['logit_probs'])

    return (np.array(orig_binary), np.array(pert_binary), np.array(base_binary),
            np.array(orig_probs), np.array(pert_probs), np.array(base_probs))


def run_analysis(eval_path, output_path):
    """Run full analysis with all 5 metrics, both baselines, Bonferroni correction.

    Args:
        eval_path: Path to JSON evaluation results from main_evaluate.py
        output_path: Path for output Excel file
    """
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)

    results = []

    for pert_type in PERTURBATION_TYPES:
        for baseline_type in ['calibrated', 'neutral']:
            for task in TASKS:
                vectors = _extract_vectors(
                    eval_results, pert_type, baseline_type, task
                )
                orig_bin, pert_bin, base_bin = vectors[:3]
                orig_prob, pert_prob, base_prob = vectors[3:]

                n_cases = len(orig_bin)
                if n_cases == 0:
                    continue

                # Population metrics (binary vectors)
                for name, module in POPULATION_METRICS.items():
                    test_result = module.bootstrap_test(
                        orig_bin, pert_bin, base_bin
                    )
                    test_result_one = module.bootstrap_test(
                        orig_bin, pert_bin, base_bin,
                        alternative=ONE_SIDED_ALTERNATIVE[name],
                    )
                    results.append({
                        'perturbation_type': pert_type,
                        'baseline_type': baseline_type,
                        'task': task,
                        'metric': name,
                        'n_cases': n_cases,
                        'observed_diff': test_result['observed_diff'],
                        'p_value': test_result['p_value'],                          # legacy: two-sided (unchanged)
                        'p_value_two_sided': test_result['p_value'],
                        'p_value_one_sided': test_result_one['p_value'],
                        'ci_low': test_result.get('ci_low'),
                        'ci_high': test_result.get('ci_high'),
                        'statistic_perturbation': test_result['statistic_perturbation'],
                        'statistic_baseline': test_result['statistic_baseline'],
                    })

                # Sample metrics (probability vectors)
                for name, module in SAMPLE_METRICS.items():
                    test_result = module.paired_ttest(
                        orig_prob, pert_prob, base_prob
                    )
                    test_result_one = module.paired_ttest(
                        orig_prob, pert_prob, base_prob,
                        alternative=ONE_SIDED_ALTERNATIVE[name],
                    )
                    results.append({
                        'perturbation_type': pert_type,
                        'baseline_type': baseline_type,
                        'task': task,
                        'metric': name,
                        'n_cases': n_cases,
                        'observed_diff': test_result['observed_diff'],
                        'p_value': test_result['p_value'],                          # legacy: two-sided (unchanged)
                        'p_value_two_sided': test_result['p_value'],
                        'p_value_one_sided': test_result_one['p_value'],
                        'ci_low': None,
                        'ci_high': None,
                        'statistic_perturbation': test_result['mean_perturbation'],
                        'statistic_baseline': test_result['mean_baseline'],
                    })

    results_df = pd.DataFrame(results)

    # Bonferroni correction: reject if raw p < alpha / n_tests
    results_df['significant'] = results_df['p_value'] < BONFERRONI_THRESHOLD

    results_df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Bonferroni threshold: p < {BONFERRONI_THRESHOLD:.6f} "
          f"(alpha={BONFERRONI_ALPHA}, n_tests={BONFERRONI_N_TESTS})")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for _, row in results_df.iterrows():
        sig = "*" if row['significant'] else ""
        print(f"{row['perturbation_type']:15} {row['baseline_type']:12} {row['task']:10} "
              f"{row['metric']:10} diff={row['observed_diff']:+.4f} "
              f"p={row['p_value']:.4f} {sig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perturbation vs baseline analysis with 5 metrics and Bonferroni correction"
    )
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to JSON evaluation results')
    parser.add_argument('--output', type=str, default='results/baseline_analysis.xlsx',
                        help='Output Excel path')

    args = parser.parse_args()

    run_analysis(args.evaluation, args.output)
