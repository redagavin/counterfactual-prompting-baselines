# ABOUTME: Statistical analysis for precision check (age swap negative control) using 5 metrics
# ABOUTME: Validates that age swap does NOT affect gender detection, confirming measurement precision

import numpy as np
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from metrics import mi, phi, flip_rate
from metrics import jsd, kl

POPULATION_METRICS = {"mi": mi, "phi": phi, "flip_rate": flip_rate}
SAMPLE_METRICS = {"jsd": jsd, "kl": kl}

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


def run_precision_check_analysis(eval_results, n_bootstrap=1000):
    """Run analysis on precision check evaluation results with all 5 metrics.

    Args:
        eval_results: List of dicts from precision_check_evaluate.py output.
            Each dict has original_GENDER, age_swap_GENDER,
            age_swap_baseline_GENDER, neutral_GENDER keys.
        n_bootstrap: Number of bootstrap iterations for population metrics

    Returns:
        dict: {
            'n_cases': int,
            'mi': {'calibrated': {...}, 'neutral': {...}},
            'phi': {'calibrated': {...}, 'neutral': {...}},
            'flip_rate': {'calibrated': {...}, 'neutral': {...}},
            'jsd': {'calibrated': {...}, 'neutral': {...}},
            'kl': {'calibrated': {...}, 'neutral': {...}},
        }
    """
    # Extract binary vectors (majority voted)
    orig_binary = []
    age_swap_binary = []
    calibrated_binary = []
    neutral_binary = []

    # Extract probability vectors
    orig_probs = []
    age_swap_probs = []
    calibrated_probs = []
    neutral_probs = []

    for r in eval_results:
        orig_binary.append(majority_vote(r['original_GENDER']['binary_answers']))
        age_swap_binary.append(majority_vote(r['age_swap_GENDER']['binary_answers']))
        calibrated_binary.append(majority_vote(r['age_swap_baseline_GENDER']['binary_answers']))
        neutral_binary.append(majority_vote(r['neutral_GENDER']['binary_answers']))

        orig_probs.append(r['original_GENDER']['logit_probs'])
        age_swap_probs.append(r['age_swap_GENDER']['logit_probs'])
        calibrated_probs.append(r['age_swap_baseline_GENDER']['logit_probs'])
        neutral_probs.append(r['neutral_GENDER']['logit_probs'])

    orig_bin = np.array(orig_binary)
    swap_bin = np.array(age_swap_binary)
    cal_bin = np.array(calibrated_binary)
    neut_bin = np.array(neutral_binary)

    orig_prob = np.array(orig_probs)
    swap_prob = np.array(age_swap_probs)
    cal_prob = np.array(calibrated_probs)
    neut_prob = np.array(neutral_probs)

    baseline_vectors = {
        'calibrated': (cal_bin, cal_prob),
        'neutral': (neut_bin, neut_prob),
    }

    results = {'n_cases': len(eval_results)}

    for name, module in POPULATION_METRICS.items():
        results[name] = {}
        for btype, (base_bin, _) in baseline_vectors.items():
            test_two = module.bootstrap_test(
                orig_bin, swap_bin, base_bin, n_bootstrap=n_bootstrap
            )
            test_one = module.bootstrap_test(
                orig_bin, swap_bin, base_bin, n_bootstrap=n_bootstrap,
                alternative=ONE_SIDED_ALTERNATIVE[name]
            )
            test_two['p_value_two_sided'] = test_two['p_value']
            test_two['p_value_one_sided'] = test_one['p_value']
            results[name][btype] = test_two

    for name, module in SAMPLE_METRICS.items():
        results[name] = {}
        for btype, (_, base_prob) in baseline_vectors.items():
            test_two = module.paired_ttest(
                orig_prob, swap_prob, base_prob
            )
            test_one = module.paired_ttest(
                orig_prob, swap_prob, base_prob,
                alternative=ONE_SIDED_ALTERNATIVE[name]
            )
            test_two['p_value_two_sided'] = test_two['p_value']
            test_two['p_value_one_sided'] = test_one['p_value']
            results[name][btype] = test_two

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Precision check analysis (age swap negative control) with 5 metrics"
    )
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to precision check evaluation JSON')
    parser.add_argument('--output', type=str, default='results/precision_check_analysis.xlsx',
                        help='Output Excel path')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')

    args = parser.parse_args()

    with open(args.evaluation, 'r') as f:
        eval_results = json.load(f)
    print(f"Loaded {len(eval_results)} evaluation results")

    results = run_precision_check_analysis(eval_results, n_bootstrap=args.n_bootstrap)

    # Flatten to DataFrame for output
    import pandas as pd
    rows = []
    for metric_name in list(POPULATION_METRICS) + list(SAMPLE_METRICS):
        for btype in ['calibrated', 'neutral']:
            r = results[metric_name][btype]
            rows.append({
                'metric': metric_name,
                'baseline_type': btype,
                'n_cases': results['n_cases'],
                'observed_diff': r['observed_diff'],
                'p_value': r['p_value'],                       # legacy: two-sided (unchanged)
                'p_value_two_sided': r['p_value_two_sided'],
                'p_value_one_sided': r['p_value_one_sided'],
            })

    results_df = pd.DataFrame(rows)
    results_df.to_excel(args.output, index=False)

    print("\n" + "=" * 60)
    print("Precision Check Results (Age Swap Negative Control)")
    print("=" * 60)
    print(f"N cases: {results['n_cases']}")
    for _, row in results_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
        print(f"  {row['metric']:10} {row['baseline_type']:12} "
              f"diff={row['observed_diff']:+.4f} p={row['p_value']:.4f} {sig}")

    # Overall pass/fail check
    all_pass = all(
        results[m][b]['p_value'] > 0.05
        for m in list(POPULATION_METRICS) + list(SAMPLE_METRICS)
        for b in ['calibrated', 'neutral']
    )
    if all_pass:
        print("\nPRECISION CHECK PASSED: Age swap does not significantly affect gender detection")
    else:
        print("\nWARNING: Some metrics show significant effect of age swap on gender detection")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
