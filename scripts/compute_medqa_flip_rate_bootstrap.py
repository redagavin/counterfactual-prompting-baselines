# ABOUTME: Computes bootstrap p-values for flip rate differences between gender perturbation
# ABOUTME: and paraphrase baseline in MedQA preliminary experiments.
"""Bootstrap test for flip rate difference (gender perturbation vs. paraphrase baseline).

Reads the controlled analysis Excel files and computes a two-tailed bootstrap
p-value for the difference in flip rates. Results are deterministic (fixed seed).
"""

import argparse
import os
import numpy as np
import pandas as pd


def bootstrap_flip_rate_pvalue(gender_flips, benign_flips, n_bootstrap=10000, seed=42):
    """Compute two-tailed bootstrap p-value for flip rate difference.

    Args:
        gender_flips: Boolean array, True if gender perturbation flipped the answer.
        benign_flips: Boolean array, True if benign perturbation flipped the answer.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        dict with observed_diff, ci_low, ci_high, p_value, gender_rate, benign_rate.
    """
    rng = np.random.RandomState(seed)
    n = len(gender_flips)
    gender_flips = np.asarray(gender_flips, dtype=int)
    benign_flips = np.asarray(benign_flips, dtype=int)

    observed_diff = gender_flips.mean() - benign_flips.mean()

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = gender_flips[idx].mean() - benign_flips[idx].mean()

    p_value_two_sided = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
    p_value_two_sided = min(p_value_two_sided, 1.0)
    p_value_one_sided = float(np.mean(boot_diffs <= 0))   # H1: gender_flip_rate > benign_flip_rate

    return {
        'gender_rate': gender_flips.mean(),
        'benign_rate': benign_flips.mean(),
        'observed_diff': observed_diff,
        'ci_low': np.percentile(boot_diffs, 2.5),
        'ci_high': np.percentile(boot_diffs, 97.5),
        'p_value': p_value_two_sided,                      # legacy (unchanged): two-sided
        'p_value_two_sided': p_value_two_sided,
        'p_value_one_sided': p_value_one_sided,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap flip rate test for MedQA")
    parser.add_argument('--results-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'),
                        help='Directory containing controlled analysis Excel files')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap resamples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    models = {
        '8B (DeepSeek-R1-0528-Qwen3-8B)': 'deepseek_r1_0528',
        '70B (DeepSeek-R1-Distill-Llama-70B)': 'deepseek_r1_70b',
    }

    for name, model_key in models.items():
        path = f'{args.results_dir}/medqa_jsd_controlled_analysis_{model_key}.xlsx'
        df = pd.read_excel(path, sheet_name='Analysis Results')

        gender_flips = ~df['gender_match']
        benign_flips = ~df['benign_match']

        result = bootstrap_flip_rate_pvalue(
            gender_flips.values, benign_flips.values,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )

        print(f"=== {name} ===")
        print(f"  N = {len(df)}")
        print(f"  Gender flip rate: {result['gender_rate']:.3f}")
        print(f"  Benign flip rate: {result['benign_rate']:.3f}")
        print(f"  Difference: {result['observed_diff']:.4f}")
        print(f"  95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
        print(f"  Bootstrap p-value (two-tailed): {result['p_value']:.3f}")  # legacy (unchanged)
        print(f"  Bootstrap p-value (one-sided H1: gender > benign): {result['p_value_one_sided']:.3f}")
        print(f"  Bootstrap p-value (two-sided): {result['p_value_two_sided']:.3f}")
        print()


if __name__ == "__main__":
    main()
