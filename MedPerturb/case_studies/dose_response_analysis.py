# ABOUTME: Analyze dose-response relationship between token change % and answer instability
# ABOUTME: Computes flip rate and MI at each level, generates plots with bootstrap error bars

import numpy as np
import pandas as pd
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def majority_vote(responses):
    """Return majority vote from list of 0/1 responses."""
    return 1 if sum(responses) > len(responses) / 2 else 0


def compute_flip_rate(orig_votes, para_votes):
    """Compute proportion of samples where answer flipped.

    Args:
        orig_votes: List of binary original answers
        para_votes: List of binary paraphrase answers

    Returns:
        float: Proportion of flipped answers
    """
    n = len(orig_votes)
    if n == 0:
        return 0.0
    flips = sum(1 for o, p in zip(orig_votes, para_votes) if o != p)
    return flips / n


def bootstrap_flip_rate_se(orig_votes, para_votes, n_bootstrap=1000):
    """Compute bootstrap SE of flip rate.

    Args:
        orig_votes: List of binary original answers
        para_votes: List of binary paraphrase answers
        n_bootstrap: Number of bootstrap iterations

    Returns:
        float: Standard error of flip rate
    """
    n = len(orig_votes)
    if n == 0:
        return 0.0
    orig_arr = np.array(orig_votes)
    para_arr = np.array(para_votes)
    rates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        flips = np.sum(orig_arr[idx] != para_arr[idx])
        rates.append(flips / n)
    return np.std(rates)


def calculate_mi(x, y):
    """Calculate mutual information between two binary arrays.

    Uses pd.crosstab-based algorithm matching baseline_analysis.py.

    Args:
        x: List/array of binary values
        y: List/array of binary values

    Returns:
        float: Mutual information in bits
    """
    x_s = pd.Series(x)
    y_s = pd.Series(y)
    joint = pd.crosstab(x_s, y_s, normalize=True)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            if joint.loc[i, j] > 0:
                mi += joint.loc[i, j] * np.log2(
                    joint.loc[i, j] / (p_x[i] * p_y[j])
                )
    return mi


QUESTIONS = ['MANAGE', 'VISIT', 'RESOURCE']


def run_analysis(eval_results, n_bootstrap=1000):
    """Run dose-response analysis on evaluation results.

    Args:
        eval_results: List of dicts from dose_response_evaluate.py output
        n_bootstrap: Number of bootstrap iterations

    Returns:
        pd.DataFrame with columns: target_pct, question, flip_rate,
            flip_rate_se, mi, mi_se, n_samples
    """
    # Discover target levels from keys
    target_pcts = set()
    for r in eval_results:
        for key in r:
            if key.startswith('pct'):
                pct_str = key.split('_')[0][3:]  # "pct5.0_MANAGE" -> "5.0"
                target_pcts.add(float(pct_str))
    target_pcts = sorted(target_pcts)

    rows = []
    for target_pct in target_pcts:
        for question in QUESTIONS:
            orig_votes = []
            para_votes = []

            for r in eval_results:
                orig_key = f'original_{question}'
                para_key = f'pct{target_pct}_{question}'

                if orig_key not in r or para_key not in r:
                    continue

                orig_votes.append(majority_vote(r[orig_key]['binary_answers']))
                para_votes.append(majority_vote(r[para_key]['binary_answers']))

            flip_rate = compute_flip_rate(orig_votes, para_votes)
            flip_se = bootstrap_flip_rate_se(orig_votes, para_votes, n_bootstrap)
            mi = calculate_mi(orig_votes, para_votes)
            mi_se_val = bootstrap_mi_se(orig_votes, para_votes, n_bootstrap)

            rows.append({
                'target_pct': target_pct,
                'question': question,
                'flip_rate': flip_rate,
                'flip_rate_se': flip_se,
                'mi': mi,
                'mi_se': mi_se_val,
                'n_samples': len(orig_votes),
            })

    df = pd.DataFrame(rows)

    # Warn if sample counts differ across target levels for the same question
    for question in QUESTIONS:
        q_rows = df[df['question'] == question]
        counts = q_rows['n_samples'].unique()
        if len(counts) > 1:
            print(f"WARNING: Inconsistent sample counts for {question} "
                  f"across target levels: {dict(zip(q_rows['target_pct'], q_rows['n_samples']))}")

    return df


def generate_plots(results_df, output_prefix):
    """Generate flip rate and MI dose-response plots.

    Args:
        results_df: DataFrame from run_analysis
        output_prefix: Path prefix for output files (e.g., 'results/dose_response')
    """
    if results_df.empty:
        print("Warning: No results to plot")
        return
    questions = results_df['question'].unique()

    # Flip rate plot
    fig, ax = plt.subplots(figsize=(4, 3))
    for q in questions:
        subset = results_df[results_df['question'] == q].sort_values('target_pct')
        ax.errorbar(
            subset['target_pct'], subset['flip_rate'],
            yerr=subset['flip_rate_se'],
            marker='o', capsize=3, label=q, linewidth=1.5, markersize=5,
        )
    ax.set_xlabel('Token Change %', fontsize=11)
    ax.set_ylabel('Flip Rate', fontsize=11)
    ax.set_xticks([5, 10, 20, 40, 60])
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_flip_rate.png', dpi=300)
    fig.savefig(f'{output_prefix}_flip_rate.pdf')
    plt.close(fig)

    # MI plot
    fig, ax = plt.subplots(figsize=(4, 3))
    for q in questions:
        subset = results_df[results_df['question'] == q].sort_values('target_pct')
        ax.errorbar(
            subset['target_pct'], subset['mi'],
            yerr=subset['mi_se'],
            marker='o', capsize=3, label=q, linewidth=1.5, markersize=5,
        )
    ax.set_xlabel('Token Change %', fontsize=11)
    ax.set_ylabel('Mutual Information (bits)', fontsize=11)
    ax.set_xticks([5, 10, 20, 40, 60])
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_mi.png', dpi=300)
    fig.savefig(f'{output_prefix}_mi.pdf')
    plt.close(fig)


def bootstrap_mi_se(x, y, n_bootstrap=1000):
    """Compute bootstrap SE of mutual information.

    Args:
        x: List/array of binary values
        y: List/array of binary values
        n_bootstrap: Number of bootstrap iterations

    Returns:
        float: Standard error of MI
    """
    n = len(x)
    if n == 0:
        return 0.0
    x_arr = np.array(x)
    y_arr = np.array(y)
    mis = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        mi = calculate_mi(x_arr[idx].tolist(), y_arr[idx].tolist())
        mis.append(mi)
    return np.std(mis)


def main():
    parser = argparse.ArgumentParser(description="Dose-response analysis")
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to dose-response evaluation JSON')
    parser.add_argument('--output', type=str,
                        default='results/dose_response_analysis.xlsx',
                        help='Output Excel path')
    parser.add_argument('--plot_prefix', type=str,
                        default='results/dose_response',
                        help='Prefix for plot output files')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')

    args = parser.parse_args()

    with open(args.evaluation, 'r') as f:
        eval_results = json.load(f)
    print(f"Loaded {len(eval_results)} evaluation results")

    results_df = run_analysis(eval_results, n_bootstrap=args.n_bootstrap)
    results_df.to_excel(args.output, index=False)

    print("\n" + "=" * 60)
    print("Dose-Response Analysis Results")
    print("=" * 60)
    for _, row in results_df.iterrows():
        print(f"  {row['target_pct']:5.1f}%  {row['question']:10}  "
              f"flip={row['flip_rate']:.3f}±{row['flip_rate_se']:.3f}  "
              f"MI={row['mi']:.4f}±{row['mi_se']:.4f}  "
              f"n={row['n_samples']}")

    generate_plots(results_df, args.plot_prefix)
    print(f"\nResults saved to: {args.output}")
    print(f"Plots saved to: {args.plot_prefix}_flip_rate.png, {args.plot_prefix}_mi.png")


if __name__ == "__main__":
    main()
