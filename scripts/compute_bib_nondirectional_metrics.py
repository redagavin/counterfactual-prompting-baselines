# ABOUTME: Compute non-directional metrics (JSD, KL, flip rate, MI, phi) for bias-in-bios experiment
# ABOUTME: Compares gender swap vs paraphrase and fixed sentence baselines with statistical testing

import json

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.special import expit  # sigmoid


def logit_to_probs(logit):
    """Convert log-odds (logit_A - logit_B) to [P(A), P(B)] distribution."""
    p_a = expit(logit)
    return np.array([p_a, 1 - p_a])


def compute_jsd(logit_orig, logit_pert):
    """Per-sample JSD between original and perturbed distributions."""
    p = logit_to_probs(logit_orig)
    q = logit_to_probs(logit_pert)
    return jensenshannon(p, q, base=2) ** 2  # scipy returns sqrt(JSD), square for JSD in [0,1]


def compute_kl(logit_orig, logit_pert):
    """Per-sample KL(original || perturbed)."""
    p = logit_to_probs(logit_orig)
    q = logit_to_probs(logit_pert)
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    q = np.clip(q, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)
    return np.sum(p * np.log2(p / q))


def compute_flip_rate(logit_orig, logit_pert):
    """Fraction of samples where the binary prediction flips."""
    pred_orig = (np.array(logit_orig) > 0).astype(int)
    pred_pert = (np.array(logit_pert) > 0).astype(int)
    return np.mean(pred_orig != pred_pert)


def compute_mi(logit_orig, logit_pert):
    """Mutual information between original and perturbed binary predictions."""
    pred_orig = (np.array(logit_orig) > 0).astype(int)
    pred_pert = (np.array(logit_pert) > 0).astype(int)
    # 2x2 contingency table
    n = len(pred_orig)
    joint = np.zeros((2, 2))
    for a, b in zip(pred_orig, pred_pert):
        joint[a, b] += 1
    joint /= n
    marginal_a = joint.sum(axis=1)
    marginal_b = joint.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and marginal_a[i] > 0 and marginal_b[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (marginal_a[i] * marginal_b[j]))
    return mi


def compute_phi(logit_orig, logit_pert):
    """Phi coefficient between original and perturbed binary predictions."""
    pred_orig = (np.array(logit_orig) > 0).astype(int)
    pred_pert = (np.array(logit_pert) > 0).astype(int)
    n11 = np.sum((pred_orig == 1) & (pred_pert == 1))
    n00 = np.sum((pred_orig == 0) & (pred_pert == 0))
    n10 = np.sum((pred_orig == 1) & (pred_pert == 0))
    n01 = np.sum((pred_orig == 0) & (pred_pert == 1))
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom


def paired_t_test(targeted_values, baseline_values):
    """One-sided paired t-test: H1: targeted > baseline."""
    diffs = np.array(targeted_values) - np.array(baseline_values)
    t_stat, p_two = stats.ttest_rel(targeted_values, baseline_values)
    if np.isnan(p_two):
        t_stat, p_two = 0.0, 1.0
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return {
        "mean_targeted": np.mean(targeted_values),
        "mean_baseline": np.mean(baseline_values),
        "mean_diff": np.mean(diffs),
        "t_stat": t_stat,
        "p_one_sided": p_one,           # legacy (unchanged): one-sided H1: targeted > baseline
        "p_two_sided": float(p_two),    # new: scipy default two-sided
    }


def bootstrap_test(logit_orig, logit_targeted, logit_baseline, metric_fn, n_bootstrap=1000, seed=42):
    """Bootstrap test for aggregate metrics: H1: targeted metric > baseline metric."""
    rng = np.random.default_rng(seed)
    n = len(logit_orig)

    observed_targeted = metric_fn(logit_orig, logit_targeted)
    observed_baseline = metric_fn(logit_orig, logit_baseline)
    observed_diff = observed_targeted - observed_baseline

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        lo = [logit_orig[i] for i in idx]
        lt = [logit_targeted[i] for i in idx]
        lb = [logit_baseline[i] for i in idx]
        boot_diffs.append(metric_fn(lo, lt) - metric_fn(lo, lb))

    boot_diffs = np.array(boot_diffs)
    if observed_diff >= 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)
    p_value = min(p_value, 1.0)
    p_one_sided_greater = float(np.mean(boot_diffs <= 0))   # H1: targeted > baseline (FR direction)
    p_one_sided_less    = float(np.mean(boot_diffs >= 0))   # H1: targeted < baseline (MI/φ direction)

    return {
        "targeted": observed_targeted,
        "baseline": observed_baseline,
        "diff": observed_diff,
        "p_value": p_value,                            # legacy (unchanged): two-sided
        "p_two_sided": p_value,                        # new: alias
        "p_one_sided_greater": p_one_sided_greater,    # new
        "p_one_sided_less": p_one_sided_less,          # new
        "ci_lo": np.percentile(boot_diffs, 2.5),
        "ci_hi": np.percentile(boot_diffs, 97.5),
    }


def analyze_one(eval_results, baseline_type, label):
    """Compute all metrics for one (model, split, baseline) condition.

    Returns a dict with all metric values and test results.
    """
    logit_orig = [r["logit_original"] for r in eval_results]
    logit_swap = [r["logit_swapped"] for r in eval_results]
    logit_key = f"logit_{baseline_type}"
    logit_base = [r[logit_key] for r in eval_results]

    n = len(eval_results)

    # Per-sample metrics
    jsd_targeted = [compute_jsd(logit_orig[i], logit_swap[i]) for i in range(n)]
    jsd_baseline = [compute_jsd(logit_orig[i], logit_base[i]) for i in range(n)]
    kl_targeted = [compute_kl(logit_orig[i], logit_swap[i]) for i in range(n)]
    kl_baseline = [compute_kl(logit_orig[i], logit_base[i]) for i in range(n)]

    jsd_test = paired_t_test(jsd_targeted, jsd_baseline)
    kl_test = paired_t_test(kl_targeted, kl_baseline)

    # Aggregate metrics
    fr_test = bootstrap_test(logit_orig, logit_swap, logit_base, compute_flip_rate)
    mi_test = bootstrap_test(logit_orig, logit_swap, logit_base, compute_mi)
    phi_test = bootstrap_test(logit_orig, logit_swap, logit_base, compute_phi)

    print(f"\n{'='*70}")
    print(f"  {label} (N={n})")
    print(f"{'='*70}")

    print(f"\n  Per-sample metrics (paired t-test, H1: targeted > baseline):")
    print(f"  {'Metric':<6} {'Targeted':>10} {'Baseline':>10} {'Diff':>10} {'t':>8} {'p (1-sided)':>12}")
    print(f"  {'JSD':<6} {jsd_test['mean_targeted']:>10.4f} {jsd_test['mean_baseline']:>10.4f} "
          f"{jsd_test['mean_diff']:>10.4f} {jsd_test['t_stat']:>8.2f} {jsd_test['p_one_sided']:>12.4e}")
    print(f"  {'KL':<6} {kl_test['mean_targeted']:>10.4f} {kl_test['mean_baseline']:>10.4f} "
          f"{kl_test['mean_diff']:>10.4f} {kl_test['t_stat']:>8.2f} {kl_test['p_one_sided']:>12.4e}")

    print(f"\n  Aggregate metrics (bootstrap test, 1000 resamples):")
    print(f"  {'Metric':<6} {'Targeted':>10} {'Baseline':>10} {'Diff':>10} {'p (2-sided)':>12} {'95% CI':>20}")
    for name, test in [("FR", fr_test), ("MI", mi_test), ("Phi", phi_test)]:
        print(f"  {name:<6} {test['targeted']:>10.4f} {test['baseline']:>10.4f} "
              f"{test['diff']:>10.4f} {test['p_value']:>12.4f} "
              f"[{test['ci_lo']:>8.4f}, {test['ci_hi']:>8.4f}]")

    return {
        "label": label,
        "n": n,
        "jsd": jsd_test,
        "kl": kl_test,
        "fr": fr_test,
        "mi": mi_test,
        "phi": phi_test,
    }


def main():
    import os
    import pandas as pd

    models = [
        ("qwen3_8b", "Qwen3-8B"),
        ("qwen3_32b", "Qwen3-32B"),
    ]
    splits = ["train", "test"]
    baselines = [
        ("paraphrase", "Paraphrase"),
        ("fixed_sentence", "Fixed sentence"),
    ]

    all_rows = []
    for model_key, model_name in models:
        for split in splits:
            path = f"results/bib/bib_eval_{split}_{model_key}_merged.json"
            with open(path) as f:
                eval_results = json.load(f)

            for baseline_key, baseline_name in baselines:
                label = f"{model_name} / {split} / {baseline_name}"
                result = analyze_one(eval_results, baseline_key, label)
                all_rows.append({
                    "model": model_name,
                    "split": split,
                    "baseline": baseline_name,
                    "n": result["n"],
                    "jsd_targeted": result["jsd"]["mean_targeted"],
                    "jsd_baseline": result["jsd"]["mean_baseline"],
                    "jsd_diff": result["jsd"]["mean_diff"],
                    "jsd_t": result["jsd"]["t_stat"],
                    "jsd_p": result["jsd"]["p_one_sided"],                       # legacy (unchanged — already one-sided)
                    "jsd_p_one_sided": result["jsd"]["p_one_sided"],             # explicit name
                    "jsd_p_two_sided": result["jsd"]["p_two_sided"],             # new: scipy default
                    "kl_targeted": result["kl"]["mean_targeted"],
                    "kl_baseline": result["kl"]["mean_baseline"],
                    "kl_diff": result["kl"]["mean_diff"],
                    "kl_t": result["kl"]["t_stat"],
                    "kl_p": result["kl"]["p_one_sided"],                         # legacy (unchanged)
                    "kl_p_one_sided": result["kl"]["p_one_sided"],
                    "kl_p_two_sided": result["kl"]["p_two_sided"],
                    "fr_targeted": result["fr"]["targeted"],
                    "fr_baseline": result["fr"]["baseline"],
                    "fr_diff": result["fr"]["diff"],
                    "fr_p": result["fr"]["p_value"],                             # legacy (unchanged — was two-sided)
                    "fr_p_one_sided": result["fr"]["p_one_sided_greater"],       # H1: gender_FR > benign_FR
                    "fr_p_two_sided": result["fr"]["p_two_sided"],
                    "fr_ci_lo": result["fr"]["ci_lo"],
                    "fr_ci_hi": result["fr"]["ci_hi"],
                    "mi_targeted": result["mi"]["targeted"],
                    "mi_baseline": result["mi"]["baseline"],
                    "mi_diff": result["mi"]["diff"],
                    "mi_p": result["mi"]["p_value"],                             # legacy (unchanged — was two-sided)
                    "mi_p_one_sided": result["mi"]["p_one_sided_less"],          # H1: gender_MI < benign_MI
                    "mi_p_two_sided": result["mi"]["p_two_sided"],
                    "mi_ci_lo": result["mi"]["ci_lo"],
                    "mi_ci_hi": result["mi"]["ci_hi"],
                    "phi_targeted": result["phi"]["targeted"],
                    "phi_baseline": result["phi"]["baseline"],
                    "phi_diff": result["phi"]["diff"],
                    "phi_p": result["phi"]["p_value"],                           # legacy (unchanged)
                    "phi_p_one_sided": result["phi"]["p_one_sided_less"],
                    "phi_p_two_sided": result["phi"]["p_two_sided"],
                    "phi_ci_lo": result["phi"]["ci_lo"],
                    "phi_ci_hi": result["phi"]["ci_hi"],
                })

    df = pd.DataFrame(all_rows)
    output_dir = "results/bib"
    os.makedirs(output_dir, exist_ok=True)

    xlsx_path = os.path.join(output_dir, "bib_nondirectional_metrics.xlsx")
    json_path = os.path.join(output_dir, "bib_nondirectional_metrics.json")

    df.to_excel(xlsx_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nResults saved to:")
    print(f"  {xlsx_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
