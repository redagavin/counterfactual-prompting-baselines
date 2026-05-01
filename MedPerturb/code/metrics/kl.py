# ABOUTME: KL Divergence from original to perturbation distribution
# ABOUTME: KL(original || perturbation) with epsilon floor, paired t-test

import numpy as np
from scipy import stats
from metrics import validate_prob_arrays


def compute(p_yes_orig, p_yes_pert, epsilon=1e-10):
    """Compute KL(original || perturbation) for binary distributions.

    Original is the reference distribution. Epsilon floor prevents infinity.

    Args:
        p_yes_orig: P(Yes) for original
        p_yes_pert: P(Yes) for perturbation
        epsilon: floor for numerical stability

    Returns:
        KL divergence (non-negative, unbounded)
    """
    p = np.array([max(p_yes_orig, epsilon), max(1 - p_yes_orig, epsilon)])
    q = np.array([max(p_yes_pert, epsilon), max(1 - p_yes_pert, epsilon)])

    # Renormalize after epsilon floor
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log2(p / q)))


def paired_ttest(orig_probs, pert_probs, base_probs, alternative='two-sided'):
    """Paired t-test comparing KL(orig||pert) vs KL(orig||base) per sample.

    Args:
        orig_probs: array of P(Yes) for originals
        pert_probs: array of P(Yes) for perturbations
        base_probs: array of P(Yes) for baselines
        alternative: 'two-sided' (default), 'greater' (H1: KL_pert > KL_base),
            or 'less' (H1: KL_pert < KL_base). For KL, the directional
            alternative when targeted has more effect is 'greater'.

    Returns:
        dict with mean_perturbation, mean_baseline, observed_diff,
        t_statistic, p_value
    """
    orig_probs = np.asarray(orig_probs)
    pert_probs = np.asarray(pert_probs)
    base_probs = np.asarray(base_probs)
    validate_prob_arrays(orig_probs, pert_probs, base_probs)

    kl_pert = np.array([compute(o, p) for o, p in zip(orig_probs, pert_probs)])
    kl_base = np.array([compute(o, b) for o, b in zip(orig_probs, base_probs)])

    differences = kl_pert - kl_base
    t_stat, p_value = stats.ttest_rel(kl_pert, kl_base, alternative=alternative)

    # ttest_rel returns NaN when all differences are zero (zero variance)
    if np.isnan(p_value):
        t_stat = 0.0
        p_value = 1.0

    return {
        "mean_perturbation": float(np.mean(kl_pert)),
        "mean_baseline": float(np.mean(kl_base)),
        "observed_diff": float(np.mean(differences)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }
