# ABOUTME: Jensen-Shannon Divergence between binary probability distributions
# ABOUTME: Per-sample metric with paired t-test for hypothesis testing

import numpy as np
from scipy import stats
from metrics import validate_prob_arrays


def compute(p_yes_1, p_yes_2):
    """Compute JSD between two binary distributions (bits, log base 2).

    Args:
        p_yes_1: P(Yes) for distribution 1
        p_yes_2: P(Yes) for distribution 2

    Returns:
        JSD value in [0, 1]
    """
    p = np.array([p_yes_1, 1 - p_yes_1])
    q = np.array([p_yes_2, 1 - p_yes_2])
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def paired_ttest(orig_probs, pert_probs, base_probs, alternative='two-sided'):
    """Paired t-test comparing JSD(orig,pert) vs JSD(orig,base) per sample.

    Args:
        orig_probs: array of P(Yes) for originals
        pert_probs: array of P(Yes) for perturbations
        base_probs: array of P(Yes) for baselines
        alternative: 'two-sided' (default), 'greater' (H1: JSD_pert > JSD_base),
            or 'less' (H1: JSD_pert < JSD_base). For JSD, the directional
            alternative when targeted has more effect is 'greater'.

    Returns:
        dict with mean_perturbation, mean_baseline, observed_diff,
        t_statistic, p_value
    """
    orig_probs = np.asarray(orig_probs)
    pert_probs = np.asarray(pert_probs)
    base_probs = np.asarray(base_probs)
    validate_prob_arrays(orig_probs, pert_probs, base_probs)

    jsd_pert = np.array([compute(o, p) for o, p in zip(orig_probs, pert_probs)])
    jsd_base = np.array([compute(o, b) for o, b in zip(orig_probs, base_probs)])

    differences = jsd_pert - jsd_base
    t_stat, p_value = stats.ttest_rel(jsd_pert, jsd_base, alternative=alternative)

    # ttest_rel returns NaN when all differences are zero (zero variance)
    if np.isnan(p_value):
        t_stat = 0.0
        p_value = 1.0

    return {
        "mean_perturbation": float(np.mean(jsd_pert)),
        "mean_baseline": float(np.mean(jsd_base)),
        "observed_diff": float(np.mean(differences)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }
