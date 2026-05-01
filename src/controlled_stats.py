# ABOUTME: Statistical analysis functions for controlled perturbation experiments
# ABOUTME: Provides paired tests, effect sizes, and flip overlap metrics

import numpy as np
from scipy import stats


def compute_paired_ttest(jsd_gender, jsd_benign):
    """
    Compute paired t-test for JSD differences.

    Args:
        jsd_gender: List of JSD values for gender perturbation
        jsd_benign: List of JSD values for benign perturbation

    Returns:
        dict: {'statistic': float, 'p_value': float, 'p_value_two_sided': float, 'p_value_one_sided': float}
    """
    statistic, p_value = stats.ttest_rel(jsd_gender, jsd_benign)
    _, p_value_one = stats.ttest_rel(jsd_gender, jsd_benign, alternative='greater')
    return {
        'statistic': statistic,
        'p_value': p_value,                  # legacy: two-sided (unchanged)
        'p_value_two_sided': p_value,        # explicit alias
        'p_value_one_sided': p_value_one,    # H1: gender > benign (paper convention for JSD)
    }


def compute_wilcoxon_test(jsd_gender, jsd_benign):
    """
    Compute Wilcoxon signed-rank test for JSD differences.

    Args:
        jsd_gender: List of JSD values for gender perturbation
        jsd_benign: List of JSD values for benign perturbation

    Returns:
        dict: {'statistic': float, 'p_value': float}
    """
    differences = np.array(jsd_gender) - np.array(jsd_benign)
    # Filter out zeros (ties)
    non_zero_diffs = differences[differences != 0]

    if len(non_zero_diffs) == 0:
        return {'statistic': 0.0, 'p_value': 1.0}

    statistic, p_value = stats.wilcoxon(non_zero_diffs)
    return {'statistic': statistic, 'p_value': p_value}


def compute_cohens_d_paired(jsd_gender, jsd_benign):
    """
    Compute Cohen's d for paired samples.

    Args:
        jsd_gender: List of JSD values for gender perturbation
        jsd_benign: List of JSD values for benign perturbation

    Returns:
        float: Cohen's d effect size
    """
    differences = np.array(jsd_gender) - np.array(jsd_benign)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    if std_diff == 0:
        return 0.0

    return mean_diff / std_diff


def compute_flip_overlap_metrics(gender_flips, benign_flips):
    """
    Compute flip set overlap metrics.

    Args:
        gender_flips: List of booleans indicating gender perturbation flipped answer
        benign_flips: List of booleans indicating benign perturbation flipped answer

    Returns:
        dict: {
            'jaccard': float,
            'p_benign_given_gender': float,
            'p_gender_given_benign': float,
            'gender_specific_rate': float,
            'benign_specific_rate': float
        }
    """
    gender_set = set(i for i, f in enumerate(gender_flips) if f)
    benign_set = set(i for i, f in enumerate(benign_flips) if f)

    intersection = gender_set & benign_set
    union = gender_set | benign_set

    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 0.0

    # Conditional probabilities
    p_benign_given_gender = len(intersection) / len(gender_set) if gender_set else 0.0
    p_gender_given_benign = len(intersection) / len(benign_set) if benign_set else 0.0

    # Specific flip rates
    gender_only = gender_set - benign_set
    benign_only = benign_set - gender_set

    gender_specific_rate = len(gender_only) / len(gender_set) if gender_set else 0.0
    benign_specific_rate = len(benign_only) / len(benign_set) if benign_set else 0.0

    return {
        'jaccard': jaccard,
        'p_benign_given_gender': p_benign_given_gender,
        'p_gender_given_benign': p_gender_given_benign,
        'gender_specific_rate': gender_specific_rate,
        'benign_specific_rate': benign_specific_rate
    }


def compute_mcnemar_test(gender_flips, benign_flips):
    """
    Compute McNemar's test for paired nominal data.

    Args:
        gender_flips: List of booleans indicating gender perturbation flipped answer
        benign_flips: List of booleans indicating benign perturbation flipped answer

    Returns:
        dict: {'statistic': float, 'p_value': float, 'contingency_table': 2x2 array}
    """
    # Build contingency table
    # [gender_no_flip & benign_no_flip, gender_no_flip & benign_flip]
    # [gender_flip & benign_no_flip,    gender_flip & benign_flip]
    a = sum(1 for g, b in zip(gender_flips, benign_flips) if not g and not b)
    b = sum(1 for g, b in zip(gender_flips, benign_flips) if not g and b)
    c = sum(1 for g, b in zip(gender_flips, benign_flips) if g and not b)
    d = sum(1 for g, b in zip(gender_flips, benign_flips) if g and b)

    contingency_table = np.array([[a, b], [c, d]])

    # McNemar's test uses the off-diagonal elements
    if b + c == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'contingency_table': contingency_table
        }

    # McNemar's test statistic (with continuity correction)
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'contingency_table': contingency_table
    }
