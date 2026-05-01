# ABOUTME: Phi correlation coefficient between binary answer vectors
# ABOUTME: Equivalent to Pearson correlation for binary data, with bootstrap test

import numpy as np
from metrics import validate_binary_vectors, bootstrap_test_common


def compute(x, y):
    """Compute phi correlation between two binary vectors.

    Returns 0.0 when either vector is constant (denominator zero).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    validate_binary_vectors(x, y)

    # 2x2 contingency counts
    n11 = np.sum((x == 1) & (y == 1))
    n10 = np.sum((x == 1) & (y == 0))
    n01 = np.sum((x == 0) & (y == 1))
    n00 = np.sum((x == 0) & (y == 0))

    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom


def bootstrap_test(orig, pert, base, n_bootstrap=1000, seed=42,
                   alternative='two-sided'):
    """Bootstrap test comparing compute(orig,pert) vs compute(orig,base).

    For phi, higher = more agreement = less perturbation effect, so the
    directional alternative when targeted has more effect is alternative='less'.
    """
    return bootstrap_test_common(compute, orig, pert, base, n_bootstrap, seed,
                                 alternative=alternative)
