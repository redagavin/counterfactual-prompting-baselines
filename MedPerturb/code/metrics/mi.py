# ABOUTME: Mutual Information metric between binary answer vectors
# ABOUTME: Computes MI in bits from 2x2 contingency table with bootstrap test

import numpy as np
from metrics import validate_binary_vectors, bootstrap_test_common


def compute(x, y):
    """Compute mutual information between two binary vectors (bits)."""
    x = np.asarray(x)
    y = np.asarray(y)
    validate_binary_vectors(x, y)
    n = len(x)

    # 2x2 contingency table (joint probabilities)
    ct = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ct[i, j] = np.sum((x == i) & (y == j))
    ct /= n

    px = ct.sum(axis=1)
    py = ct.sum(axis=0)

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if ct[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += ct[i, j] * np.log2(ct[i, j] / (px[i] * py[j]))
    return mi


def bootstrap_test(orig, pert, base, n_bootstrap=1000, seed=42,
                   alternative='two-sided'):
    """Bootstrap test comparing compute(orig,pert) vs compute(orig,base).

    For MI, higher = more agreement = less perturbation effect, so the
    directional alternative when targeted has more effect is alternative='less'.
    """
    return bootstrap_test_common(compute, orig, pert, base, n_bootstrap, seed,
                                 alternative=alternative)
