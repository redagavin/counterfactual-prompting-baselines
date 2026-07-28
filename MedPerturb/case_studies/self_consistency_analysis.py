# ABOUTME: Self-consistency baseline analysis for MedPerturb main experiment
# ABOUTME: Computes per-case empirical JSD and aggregate permutation tests

import argparse
import json
import math
import sys
import numpy as np
import pandas as pd


def jsd_bernoulli(p: float, q: float, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q).

    Uses 0*log(0) = 0 convention. Bounded in [0, log_base(2)]; base=2 -> max 1.0.

    Boundary safety: the only case where m=0 (or 1-m=0) is when p=q=0 (or p=q=1).
    The p == q short-circuit catches both, so division by zero is impossible
    in the subsequent kl_term calls.
    """
    if p == q:
        return 0.0
    m = (p + q) / 2

    def kl_term(x, y):
        if x == 0:
            return 0.0
        return x * math.log(x / y, base)

    kl_p_m = kl_term(p, m) + kl_term(1 - p, 1 - m)
    kl_q_m = kl_term(q, m) + kl_term(1 - q, 1 - m)
    return (kl_p_m + kl_q_m) / 2


MIN_CLEAN_SAMPLES = 5


def permutation_test_cell(cases, perturbation, task, n_perms, rng,
                          base=2.0, drop_log=None):
    """For one (perturbation, task), compute observed mean JSD across cases plus
    aggregate permutation-test p-value via within-case label shuffling."""
    orig_key = f"original_{task}"
    pert_key = f"{perturbation}_{task}"

    per_case_pools = []
    per_case_observed = []
    n_dropped_missing_key = 0
    n_dropped_empty = 0
    n_dropped_low_count = 0

    for case in cases:
        cid = case.get('context_id', '<unknown>')
        if orig_key not in case or pert_key not in case:
            n_dropped_missing_key += 1
            if drop_log is not None:
                drop_log.append((cid, perturbation, task, 'missing_key'))
            continue
        orig = [x for x in case[orig_key]['binary_answers'] if x in (0, 1)]
        pert = [x for x in case[pert_key]['binary_answers'] if x in (0, 1)]
        if not orig or not pert:
            n_dropped_empty += 1
            if drop_log is not None:
                drop_log.append((cid, perturbation, task, 'empty_after_parse_filter'))
            continue
        if len(orig) < MIN_CLEAN_SAMPLES or len(pert) < MIN_CLEAN_SAMPLES:
            n_dropped_low_count += 1
            if drop_log is not None:
                drop_log.append((cid, perturbation, task,
                                 f'low_count_orig={len(orig)}_pert={len(pert)}'))
            continue
        per_case_observed.append(jsd_bernoulli(
            sum(orig)/len(orig), sum(pert)/len(pert), base=base
        ))
        per_case_pools.append((orig, pert))

    if not per_case_pools:
        print(f"WARNING: no usable cases for ({perturbation}, {task})", file=sys.stderr)
        return {
            'observed_mean_jsd': float('nan'),
            'null_mean_jsd': float('nan'),
            'null_std_jsd': float('nan'),
            'jsd_excess': float('nan'),
            'p_value': float('nan'),
            'n_cases_used': 0,
            'n_active': 0,
            'n_dropped_missing_key': n_dropped_missing_key,
            'n_dropped_empty': n_dropped_empty,
            'n_dropped_low_count': n_dropped_low_count,
        }

    observed_mean = float(np.mean(per_case_observed))

    def _is_active(orig, pert):
        pool = orig + pert
        return not (all(x == 0 for x in pool) or all(x == 1 for x in pool))
    n_active = sum(1 for o, p in per_case_pools if _is_active(o, p))

    null_means = np.empty(n_perms)
    for k in range(n_perms):
        per_case_null = []
        for orig, pert in per_case_pools:
            pool = list(orig) + list(pert)
            n_orig = len(orig)
            rng.shuffle(pool)
            p_emp = sum(pool[:n_orig]) / n_orig
            q_emp = sum(pool[n_orig:]) / (len(pool) - n_orig)
            per_case_null.append(jsd_bernoulli(p_emp, q_emp, base=base))
        null_means[k] = np.mean(per_case_null)

    # Phipson & Smyth (2010) unbiased exact p-value
    p_value = float((1 + (null_means >= observed_mean).sum()) / (1 + n_perms))

    return {
        'observed_mean_jsd': observed_mean,
        'null_mean_jsd': float(null_means.mean()),
        'null_std_jsd': float(null_means.std()),
        'jsd_excess': observed_mean - float(null_means.mean()),
        'p_value': p_value,
        'n_cases_used': len(per_case_pools),
        'n_active': n_active,
        'n_dropped_missing_key': n_dropped_missing_key,
        'n_dropped_empty': n_dropped_empty,
        'n_dropped_low_count': n_dropped_low_count,
    }


PERTURBATIONS = ['gender_swap', 'gender_remove', 'uncertain', 'colorful']
TASKS = ['MANAGE', 'VISIT', 'RESOURCE']
N_TESTS_PER_MODEL = len(PERTURBATIONS) * len(TASKS)
BONFERRONI_THRESHOLD = 0.05 / N_TESTS_PER_MODEL


def per_case_jsd_row(case, base=2.0):
    row = {'context_id': case.get('context_id')}
    for pert in PERTURBATIONS:
        for task in TASKS:
            orig_key = f"original_{task}"
            pert_key = f"{pert}_{task}"
            if orig_key not in case or pert_key not in case:
                row[f"{pert}_{task}"] = None
                continue
            orig = [x for x in case[orig_key]['binary_answers'] if x in (0, 1)]
            pertb = [x for x in case[pert_key]['binary_answers'] if x in (0, 1)]
            if not orig or not pertb:
                row[f"{pert}_{task}"] = None
                continue
            p = sum(orig) / len(orig)
            q = sum(pertb) / len(pertb)
            row[f"{pert}_{task}"] = jsd_bernoulli(p, q, base=base)
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-consistency baseline analysis for MedPerturb main experiment"
    )
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to main_evaluation_sc_<model>.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output .xlsx')
    parser.add_argument('--n_permutations', type=int, default=10000)
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--base', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.evaluation) as f:
        cases = json.load(f)
    rng = np.random.default_rng(args.rng_seed)

    drop_log = []
    summary_rows = []
    for pert in PERTURBATIONS:
        for task in TASKS:
            cell = permutation_test_cell(
                cases, pert, task,
                n_perms=args.n_permutations,
                rng=rng,
                base=args.base,
                drop_log=drop_log,
            )
            summary_rows.append({
                'perturbation': pert,
                'task': task,
                'n_cases_used': cell['n_cases_used'],
                'n_active': cell['n_active'],
                'mean_jsd_observed': cell['observed_mean_jsd'],
                'mean_jsd_null': cell['null_mean_jsd'],
                'null_std_jsd': cell['null_std_jsd'],
                'jsd_excess': cell['jsd_excess'],
                'p_value': cell['p_value'],
                'bonferroni_threshold': BONFERRONI_THRESHOLD,
                'significant': (cell['p_value'] < BONFERRONI_THRESHOLD
                                if not np.isnan(cell['p_value']) else False),
                'n_dropped_missing_key': cell['n_dropped_missing_key'],
                'n_dropped_empty': cell['n_dropped_empty'],
                'n_dropped_low_count': cell['n_dropped_low_count'],
            })

    per_case_rows = [per_case_jsd_row(c, base=args.base) for c in cases]

    summary_df = pd.DataFrame(summary_rows)
    per_case_df = pd.DataFrame(per_case_rows)
    drop_df = pd.DataFrame(drop_log, columns=['context_id', 'perturbation', 'task', 'reason'])

    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        per_case_df.to_excel(writer, sheet_name='PerCaseJSD', index=False)
        drop_df.to_excel(writer, sheet_name='DroppedCases', index=False)

    n_sig = int(summary_df['significant'].sum())
    print(f"Wrote {args.output}: {len(summary_df)} cells, {n_sig} significant after Bonferroni.")


if __name__ == "__main__":
    main()
