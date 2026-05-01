# ABOUTME: Paired t-tests with Bonferroni correction for Q-Pain replication
# ABOUTME: Compares 28 demographic swaps against calibrated paraphrase and fixed sentence baselines

import os

import numpy as np
import pandas as pd
from scipy import stats


N_COMPARISONS = 28
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS


def run_paired_ttest(group_a, group_b):
    """Paired two-tailed t-test on p(No) differences."""
    a = np.array(group_a)
    b = np.array(group_b)
    diffs = a - b
    n = len(diffs)
    mean_diff = np.mean(diffs)
    se_diff = np.std(diffs, ddof=1) / np.sqrt(n) if n > 1 else 0.0

    if se_diff == 0:
        return {
            "t_stat": 0.0, "p_value": 1.0, "mean_diff": float(mean_diff),
            "se_diff": se_diff, "n": n,
            "ci_95": (float(mean_diff), float(mean_diff)),
            "ci_bonferroni": (float(mean_diff), float(mean_diff)),
        }

    t_stat, p_value = stats.ttest_rel(a, b)

    df = n - 1
    t_crit_95 = stats.t.ppf(1 - 0.05 / 2, df)
    t_crit_bonf = stats.t.ppf(1 - BONFERRONI_ALPHA / 2, df)

    ci_95 = (float(mean_diff - t_crit_95 * se_diff), float(mean_diff + t_crit_95 * se_diff))
    ci_bonf = (float(mean_diff - t_crit_bonf * se_diff), float(mean_diff + t_crit_bonf * se_diff))

    return {
        "t_stat": float(t_stat), "p_value": float(p_value), "mean_diff": float(mean_diff),
        "se_diff": float(se_diff), "n": n,
        "ci_95": ci_95, "ci_bonferroni": ci_bonf,
    }


def run_all_comparisons(eval_results, comparisons):
    """Run 28 x 3 paired t-tests (demographic, paraphrase, fixed sentence)."""
    by_key = {}
    for r in eval_results:
        vid = r["vignette_id"]
        sg = r["subgroup"]
        vt = r["variant_type"]
        comp = r.get("comparison", "")
        by_key[(vid, sg, vt, comp)] = r

    vignette_ids = sorted(set(r["vignette_id"] for r in eval_results))
    results = []

    for comp in comparisons:
        orig = comp["original"]
        swap = comp["swapped"]
        label = f"{orig}_vs_{swap}"

        orig_vals, swap_vals = [], []
        para_orig_vals, para_vals = [], []
        fixed_orig_vals, fixed_vals = [], []

        for vid in vignette_ids:
            orig_r = by_key.get((vid, orig, "demographic", ""))
            swap_r = by_key.get((vid, swap, "demographic", ""))
            if orig_r and swap_r:
                orig_vals.append(orig_r["prob_no"])
                swap_vals.append(swap_r["prob_no"])

            fixed_r = by_key.get((vid, orig, "fixed_sentence", ""))
            orig_r_demo = by_key.get((vid, orig, "demographic", ""))
            if fixed_r and orig_r_demo:
                fixed_orig_vals.append(orig_r_demo["prob_no"])
                fixed_vals.append(fixed_r["prob_no"])

            para_r = by_key.get((vid, orig, "paraphrase", label))
            if para_r and orig_r_demo:
                para_orig_vals.append(orig_r_demo["prob_no"])
                para_vals.append(para_r["prob_no"])

        if len(orig_vals) >= 2:
            result = run_paired_ttest(orig_vals, swap_vals)
            result["comparison"] = label
            result["test_type"] = "demographic"
            results.append(result)

        if len(para_orig_vals) >= 2:
            result = run_paired_ttest(para_orig_vals, para_vals)
            result["comparison"] = label
            result["test_type"] = "paraphrase"
            results.append(result)

        if len(fixed_orig_vals) >= 2:
            result = run_paired_ttest(fixed_orig_vals, fixed_vals)
            result["comparison"] = label
            result["test_type"] = "fixed_sentence"
            results.append(result)

    return results


def export_to_excel(results, output_path):
    """Export analysis results to Excel with summary statistics."""
    if not results:
        return
    df = pd.DataFrame(results)

    if "ci_95" in df.columns:
        df["ci_95_lower"] = df["ci_95"].apply(lambda x: x[0])
        df["ci_95_upper"] = df["ci_95"].apply(lambda x: x[1])
        df["ci_bonf_lower"] = df["ci_bonferroni"].apply(lambda x: x[0])
        df["ci_bonf_upper"] = df["ci_bonferroni"].apply(lambda x: x[1])
        df = df.drop(columns=["ci_95", "ci_bonferroni"])

    df["sig_05"] = df["p_value"] < 0.05
    df["sig_bonferroni"] = df["p_value"] < BONFERRONI_ALPHA

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)

        summary_rows = []
        for test_type in ["demographic", "paraphrase", "fixed_sentence"]:
            sub = df[df["test_type"] == test_type]
            summary_rows.append({
                "test_type": test_type,
                "total_comparisons": len(sub),
                "sig_05_count": int(sub["sig_05"].sum()),
                "sig_bonferroni_count": int(sub["sig_bonferroni"].sum()),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
