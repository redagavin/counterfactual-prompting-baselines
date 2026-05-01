# ABOUTME: Regression analysis and held-out prediction for bias-in-bios experiment
# ABOUTME: Two models compared: difference (paired) and level (stacked with z_i covariate)

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_difference_data(eval_results, baseline_type):
    """Build paired difference data: Δ_i = logit_swapped - logit_baseline.

    One row per bio. Used by Model A (difference model).
    """
    logit_key = f"logit_{baseline_type}"
    rows = []
    for r in eval_results:
        rows.append({
            "bio_id": r["bio_id"],
            "delta": r["logit_swapped"] - r[logit_key],
            "swap_direction": r["swap_direction"],
        })
    return pd.DataFrame(rows)


def build_level_data(eval_results, baseline_type):
    """Stack baseline + perturbed rows for level regression.

    Two rows per bio. Used by Model B (level model with z_i).
    """
    logit_key = f"logit_{baseline_type}"
    rows = []
    for r in eval_results:
        z_i = r["logit_original"]
        rows.append({"bio_id": r["bio_id"], "y": r[logit_key], "z_i": z_i, "swap_direction": 0})
        rows.append({"bio_id": r["bio_id"], "y": r["logit_swapped"], "z_i": z_i, "swap_direction": r["swap_direction"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model A: Difference model (paired)
# ---------------------------------------------------------------------------

def run_difference_regression(df):
    """Model A: Δ_i = β_gender × swap_direction + ε.

    No intercept. One row per bio. Standard OLS (no clustering needed).
    One-sided test H₁: β_gender < 0.
    """
    X = df[["swap_direction"]]
    model = sm.OLS(df["delta"], X).fit()
    t_stat = model.tvalues["swap_direction"]
    p_one_sided = stats.t.cdf(t_stat, model.df_resid)
    return {
        "model": "A_difference",
        "beta_gender": model.params["swap_direction"],
        "se_beta_gender": model.bse["swap_direction"],
        "t_stat": t_stat,
        "p_value": model.pvalues["swap_direction"],
        "p_value_one_sided": p_one_sided,
        "r2": model.rsquared,
        "n_obs": int(model.nobs),
    }


def held_out_difference(train_df, test_df):
    """Model A held-out: predict Δ on test using β_gender from train."""
    X_train = train_df[["swap_direction"]]
    model = sm.OLS(train_df["delta"], X_train).fit()

    delta_test = test_df["delta"].values
    delta_pred = model.predict(test_df[["swap_direction"]])

    ss_total = np.sum((delta_test - delta_test.mean()) ** 2)
    ss_res = np.sum((delta_test - delta_pred) ** 2)

    return {
        "rmse": np.sqrt(np.mean((delta_test - delta_pred) ** 2)),
        "r2": 1 - ss_res / ss_total if ss_total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Model B: Level model (stacked, with z_i covariate)
# ---------------------------------------------------------------------------

def run_level_regression(df):
    """Model B: y = β₀ + β₁z_i + β_gender × swap_direction + ε.

    Clustered SEs by bio_id. One-sided test H₁: β_gender < 0.
    """
    X = sm.add_constant(df[["z_i", "swap_direction"]])
    model = sm.OLS(df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": df["bio_id"]})
    t_stat = model.tvalues["swap_direction"]
    p_one_sided = stats.t.cdf(t_stat, model.df_resid)
    return {
        "model": "B_level",
        "beta_0": model.params["const"],
        "beta_1": model.params["z_i"],
        "beta_gender": model.params["swap_direction"],
        "se_beta_gender": model.bse["swap_direction"],
        "t_stat": t_stat,
        "p_value": model.pvalues["swap_direction"],
        "p_value_one_sided": p_one_sided,
        "r2": model.rsquared,
        "n_obs": int(model.nobs),
    }


def held_out_level(train_df, test_df):
    """Model B held-out: predict y on test, compute ΔR² (full vs null)."""
    X_train = sm.add_constant(train_df[["z_i", "swap_direction"]])
    full_model = sm.OLS(train_df["y"], X_train).fit()

    X_train_null = sm.add_constant(train_df[["z_i"]])
    null_model = sm.OLS(train_df["y"], X_train_null).fit()

    X_test = sm.add_constant(test_df[["z_i", "swap_direction"]])
    X_test_null = sm.add_constant(test_df[["z_i"]])
    y_test = test_df["y"].values

    y_pred_full = full_model.predict(X_test)
    y_pred_null = null_model.predict(X_test_null)

    ss_total = np.sum((y_test - y_test.mean()) ** 2)
    ss_res_full = np.sum((y_test - y_pred_full) ** 2)
    ss_res_null = np.sum((y_test - y_pred_null) ** 2)

    r2_full = 1 - ss_res_full / ss_total
    r2_null = 1 - ss_res_null / ss_total

    return {
        "r2_full": r2_full,
        "r2_null": r2_null,
        "delta_r2": r2_full - r2_null,
        "rmse": np.sqrt(np.mean((y_test - y_pred_full) ** 2)),
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by existing tests)
# ---------------------------------------------------------------------------

build_regression_data = build_level_data
run_regression = run_level_regression
held_out_prediction = held_out_level


# ---------------------------------------------------------------------------
# End-to-end analysis runner
# ---------------------------------------------------------------------------

def load_eval_results(path):
    """Load evaluation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_full_analysis(train_results_path, test_results_path, output_path):
    """Run both regression models and held-out prediction for both baselines.

    Produces an Excel file with one row per (regression model × baseline),
    containing train/test regression results and held-out prediction metrics.
    """
    train_results = load_eval_results(train_results_path)
    test_results = load_eval_results(test_results_path)

    all_results = []
    for baseline_type in ["paraphrase", "fixed_sentence"]:
        # Model A: difference
        diff_train = build_difference_data(train_results, baseline_type)
        diff_test = build_difference_data(test_results, baseline_type)
        reg_a_train = run_difference_regression(diff_train)
        reg_a_test = run_difference_regression(diff_test)
        pred_a = held_out_difference(diff_train, diff_test)

        all_results.append({
            "regression": "A_difference",
            "baseline": baseline_type,
            "beta_gender_train": reg_a_train["beta_gender"],
            "p_value_train": reg_a_train["p_value_one_sided"],            # legacy (unchanged — already one-sided despite name)
            "p_value_one_sided_train": reg_a_train["p_value_one_sided"],
            "p_value_two_sided_train": reg_a_train["p_value"],
            "beta_gender_test": reg_a_test["beta_gender"],
            "p_value_test": reg_a_test["p_value_one_sided"],              # legacy (unchanged)
            "p_value_one_sided_test": reg_a_test["p_value_one_sided"],
            "p_value_two_sided_test": reg_a_test["p_value"],
            "rmse": pred_a["rmse"],
            "r2_full": None,
            "r2_null": None,
            "delta_r2": None,
        })

        # Model B: level
        level_train = build_level_data(train_results, baseline_type)
        level_test = build_level_data(test_results, baseline_type)
        reg_b_train = run_level_regression(level_train)
        reg_b_test = run_level_regression(level_test)
        pred_b = held_out_level(level_train, level_test)

        all_results.append({
            "regression": "B_level",
            "baseline": baseline_type,
            "beta_gender_train": reg_b_train["beta_gender"],
            "p_value_train": reg_b_train["p_value_one_sided"],            # legacy (unchanged — already one-sided despite name)
            "p_value_one_sided_train": reg_b_train["p_value_one_sided"],
            "p_value_two_sided_train": reg_b_train["p_value"],
            "beta_gender_test": reg_b_test["beta_gender"],
            "p_value_test": reg_b_test["p_value_one_sided"],              # legacy (unchanged)
            "p_value_one_sided_test": reg_b_test["p_value_one_sided"],
            "p_value_two_sided_test": reg_b_test["p_value"],
            "rmse": pred_b["rmse"],
            "r2_full": pred_b["r2_full"],
            "r2_null": pred_b["r2_null"],
            "delta_r2": pred_b["delta_r2"],
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(output_path, index=False)
    print(f"Analysis saved to {output_path}")
    return results_df
