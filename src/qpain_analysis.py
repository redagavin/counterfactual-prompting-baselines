# ABOUTME: Regression analysis for Q-Pain race bias experiment
# ABOUTME: Runs difference and level regressions across two baselines, remapping gender->race

import json

import pandas as pd

from bib_analysis import (
    build_difference_data,
    build_level_data,
    run_difference_regression,
    run_level_regression,
)


def load_eval_results(path):
    """Load evaluation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_qpain_analysis(eval_results, output_path, comparison="black"):
    """Run both regression models for both baselines on Q-Pain eval results.

    Args:
        eval_results: list of eval result dicts
        output_path: Excel output path, or None to skip
        comparison: 'black' or 'asian' — selects which logit fields to use
    """
    if comparison == "asian":
        swapped_key = "logit_asian"
        paraphrase_key = "logit_asian_paraphrase"
        paraphrase_label = "asian_paraphrase"
    else:
        swapped_key = "logit_swapped"
        paraphrase_key = "logit_paraphrase"
        paraphrase_label = "paraphrase"

    # Remap fields for bib_analysis compatibility
    remapped = []
    for r in eval_results:
        entry = {
            "bio_id": r["bio_id"],
            "swap_direction": r["swap_direction"],
            "logit_original": r["logit_original"],
            "logit_swapped": r[swapped_key],
            "logit_fixed_sentence": r["logit_fixed_sentence"],
        }
        if paraphrase_key in r:
            entry["logit_paraphrase"] = r[paraphrase_key]
        remapped.append(entry)

    rows = []
    for baseline_type, baseline_label in [("paraphrase", paraphrase_label), ("fixed_sentence", "fixed_sentence")]:
        diff_df = build_difference_data(remapped, baseline_type)
        reg_a = run_difference_regression(diff_df)
        rows.append({
            "regression": reg_a["model"],
            "baseline": baseline_label,
            "beta_race": reg_a["beta_gender"],
            "se_beta_race": reg_a["se_beta_gender"],
            "t_stat": reg_a["t_stat"],
            "p_value": reg_a["p_value_one_sided"],
            "p_value_one_sided": reg_a["p_value_one_sided"],
            "p_value_two_sided": reg_a["p_value"],
            "r2": reg_a["r2"],
            "n_obs": reg_a["n_obs"],
        })

        level_df = build_level_data(remapped, baseline_type)
        reg_b = run_level_regression(level_df)
        rows.append({
            "regression": reg_b["model"],
            "baseline": baseline_label,
            "beta_race": reg_b["beta_gender"],
            "se_beta_race": reg_b["se_beta_gender"],
            "t_stat": reg_b["t_stat"],
            "p_value": reg_b["p_value_one_sided"],
            "p_value_one_sided": reg_b["p_value_one_sided"],
            "p_value_two_sided": reg_b["p_value"],
            "r2": reg_b["r2"],
            "n_obs": reg_b["n_obs"],
        })

    results_df = pd.DataFrame(rows)
    if output_path is not None:
        results_df.to_excel(output_path, index=False)
    return results_df
