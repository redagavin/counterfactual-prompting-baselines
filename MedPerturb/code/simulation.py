# ABOUTME: Monte Carlo power simulation for all 5 MedPerturb metrics.
# ABOUTME: Uses real experiment logits to ground the generative model.

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid

from metrics import mi, phi, flip_rate, jsd, kl


ALL_METRICS = ["mi", "phi", "flip_rate", "jsd", "kl"]


def generate_responses(
    z_i: np.ndarray,
    y_orig: np.ndarray,
    sigma_pert: float,
    sigma: float,
    rng: np.random.Generator,
) -> dict:
    """Generate simulated responses from real logits.

    For each case i with real logit z_i:
        epsilon_pert_i  ~ N(0, sigma_pert^2)
        epsilon_noise_i ~ N(0, sigma^2)       (perturbation arm)
        epsilon_noise_i'~ N(0, sigma^2)       (baseline arm, independent)

        logit_pert_i = z_i + epsilon_pert_i + epsilon_noise_i
        logit_base_i = z_i + epsilon_noise_i'

        p_pert = sigmoid(logit_pert), p_base = sigmoid(logit_base)
        y_pert ~ Bernoulli(p_pert), y_base ~ Bernoulli(p_base)

    Returns dict with binary vectors and probability vectors.
    """
    n = len(z_i)

    epsilon_pert = rng.normal(0, sigma_pert, size=n) if sigma_pert > 0 else 0.0
    epsilon_noise_pert = rng.normal(0, sigma, size=n) if sigma > 0 else 0.0
    epsilon_noise_base = rng.normal(0, sigma, size=n) if sigma > 0 else 0.0

    logit_pert = z_i + epsilon_pert + epsilon_noise_pert
    logit_base = z_i + epsilon_noise_base

    p_orig = sigmoid(z_i)
    p_pert = sigmoid(logit_pert)
    p_base = sigmoid(logit_base)

    pert_bin = rng.binomial(1, p_pert)
    base_bin = rng.binomial(1, p_base)

    return {
        "orig": y_orig.copy(),
        "pert": pert_bin,
        "base": base_bin,
        "p_orig": p_orig,
        "p_pert": p_pert,
        "p_base": p_base,
    }


def run_single_simulation(
    data: dict,
    n_bootstrap: int = 1000,
    rng: np.random.Generator = None,
) -> dict:
    """Run all 5 metric tests on one simulated dataset.

    Per-population metrics (MI, Phi, Flip Rate) use bootstrap tests on binary vectors.
    Per-sample metrics (JSD, KL) use paired t-tests on probability vectors.

    Returns dict mapping metric name to p-value.
    """
    if rng is None:
        rng = np.random.default_rng()

    orig_bin = np.asarray(data["orig"])
    pert_bin = np.asarray(data["pert"])
    base_bin = np.asarray(data["base"])

    p_orig = np.asarray(data["p_orig"])
    p_pert = np.asarray(data["p_pert"])
    p_base = np.asarray(data["p_base"])

    boot_seed = int(rng.integers(0, 2**31))

    # Per-population metrics: bootstrap tests on binary vectors.
    # Direction per metric: FR (greater = more flips = more effect),
    # MI / phi (less = lower agreement = more effect).
    mi_result = mi.bootstrap_test(orig_bin, pert_bin, base_bin,
                                  n_bootstrap=n_bootstrap, seed=boot_seed,
                                  alternative='less')
    phi_result = phi.bootstrap_test(orig_bin, pert_bin, base_bin,
                                    n_bootstrap=n_bootstrap, seed=boot_seed,
                                    alternative='less')
    flip_result = flip_rate.bootstrap_test(orig_bin, pert_bin, base_bin,
                                           n_bootstrap=n_bootstrap, seed=boot_seed,
                                           alternative='greater')

    # Per-sample metrics: paired t-tests on probability vectors.
    # JSD / KL: greater = more divergence = more effect.
    jsd_result = jsd.paired_ttest(p_orig, p_pert, p_base, alternative='greater')
    kl_result = kl.paired_ttest(p_orig, p_pert, p_base, alternative='greater')

    return {
        "mi": mi_result["p_value"],
        "phi": phi_result["p_value"],
        "flip_rate": flip_result["p_value"],
        "jsd": jsd_result["p_value"],
        "kl": kl_result["p_value"],
    }


def _combo_seed(global_seed, sigma_pert, sigma, condition):
    """Derive a deterministic seed from parameters.

    Uses formatted float strings (:.6f) to ensure floating-point representation
    artifacts don't produce different seeds. Precision matches round(..., 6) in main().
    """
    import hashlib
    key = f"{global_seed}:{sigma_pert:.6f}:{sigma:.6f}:{condition}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _run_one_combo(args):
    """Run all simulations for one (sigma_pert, sigma) combo using real logits.

    Designed as a top-level function for multiprocessing.Pool.
    """
    sigma_pert, sigma, condition, n_simulations, n_bootstrap, global_seed, z_i, y_orig = args

    combo_seed = _combo_seed(global_seed, sigma_pert, sigma, condition)
    base_rng = np.random.default_rng(combo_seed)

    p_values_by_metric = {m: [] for m in ALL_METRICS}

    for _ in range(n_simulations):
        sim_seed = base_rng.integers(0, 2**31)
        sim_rng = np.random.default_rng(sim_seed)

        data = generate_responses(z_i, y_orig, sigma_pert, sigma, sim_rng)
        boot_rng = np.random.default_rng(sim_rng.integers(0, 2**31))
        pvals = run_single_simulation(data, n_bootstrap=n_bootstrap, rng=boot_rng)
        for m in ALL_METRICS:
            p_values_by_metric[m].append(pvals[m])

    results = []
    for m in ALL_METRICS:
        pv = np.array(p_values_by_metric[m])
        results.append({
            "metric": m,
            "sigma_pert": sigma_pert,
            "sigma": sigma,
            "condition": condition,
            "detection_rate": float(np.mean(pv < 0.05)),
            "mean_p_value": float(np.mean(pv)),
        })
    return results


def run_power_analysis(
    z_i: np.ndarray,
    y_orig: np.ndarray,
    condition: str,
    sigma_pert_values: list[float],
    sigma_values: list[float],
    n_simulations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 42,
    n_workers: int = 1,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulation across (sigma_pert, sigma) grid for one condition.

    Uses real logits z_i and binary answers y_orig. Parallelizes across
    parameter combos using n_workers processes. Checkpoints to CSV.

    Returns DataFrame with columns: metric, sigma_pert, sigma, condition,
    detection_rate, mean_p_value.
    """
    import multiprocessing

    completed = set()
    all_results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            existing = pd.read_csv(checkpoint_path)
            all_results = existing.to_dict('records')
            for _, row in existing.iterrows():
                completed.add((row['sigma_pert'], row['sigma'], row['metric']))
            n_done = len(completed) // len(ALL_METRICS)
            print(f"  Resuming from checkpoint: {n_done} combos already done", flush=True)
        except (pd.errors.EmptyDataError, KeyError, UnicodeDecodeError):
            completed = set()
            all_results = []
            print("  Warning: checkpoint file corrupted or empty, starting fresh", flush=True)

    work_items = []
    for sigma_pert in sigma_pert_values:
        for sigma in sigma_values:
            if (sigma_pert, sigma, ALL_METRICS[0]) in completed:
                continue
            work_items.append((
                sigma_pert, sigma, condition,
                n_simulations, n_bootstrap, seed, z_i, y_orig,
            ))

    total = len(work_items)
    print(f"  {total} combos remaining, using {n_workers} workers", flush=True)

    if total == 0:
        return pd.DataFrame(all_results)

    done = 0
    if n_workers <= 1:
        for item in work_items:
            combo_results = _run_one_combo(item)
            all_results.extend(combo_results)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", flush=True)
                if checkpoint_path:
                    pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
    else:
        with multiprocessing.Pool(n_workers) as pool:
            for combo_results in pool.imap_unordered(_run_one_combo, work_items):
                all_results.extend(combo_results)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", flush=True)
                    if checkpoint_path:
                        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    if checkpoint_path:
        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_results)


def generate_power_curves(results: pd.DataFrame, output_dir: str, condition: str) -> None:
    """Generate power curve figure for one condition.

    Layout: one subplot per sigma value, 5 metric curves per subplot.
    """
    sigma_values = sorted(results["sigma"].unique())
    n_cols = min(len(sigma_values), 2)
    n_rows = (len(sigma_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    metric_colors = {
        "jsd": "#1f77b4", "kl": "#ff7f0e",
        "mi": "#2ca02c", "phi": "#d62728", "flip_rate": "#9467bd",
    }

    for idx, sigma in enumerate(sigma_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        subset = results[results["sigma"] == sigma]

        for metric in ["jsd", "kl", "mi", "phi", "flip_rate"]:
            m_data = subset[subset["metric"] == metric].sort_values("sigma_pert")
            ax.plot(
                m_data["sigma_pert"], m_data["detection_rate"],
                marker='o', markersize=3, color=metric_colors[metric],
                label=metric.upper() if metric != "flip_rate" else "Flip Rate",
            )

        ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0.80, color='lightgray', linestyle=':', linewidth=1)
        ax.set_xlabel(r'$\sigma_{pert}$')
        ax.set_ylabel('Detection rate')
        ax.set_title(f'{condition} ($\\sigma$={sigma})')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    for idx in range(len(sigma_values), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'power_curves_{condition}.png'), dpi=300)
    plt.close(fig)


CONDITIONS = [
    ("MANAGE", "8b", "results/main_evaluation_llama_3.1_8b_instruct.json"),
    ("VISIT", "8b", "results/main_evaluation_llama_3.1_8b_instruct.json"),
    ("RESOURCE", "8b", "results/main_evaluation_llama_3.1_8b_instruct.json"),
    ("MANAGE", "70b", "results/main_evaluation_llama_3.1_70b_instruct.json"),
    ("VISIT", "70b", "results/main_evaluation_llama_3.1_70b_instruct.json"),
    ("RESOURCE", "70b", "results/main_evaluation_llama_3.1_70b_instruct.json"),
]


def main():
    import argparse
    from load_experiment_logits import load_condition

    parser = argparse.ArgumentParser(
        description="Monte Carlo power simulation v2 — empirically grounded"
    )
    parser.add_argument('--sigma-pert-max', type=float, default=3.0,
                        help='Maximum sigma_pert to sweep')
    parser.add_argument('--sigma-pert-step', type=float, default=0.1,
                        help='Step size for sigma_pert sweep')
    parser.add_argument('--sigma-values', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 1.0],
                        help='Baseline noise levels')
    parser.add_argument('--n-simulations', type=int, default=1000)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='results/simulation_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_steps = int(round(args.sigma_pert_max / args.sigma_pert_step)) + 1
    sigma_pert_values = [round(i * args.sigma_pert_step, 6) for i in range(n_steps)]

    print("Running power analysis v2 (empirically grounded):", flush=True)
    print(f"  sigma_pert: 0 to {args.sigma_pert_max} (step {args.sigma_pert_step},"
          f" {len(sigma_pert_values)} values)", flush=True)
    print(f"  sigma: {args.sigma_values}", flush=True)
    print(f"  {args.n_simulations} simulations, {args.n_bootstrap} bootstrap", flush=True)
    print(f"  Workers: {args.n_workers}", flush=True)
    print(f"  Conditions: {len(CONDITIONS)}", flush=True)

    for question, model_short, json_path in CONDITIONS:
        condition = f"{question}_{model_short}"
        print(f"\n{'='*50}", flush=True)
        print(f"Condition: {condition}", flush=True)
        print(f"{'='*50}", flush=True)

        data = load_condition(json_path, question)
        print(f"  Loaded {len(data['z_i'])} cases, "
              f"z_i range: [{data['z_i'].min():.1f}, {data['z_i'].max():.1f}]", flush=True)

        checkpoint = os.path.join(args.output_dir, f'simulation_v2_{condition}.csv')
        results = run_power_analysis(
            z_i=data["z_i"], y_orig=data["y_orig"], condition=condition,
            sigma_pert_values=sigma_pert_values, sigma_values=args.sigma_values,
            n_simulations=args.n_simulations, n_bootstrap=args.n_bootstrap,
            seed=args.seed, n_workers=args.n_workers,
            checkpoint_path=checkpoint,
        )

        results.to_csv(checkpoint, index=False)
        print(f"  Results saved to: {checkpoint}", flush=True)

        generate_power_curves(results, args.output_dir, condition)
        print(f"  Power curves saved", flush=True)

    print(f"\nAll conditions complete.", flush=True)


if __name__ == '__main__':
    main()
