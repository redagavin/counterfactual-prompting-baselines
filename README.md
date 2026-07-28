# Compared to What? Baselines and Metrics for Counterfactual Prompting

This repository contains code to reproduce experiments in the paper.

## Repository structure

```
.
├── src/                     # Core Python modules: data loading, paraphrase generation,
│                            # model evaluation, statistical analyses for each case study
├── MedPerturb/              # MedPerturb case study (derivative of upstream MedPerturb repo;
│                            # contains data CSVs, code/, scripts/, slurm/, case_studies/)
├── scripts/                 # Top-level driver shell scripts (auto_run_*.sh) and
│                            # post-hoc statistical scripts (compute_*.py)
├── slurm_jobs/              # SLURM sbatch files for top-level (non-MedPerturb) experiments
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

## Setup

- Python 3.10+ recommended (a conda environment such as `cot` works well).
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- HuggingFace authentication (required for gated Llama-3.1 models):

  ```bash
  huggingface-cli login
  ```

- OpenAI API key (required for paraphrase generation across all case studies):

  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

  Alternatively, place it in a `.env` file at the repo root; `python-dotenv` will pick it up.

## Data setup

- **Bias-in-Bios**: HuggingFace dataset `LabHC/bias_in_bios`, fetched automatically via `datasets.load_dataset` on first run. No manual download.
- **MedQA**: HuggingFace dataset `GBaker/MedQA-USMLE-4-options-hf`, fetched automatically.
- **Q-Pain**: PhysioNet credentialed access required.
  1. Register at https://physionet.org and complete the data use agreement.
  2. Download the Q-Pain v1.0.0 release into `physionet.org/files/q-pain/1.0.0/` relative to this repo's root.
  3. The Q-Pain scripts read CSVs from that path; without it, paraphrase and evaluation jobs will fail.
- **DiscrimEval**: HuggingFace dataset `Anthropic/discrim-eval` ("explicit" config), fetched automatically.
- **MedPerturb input CSVs**: shipped with this repo at `MedPerturb/data.csv` and `MedPerturb/data_with_baselines.csv`.

## Running experiments

All commands assume invocation from the **repository root**, except for `MedPerturb/slurm/*.sbatch`, which must be submitted from inside `MedPerturb/` (see *SLURM submission discipline* below).

### MedPerturb — main + sanity + precision + simulation + analysis

The umbrella driver runs all MedPerturb-core experiments and iterates over Llama-3.1 8B and 70B internally (no positional model argument):

```bash
bash MedPerturb/scripts/auto_run_all.sh
# or, for a smoke run:
bash MedPerturb/scripts/auto_run_all.sh --test
```

Note: this driver does **not** include the dose-response experiment (run it separately, below). The simulation is launched fire-and-forget; wait for it to finish before running `plot_power_curves.py` (below).

### MedPerturb — dose response (figures, no p-values)

```bash
bash MedPerturb/scripts/auto_run_dose_response.sh meta-llama/Llama-3.1-8B-Instruct
python MedPerturb/case_studies/dose_response_analysis.py \
    --evaluation <eval_json> \
    --output MedPerturb/results/dose_response.xlsx \
    --plot_prefix MedPerturb/results/dose_response
```

### MedPerturb — precision check

```bash
bash MedPerturb/scripts/auto_run_precision_plus1.sh meta-llama/Llama-3.1-8B-Instruct
```

### MedPerturb — individual experiments

Each per-experiment wrapper takes a positional model argument and runs paraphrase generation + model evaluation. The downstream `*_analysis.py` script (under `MedPerturb/case_studies/`) must be invoked manually after results are merged:

```bash
bash MedPerturb/scripts/auto_run_main_experiment.sh   meta-llama/Llama-3.1-8B-Instruct
bash MedPerturb/scripts/auto_run_sanity_check.sh      meta-llama/Llama-3.1-8B-Instruct

python MedPerturb/case_studies/<analysis_script>.py \
    --evaluation <merged_eval_json> \
    --output <out.xlsx>
```

### MedPerturb — power simulation

The simulation sbatch resolves its working directory from `SLURM_SUBMIT_DIR`, so you must submit it from inside `MedPerturb/`:

```bash
cd MedPerturb && sbatch slurm/run_simulation.sbatch
```

Regenerate the paper's power-curve figures and inline calibration summary:

```bash
python MedPerturb/scripts/plot_power_curves.py \
    --sim-dir MedPerturb/results/simulation_v2 \
    --output-dir <where-paper-figures-go>

python MedPerturb/scripts/sim_calibration_summary.py
```

(Defaults of `plot_power_curves.py` are CWD-relative; pass explicit paths if you invoke it from elsewhere.)

### MedPerturb — self-consistency control

Repeated sampling (10 samples at T=0.7) of unchanged prompts, measuring the model's own output stochasticity as a reference level. The sbatch is a 4-way data-parallel array; submit from inside `MedPerturb/` and merge shards before analysis:

```bash
cd MedPerturb && sbatch slurm/run_main_experiment_sc.sbatch meta-llama/Llama-3.1-8B-Instruct

python MedPerturb/case_studies/self_consistency_analysis.py \
    --evaluation <merged_sc_json> \
    --output MedPerturb/results/self_consistency.xlsx
```

`main_evaluate_sc.py` accepts `--temperature` (default 0.7); non-default temperatures tag the shard filenames (e.g., `_t1.5`).

### DiscrimEval

70 "Explicit" scenarios × 14 demographic contrasts, adjusted paraphrase baseline. The driver generates paraphrases once (OpenAI API), submits the eval as a SLURM array, merges shards, and runs the analysis:

```bash
bash scripts/auto_run_discrimeval.sh meta-llama/Llama-3.1-8B-Instruct 1
bash scripts/auto_run_discrimeval.sh meta-llama/Llama-3.1-70B-Instruct 4
```

Outputs land at `results/discrimeval/discrimeval_analysis_<model>.xlsx` (per-contrast p-values + per-axis survival summary; a contrast "survives" if any of the five metrics reaches Bonferroni significance). `scripts/validate_discrimeval_preflight.py` checks the paraphrase file before long runs.

### Bias-in-Bios

```bash
bash scripts/auto_run_bib.sh
python scripts/compute_bib_nondirectional_metrics.py
```

The post-hoc script writes `results/bib/bib_nondirectional_metrics.{xlsx,json}` with per-metric `*_p_one_sided` and `*_p_two_sided` columns.

### Q-Pain regression

```bash
bash scripts/auto_run_qpain.sh
```

Runs 3 models × 2 race comparisons × 2 baselines. Outputs land at `results/qpain/qpain_analysis_<model>_<comparison>.xlsx`.

### Q-Pain replication (GPT-2 Large)

```bash
bash scripts/auto_run_qpain_replication.sh
```

This pipeline is **two-tailed by design**, faithful to the original Q-Pain methodology.

### MedQA preliminary

The driver script runs the **8B** model only (it hardcodes `MODEL="deepseek_r1_0528"`):

```bash
bash scripts/auto_run_medqa_jsd_controlled.sh
```

For the 70B variant, invoke the analysis directly:

```bash
python src/medqa_jsd_controlled_analysis.py --model deepseek_r1_70b ...
```

(See the script for the full argument list.)

After evaluation, compute the flip-rate bootstrap p-values (paper Table 3):

```bash
python scripts/compute_medqa_flip_rate_bootstrap.py
```

### SLURM submission discipline

- Top-level `slurm_jobs/*.sbatch` and `scripts/auto_run_*.sh`: invoke from the **repo root**.
- `MedPerturb/slurm/*.sbatch`: invoke from inside **`MedPerturb/`** (or via the `MedPerturb/scripts/auto_run_*.sh` wrappers).

## P-value column conventions

Output spreadsheets include both `p_value_one_sided` and `p_value_two_sided` alongside a legacy `p_value` column. The paper's reported p-values correspond to the one-sided variant in most experiments, but the legacy column's semantics differ across files — consult the relevant analysis script to confirm which column matches the paper for a given experiment.

## Notes

- `src/bhcs_analysis.py` is a utility-only module (it provides `GENDER_MAPPING` and `with_timeout`). Running it directly exits with an error.
- `MedPerturb/README.md` is from the upstream MedPerturb repository (CC BY 4.0). It documents the dataset, not this project.
- All shell scripts and sbatches assume invocation from the repository root, with the single exception of `MedPerturb/slurm/*.sbatch` (which must be submitted from inside `MedPerturb/`).
- SLURM partition names hardcoded in shell scripts and sbatch files (e.g., `frink`, `gpu`, `177huntington`) are specific to the original cluster. Users running this on other systems must update `--partition=` flags to match their site's available partitions.

## License

This project is released under the MIT License (see `LICENSE`).

The `MedPerturb/` directory is a derivative of the upstream MedPerturb repository, which is distributed under CC BY 4.0; attribution to the original authors is retained.
