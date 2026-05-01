# ABOUTME: Summarize false-positive rate at sigma_pert=0 across all conditions.
# ABOUTME: Prints mean and max false-positive rate for the paper calibration claim.

import os
import glob
import pandas as pd

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'simulation_v2')

frames = []
for path in sorted(glob.glob(os.path.join(SIM_DIR, 'simulation_v2_*.csv'))):
    df = pd.read_csv(path)
    frames.append(df)

if not frames:
    print('No simulation CSVs found yet.')
    raise SystemExit(1)

df = pd.concat(frames, ignore_index=True)
print(f'Loaded {len(frames)} CSVs, total rows: {len(df)}')

# Calibration: false positive rate at sigma_pert == 0, across all conditions, metrics, sigma > 0.
# Paper claim: "across all six conditions, metrics, and sigma > 0 values, the mean false
# positive rate at sigma_pert = 0 is X (nominal: 0.05), with no single test exceeding Y."
null_df = df[(df['sigma_pert'] == 0) & (df['sigma'] > 0)]
print(f'\nNull rows (sigma_pert=0, sigma>0): {len(null_df)}')
print(f'  Mean detection_rate: {null_df["detection_rate"].mean():.4f}')
print(f'  Max  detection_rate: {null_df["detection_rate"].max():.4f}')

# Breakdown by metric
print('\nBy metric (sigma_pert=0, sigma>0):')
for metric, g in null_df.groupby('metric'):
    print(f'  {metric:10s}  mean={g["detection_rate"].mean():.4f}  max={g["detection_rate"].max():.4f}  n={len(g)}')

# Breakdown by condition
print('\nBy condition (sigma_pert=0, sigma>0):')
for cond, g in null_df.groupby('condition'):
    print(f'  {cond:14s}  mean={g["detection_rate"].mean():.4f}  max={g["detection_rate"].max():.4f}  n={len(g)}')
