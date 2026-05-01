#!/usr/bin/env python3
# ABOUTME: MedQA JSD Controlled Perturbation Analysis
# ABOUTME: Compares gender JSD to token-matched benign JSD for bias detection

import os
import time
import argparse
import pickle
import fcntl
import numpy as np
import pandas as pd

from medqa_jsd_analysis import (
    MedQAJSDAnalyzer,
    mark_gpu_completion,
    check_all_gpus_complete,
    try_acquire_merge_lock
)
from merge_results import merge_parallel_results
from token_edit_distance import token_edit_distance_percent
from jsd_utils import calculate_jsd
from calibrated_paraphrase import generate_calibrated_paraphrase
from controlled_stats import (
    compute_paired_ttest,
    compute_wilcoxon_test,
    compute_cohens_d_paired,
    compute_flip_overlap_metrics,
    compute_mcnemar_test
)


class MedQAJSDControlledAnalyzer(MedQAJSDAnalyzer):
    """
    Analyzer for controlled perturbation JSD comparison.

    Extends MedQAJSDAnalyzer to add:
    - Token-matched benign perturbations
    - Calibrated paraphrasing with retries
    - Paired statistical analysis
    """

    def __init__(self, args):
        """Initialize the controlled analyzer."""
        super().__init__(args)

        # Controlled analysis specific settings
        self.max_retries = getattr(args, 'max_retries', 50)
        self.tolerance = getattr(args, 'tolerance', 0.5)

    def generate_controlled_benign(self, question, target_pct):
        """
        Generate a benign perturbation matching the target token change percentage.

        Args:
            question: Original question text
            target_pct: Target token change percentage to match

        Returns:
            dict: Result from generate_calibrated_paraphrase
        """
        model_name = self.args.model
        tokenizer = self.tokenizers[model_name]

        return generate_calibrated_paraphrase(
            question=question,
            target_pct=target_pct,
            tokenizer=tokenizer,
            openai_client=self.openai_client,
            max_retries=self.max_retries,
            tolerance=self.tolerance
        )

    def compute_statistics(self, results):
        """
        Compute aggregate statistics from results.

        Args:
            results: List of result dictionaries

        Returns:
            dict: Aggregate statistics
        """
        jsd_gender = [r['jsd_gender'] for r in results if r.get('jsd_gender') is not None]
        jsd_benign = [r['jsd_benign'] for r in results if r.get('jsd_benign') is not None]

        if not jsd_gender or not jsd_benign:
            return {}

        # Ensure paired data has same length
        if len(jsd_gender) != len(jsd_benign):
            print(f"Warning: JSD arrays have different lengths ({len(jsd_gender)} vs {len(jsd_benign)}). Using minimum length.")
            min_len = min(len(jsd_gender), len(jsd_benign))
            jsd_gender = jsd_gender[:min_len]
            jsd_benign = jsd_benign[:min_len]

        # Descriptive statistics
        stats = {
            'n_cases': len(results),
            'mean_jsd_gender': np.mean(jsd_gender),
            'mean_jsd_benign': np.mean(jsd_benign),
            'mean_jsd_diff': np.mean(np.array(jsd_gender) - np.array(jsd_benign)),
            'median_jsd_gender': np.median(jsd_gender),
            'median_jsd_benign': np.median(jsd_benign),
            'std_jsd_gender': np.std(jsd_gender),
            'std_jsd_benign': np.std(jsd_benign),
        }

        # Paired tests
        ttest_result = compute_paired_ttest(jsd_gender, jsd_benign)
        stats['ttest_statistic'] = ttest_result['statistic']
        stats['ttest_p_value'] = ttest_result['p_value']                            # legacy: two-sided (unchanged)
        stats['ttest_p_value_two_sided'] = ttest_result['p_value_two_sided']
        stats['ttest_p_value_one_sided'] = ttest_result['p_value_one_sided']

        wilcoxon_result = compute_wilcoxon_test(jsd_gender, jsd_benign)
        stats['wilcoxon_statistic'] = wilcoxon_result['statistic']
        stats['wilcoxon_p_value'] = wilcoxon_result['p_value']

        # Effect size
        stats['cohens_d'] = compute_cohens_d_paired(jsd_gender, jsd_benign)

        # Flip overlap analysis
        gender_flips = [not r.get('gender_match', True) for r in results]
        benign_flips = [not r.get('benign_match', True) for r in results]

        overlap_metrics = compute_flip_overlap_metrics(gender_flips, benign_flips)
        stats.update({f'flip_{k}': v for k, v in overlap_metrics.items()})

        mcnemar_result = compute_mcnemar_test(gender_flips, benign_flips)
        stats['mcnemar_statistic'] = mcnemar_result['statistic']
        stats['mcnemar_p_value'] = mcnemar_result['p_value']

        # Calibration statistics
        deviations = [r.get('benign_deviation', 0) for r in results]
        retries = [r.get('benign_retries_used', 0) for r in results]
        stats['mean_benign_deviation'] = np.mean(deviations)
        stats['mean_retries_used'] = np.mean(retries)

        return stats

    def build_controlled_result_dict(self, case_idx, case,
                                      orig_response, orig_probs, orig_logits,
                                      gender_response, gender_probs, gender_logits,
                                      benign_response, benign_probs, benign_logits,
                                      benign_question,
                                      gender_token_pct, benign_target_pct,
                                      benign_actual_pct, benign_deviation,
                                      benign_retries, benign_all_attempts,
                                      jsd_gender, jsd_benign):
        """Build result dictionary for a processed case."""
        orig_answer = max(orig_probs, key=orig_probs.get)
        gender_answer = max(gender_probs, key=gender_probs.get)
        benign_answer = max(benign_probs, key=benign_probs.get)

        return {
            'case_idx': case_idx,
            'case_id': case['id'],
            'question': case['question'],
            'swapped_question': case['swapped_question'],
            'benign_question': benign_question,
            'option_a': case['option_a'],
            'option_b': case['option_b'],
            'option_c': case['option_c'],
            'option_d': case['option_d'],
            'ground_truth': case['ground_truth'],
            'original_gender': case['original_gender'],
            # Token change tracking
            'gender_token_change_pct': gender_token_pct,
            'benign_target_pct': benign_target_pct,
            'benign_actual_pct': benign_actual_pct,
            'benign_deviation': benign_deviation,
            'benign_retries_used': benign_retries,
            'benign_all_attempts': benign_all_attempts,
            # Model outputs
            'original_response': orig_response,
            'gender_response': gender_response,
            'benign_response': benign_response,
            'original_probs': orig_probs,
            'gender_probs': gender_probs,
            'benign_probs': benign_probs,
            'original_logits': orig_logits,
            'gender_logits': gender_logits,
            'benign_logits': benign_logits,
            # JSD metrics
            'jsd_gender': jsd_gender,
            'jsd_benign': jsd_benign,
            'jsd_difference': jsd_gender - jsd_benign,
            'jsd_ratio': jsd_gender / jsd_benign if jsd_benign > 0 else float('inf'),
            # Answer tracking
            'original_answer': orig_answer,
            'gender_answer': gender_answer,
            'benign_answer': benign_answer,
            'gender_match': (orig_answer == gender_answer),
            'benign_match': (orig_answer == benign_answer),
        }

    def process_cases_controlled(self):
        """
        Main processing loop for controlled perturbation analysis.

        For each case:
        1. Generate original response and extract logits
        2. Generate gender-swapped response and extract logits
        3. Measure gender token change percentage
        4. Generate calibrated benign perturbation matching that percentage
        5. Generate benign response and extract logits
        6. Calculate JSDs
        7. Save checkpoint periodically
        """
        model_name = self.args.model
        checkpoint_path = self.get_checkpoint_path().replace('.pkl', '_controlled.pkl')

        # Check for existing checkpoint
        results = []
        logits_data = {'metadata': {'model': model_name, 'type': 'controlled'}, 'samples': []}
        completed_indices = set()

        if os.path.exists(checkpoint_path):
            print(f"Found existing checkpoint: {checkpoint_path}")
            results, logits_data, completed_indices = self.load_checkpoint(checkpoint_path)
            print(f"Resuming from {len(completed_indices)} completed samples")

        total_cases = len(self.filtered_cases)
        print(f"\nProcessing {total_cases} cases with controlled perturbations...")

        for i, case in enumerate(self.filtered_cases):
            if i in completed_indices:
                continue

            print(f"\n[{i+1}/{total_cases}] Processing case {case['id']}...")

            # Format prompts
            orig_prompt = self.format_prompt(
                case['question'], case['option_a'], case['option_b'],
                case['option_c'], case['option_d']
            )
            gender_prompt = self.format_prompt(
                case['swapped_question'], case['option_a'], case['option_b'],
                case['option_c'], case['option_d']
            )

            # Generate original response
            print("  Generating original response...")
            orig_response, orig_logits, orig_probs = self.generate_and_extract_logits(orig_prompt)

            # Generate gender-swapped response
            print("  Generating gender-swapped response...")
            gender_response, gender_logits, gender_probs = self.generate_and_extract_logits(gender_prompt)

            # Measure gender token change percentage
            tokenizer = self.tokenizers[model_name]
            gender_token_pct = token_edit_distance_percent(
                case['question'], case['swapped_question'], tokenizer
            )
            print(f"  Gender token change: {gender_token_pct:.2f}%")

            # Generate calibrated benign perturbation
            print(f"  Generating calibrated benign (target: {gender_token_pct:.2f}%)...")
            benign_result = self.generate_controlled_benign(case['question'], gender_token_pct)
            benign_question = benign_result['paraphrase']
            print(f"  Benign actual: {benign_result['actual_pct']:.2f}% (deviation: {benign_result['deviation']:.2f}%, retries: {benign_result['retries_used']})")

            # Generate benign response
            benign_prompt = self.format_prompt(
                benign_question, case['option_a'], case['option_b'],
                case['option_c'], case['option_d']
            )
            print("  Generating benign response...")
            benign_response, benign_logits, benign_probs = self.generate_and_extract_logits(benign_prompt)

            # Calculate JSDs
            jsd_gender = calculate_jsd(orig_probs, gender_probs)
            jsd_benign = calculate_jsd(orig_probs, benign_probs)
            print(f"  JSD gender: {jsd_gender:.6f}, JSD benign: {jsd_benign:.6f}")

            # Build result
            result = self.build_controlled_result_dict(
                case_idx=i, case=case,
                orig_response=orig_response, orig_probs=orig_probs, orig_logits=orig_logits,
                gender_response=gender_response, gender_probs=gender_probs, gender_logits=gender_logits,
                benign_response=benign_response, benign_probs=benign_probs, benign_logits=benign_logits,
                benign_question=benign_question,
                gender_token_pct=gender_token_pct,
                benign_target_pct=gender_token_pct,
                benign_actual_pct=benign_result['actual_pct'],
                benign_deviation=benign_result['deviation'],
                benign_retries=benign_result['retries_used'],
                benign_all_attempts=benign_result['all_attempts'],
                jsd_gender=jsd_gender,
                jsd_benign=jsd_benign
            )
            results.append(result)

            # Build logits sample (full logits stored separately)
            logits_sample = {
                'case_id': case['id'],
                'case_idx': i,
                'original_logits': orig_logits,
                'gender_logits': gender_logits,
                'benign_logits': benign_logits
            }
            logits_data['samples'].append(logits_sample)

            # Checkpoint
            if (i + 1) % self.args.checkpoint_freq == 0:
                print(f"  Saving checkpoint at {i+1} samples...")
                metadata = {
                    'model_name': model_name,
                    'gpu_id': self.args.gpu_id,
                    'total_gpus': self.args.total_gpus,
                    'last_completed_idx': i,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.save_checkpoint(checkpoint_path, results, logits_data, metadata)

        # Final save
        print(f"\nSaving final results...")
        metadata = {
            'model_name': model_name,
            'gpu_id': self.args.gpu_id,
            'total_gpus': self.args.total_gpus,
            'last_completed_idx': len(self.filtered_cases) - 1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_checkpoint(checkpoint_path, results, logits_data, metadata)

        self.results = results
        self.logits_data = logits_data

        print(f"\n✓ Processing complete: {len(results)} cases processed")
        return results, logits_data

    def save_controlled_results(self, output_path, results, stats):
        """
        Save results to Excel with multiple sheets.

        Args:
            output_path: Path for output xlsx file
            results: List of result dictionaries
            stats: Dictionary with aggregate statistics
        """
        # Prepare results DataFrame (exclude large fields)
        df_data = []
        for r in results:
            row = {k: v for k, v in r.items()
                   if k not in ['benign_all_attempts', 'original_probs', 'gender_probs',
                               'benign_probs', 'original_logits', 'gender_logits', 'benign_logits']}
            df_data.append(row)
        df_results = pd.DataFrame(df_data)

        # Summary statistics
        df_stats = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in stats.items()])

        # Calibration details
        calibration_data = []
        for r in results:
            calibration_data.append({
                'case_id': r['case_id'],
                'target_pct': r['benign_target_pct'],
                'actual_pct': r['benign_actual_pct'],
                'deviation': r['benign_deviation'],
                'retries_used': r['benign_retries_used']
            })
        df_calibration = pd.DataFrame(calibration_data)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Analysis Results', index=False)
            df_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
            df_calibration.to_excel(writer, sheet_name='Calibration Details', index=False)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedQA JSD Controlled Analysis - Token-Matched Perturbation Comparison"
    )

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['deepseek_r1_0528', 'deepseek_r1_70b'],
                        help='Model to run analysis with')

    # Parallel execution
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU ID for this process')
    parser.add_argument('--total_gpus', type=int, default=1,
                        help='Total number of GPUs running in parallel')

    # Controlled perturbation settings
    parser.add_argument('--max_retries', type=int, default=50,
                        help='Maximum retries for calibrated paraphrasing')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='Tolerance for token change percentage matching')

    # Checkpointing
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/medqa_jsd_controlled',
                        help='Directory for checkpoints')

    # Output
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files')

    # Testing
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit number of samples for testing')

    args = parser.parse_args()

    # Auto-detect SLURM array job parameters
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.gpu_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.total_gpus = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0

    # Add n_benign for parent class compatibility (not used but required)
    args.n_benign = 1

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis
    analyzer = MedQAJSDControlledAnalyzer(args)
    analyzer.validate_gpu_assignment(args.gpu_id)
    analyzer.load_data()
    analyzer.filter_and_prepare_cases()
    analyzer.load_model()

    results, logits_data = analyzer.process_cases_controlled()

    # Automatic merge detection (parallel execution only)
    if args.total_gpus > 1:
        model_name = args.model

        # Mark this GPU as complete
        mark_gpu_completion(args.checkpoint_dir, model_name, args.gpu_id, args.total_gpus)
        print(f"\n✓ GPU {args.gpu_id} marked as complete")

        # Try to acquire merge lock (non-blocking)
        lock_file = try_acquire_merge_lock(args.checkpoint_dir, model_name)

        if lock_file:
            try:
                # We got the lock - check if all GPUs are done
                if check_all_gpus_complete(args.checkpoint_dir, model_name, args.total_gpus):
                    print(f"\n{'='*60}")
                    print(f"All {args.total_gpus} GPUs complete - triggering merge")
                    print(f"{'='*60}\n")

                    # Merge results with _controlled suffix
                    merged_results, merged_logits = merge_parallel_results(
                        model_name,
                        args.total_gpus,
                        args.checkpoint_dir,
                        args.output_dir,
                        suffix='_controlled'
                    )

                    print(f"\n✓ Merge complete: {len(merged_results)} total results")

                    # Compute aggregate statistics on merged results
                    stats = analyzer.compute_statistics(merged_results)

                    if stats:
                        # Save merged results to Excel
                        output_xlsx = f"{args.output_dir}/medqa_jsd_controlled_analysis_{model_name}.xlsx"
                        analyzer.save_controlled_results(output_xlsx, merged_results, stats)

                        # Save merged logits
                        logits_path = f"{args.output_dir}/medqa_jsd_controlled_logits_{model_name}.pkl"
                        with open(logits_path, 'wb') as f:
                            pickle.dump(merged_logits, f)
                        print(f"Logits saved to: {logits_path}")

                        # Print summary
                        print(f"\n{'='*60}")
                        print("Summary Statistics (Merged)")
                        print(f"{'='*60}")
                        for k, v in stats.items():
                            if isinstance(v, float):
                                print(f"  {k}: {v:.6f}")
                            else:
                                print(f"  {k}: {v}")
                    else:
                        print("\n⚠ No valid results found in merged data")
                else:
                    print(f"\n⏳ Waiting for other GPUs to complete...")
            finally:
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
        else:
            print(f"\n⏳ Another GPU is performing merge...")
    else:
        # Single GPU mode - just save results directly
        stats = analyzer.compute_statistics(results)

        output_xlsx = f"{args.output_dir}/medqa_jsd_controlled_analysis_{args.model}.xlsx"
        analyzer.save_controlled_results(output_xlsx, results, stats)

        logits_path = f"{args.output_dir}/medqa_jsd_controlled_logits_{args.model}.pkl"
        with open(logits_path, 'wb') as f:
            pickle.dump(logits_data, f)
        print(f"Logits saved to: {logits_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
