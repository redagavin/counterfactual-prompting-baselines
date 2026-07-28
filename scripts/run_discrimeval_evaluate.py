# ABOUTME: Driver for DiscrimEval log-odds evaluation with optional SLURM sharding
# ABOUTME: Supports data-parallel execution via SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from discrimeval_data import load_discrimeval_explicit, select_samples, merge_paraphrases_from_file
from discrimeval_evaluate import (DiscrimEvalEvaluator, run_evaluation,
                                  detect_slurm, shard_samples, model_short_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--paraphrases", required=True)
    ap.add_argument("--output_dir", default="results/discrimeval")
    ap.add_argument("--checkpoint_dir", default="checkpoints/discrimeval")
    ap.add_argument("--sample_size", type=int, default=None)
    args = ap.parse_args()

    samples = merge_paraphrases_from_file(select_samples(load_discrimeval_explicit()),
                                          args.paraphrases)
    if args.sample_size:
        samples = samples[:args.sample_size]

    gpu, total = detect_slurm()
    short = model_short_name(args.model)
    if gpu is not None:
        samples = shard_samples(samples, gpu, total)
        tag = f"{short}_gpu{gpu}_of_{total}"
    else:
        tag = short
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"discrimeval_eval_{tag}.json")
    ckpt = os.path.join(args.checkpoint_dir, f"discrimeval_eval_{tag}.pkl")
    run_evaluation(DiscrimEvalEvaluator(args.model), samples, out, ckpt)


if __name__ == "__main__":
    main()
