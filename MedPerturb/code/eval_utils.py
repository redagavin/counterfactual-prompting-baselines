# ABOUTME: Shared evaluation utilities for checkpoint, SLURM, sharding, and completion
# ABOUTME: Consolidates duplicated code from sanity_check_evaluate.py and precision_check_evaluate.py

import os
import pickle
import shutil
import tempfile
import time


def model_short_name(model_name):
    """Convert full model name to filename-safe short version.

    Example: 'meta-llama/Llama-3.1-8B-Instruct' -> 'llama_3.1_8b_instruct'
    """
    return model_name.split("/")[-1].lower().replace("-", "_")


def detect_slurm():
    """Detect SLURM array job parameters.

    Returns:
        (gpu_id, total_gpus) if in SLURM array job, (None, None) otherwise.
    """
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        gpu_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        total_gpus = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        return gpu_id, total_gpus
    return None, None


def shard_samples(samples, gpu_id, total_gpus):
    """Stride-based sharding for deterministic load balancing."""
    return samples[gpu_id::total_gpus]


def save_checkpoint(path, results, completed_ids):
    """Atomic pickle save with tempfile + shutil.move."""
    data = {"results": results, "completed_ids": list(completed_ids)}
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f)
        shutil.move(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_checkpoint(path):
    """Load checkpoint. Returns (results, completed_ids_set).

    Returns ([], set()) if file missing or corrupt.
    """
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["results"], set(data["completed_ids"])
    except (EOFError, pickle.UnpicklingError, KeyError, ValueError,
            TypeError, AttributeError, OSError):
        return [], set()


def mark_complete(checkpoint_dir, scenario, model_short, gpu_id, total_gpus):
    """Write completion marker file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    marker = os.path.join(
        checkpoint_dir,
        f"{scenario}_eval_{model_short}_gpu{gpu_id}_of_{total_gpus}_COMPLETE",
    )
    with open(marker, "w") as f:
        f.write(str(time.time()))


def result_path(results_dir, scenario, model_short, gpu_id, total_gpus):
    """Generate result file path."""
    return os.path.join(
        results_dir,
        f"{scenario}_eval_{model_short}_gpu{gpu_id}_of_{total_gpus}.json",
    )


def checkpoint_path(checkpoint_dir, scenario, model_short, gpu_id, total_gpus):
    """Generate checkpoint file path."""
    return os.path.join(
        checkpoint_dir,
        f"{scenario}_eval_{model_short}_gpu{gpu_id}_of_{total_gpus}_checkpoint.pkl",
    )
