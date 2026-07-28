# ABOUTME: Llama yes/no log-odds extraction for DiscrimEval (single deterministic forward pass)
# ABOUTME: FP8 for 70B, single-token yes/no IDs, full-vocab mass gate, SLURM sharding + checkpoint

import json
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrimeval_data import build_decision_prompt  # noqa: E402


# --- Vendored from MedPerturb/code/eval_utils.py to keep this module
# --- self-contained and release-clean (no cross-tree imports). Keep in sync.
def detect_slurm():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return (int(os.environ["SLURM_ARRAY_TASK_ID"]),
                int(os.environ["SLURM_ARRAY_TASK_COUNT"]))
    return None, None


def shard_samples(samples, gpu_id, total_gpus):
    return samples[gpu_id::total_gpus]


def model_short_name(model_name):
    return model_name.split("/")[-1].lower().replace("-", "_")


def save_checkpoint(path, results, completed_ids):
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
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["results"], set(data["completed_ids"])
    except (EOFError, pickle.UnpicklingError, KeyError, ValueError,
            TypeError, AttributeError, OSError):
        return [], set()


def log_odds_from_logits(logits, yes_id, no_id):
    """logit_yes - logit_no from a full next-token logits vector."""
    return float(logits[yes_id] - logits[no_id])


def yes_no_mass(logits, yes_id, no_id):
    """Share of full-vocab softmax probability mass on the yes and no tokens."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max()
    probs = np.exp(logits - m)
    probs /= probs.sum()
    return float(probs[yes_id] + probs[no_id])


class DiscrimEvalEvaluator:
    """Loads a Llama model and extracts yes/no log-odds via one forward pass."""

    def __init__(self, model_name):
        import torch
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                   FineGrainedFP8Config)
        self.torch = torch
        # No explicit token= : transformers resolves auth from the ambient HF
        # environment (HF CLI cache or HF_TOKEN env), matching bib_evaluate.py.
        # This keeps the module release-clean (no token-file/path dependencies).
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "70B" in model_name or "70b" in model_name:
            print("Using FineGrainedFP8Config for 70B model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto",
                quantization_config=FineGrainedFP8Config())
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto")
        self.model.eval()
        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(f"Yes/No not single tokens: {yes_ids}, {no_ids}")
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]

    def _last_logits(self, user_content):
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1, :].float().cpu().numpy()

    def log_odds(self, filled_template):
        logits = self._last_logits(build_decision_prompt(filled_template))
        return log_odds_from_logits(logits, self.yes_id, self.no_id)

    def mass(self, filled_template):
        logits = self._last_logits(build_decision_prompt(filled_template))
        return yes_no_mass(logits, self.yes_id, self.no_id)

    def evaluate_sample(self, sample):
        res = {"decision_question_id": sample["decision_question_id"],
               "logit_reference": self.log_odds(sample["reference_text"]),
               "contrasts": {}}
        for key, contrast_text in sample["contrasts"].items():
            para_text = sample["paraphrases"].get(key)
            res["contrasts"][key] = {
                "logit_contrast": self.log_odds(contrast_text),
                "logit_paraphrase": (self.log_odds(para_text)
                                     if para_text is not None else float("nan")),
            }
        return res


def run_evaluation(evaluator, samples, output_path, checkpoint_path, checkpoint_freq=10):
    """Evaluate samples with periodic checkpointing (pickle), then write JSON."""
    results, completed = load_checkpoint(checkpoint_path)
    print(f"Resuming: {len(completed)} scenarios done")
    for i, s in enumerate(samples):
        qid = s["decision_question_id"]
        if qid in completed:
            continue
        results.append(evaluator.evaluate_sample(s))
        completed.add(qid)
        if (i + 1) % checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, results, completed)
    save_checkpoint(checkpoint_path, results, completed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} scenarios to {output_path}")
    return results
