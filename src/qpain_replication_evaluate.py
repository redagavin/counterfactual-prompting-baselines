# ABOUTME: GPT-2 Large evaluator for Q-Pain replication experiment
# ABOUTME: Extracts raw p(No) from vocabulary softmax for 28 pairwise comparisons

import json
import os
import pickle
import random
import shutil
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpain_replication_data import (
    build_open_prompt, build_closed_prompts_for_vignette, SUBGROUPS,
)

FIXED_SENTENCE_PREFIX = "This patient record has been reviewed. "


def extract_probabilities(logits, token_id_yes, token_id_no):
    """Extract raw p(No) and p(Yes) from full vocabulary softmax.

    Returns raw probabilities (not normalized over Yes/No), matching
    the original Q-Pain code: np.exp(logprobs_yn.get(' No', -10000)).
    """
    probs = torch.softmax(logits, dim=0)
    return probs[token_id_no].item(), probs[token_id_yes].item()


class Gpt2ReplicationEvaluator:
    """Evaluates prompts using GPT-2 Large, extracting raw p(No) from softmax."""

    def __init__(self, model_name="gpt2-large", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=device,
        )
        self.model.eval()
        self.token_id_yes = self.tokenizer.encode(" Yes", add_special_tokens=False)
        self.token_id_no = self.tokenizer.encode(" No", add_special_tokens=False)
        assert len(self.token_id_yes) == 1, "' Yes' must be a single token"
        assert len(self.token_id_no) == 1, "' No' must be a single token"
        self.token_id_yes = self.token_id_yes[0]
        self.token_id_no = self.token_id_no[0]

    @torch.no_grad()
    def evaluate_prompt(self, prompt_text):
        """Run forward pass, return (prob_no, prob_yes) from last-token softmax."""
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :]
        return extract_probabilities(last_logits, self.token_id_yes, self.token_id_no)


def save_eval_checkpoint(path, results, completed_ids):
    """Atomically save evaluation checkpoint to disk."""
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


def load_eval_checkpoint(path):
    """Load evaluation checkpoint, returning empty state if missing or corrupt."""
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["results"], set(data["completed_ids"])
    except (EOFError, pickle.UnpicklingError, KeyError, ValueError,
            TypeError, AttributeError, OSError):
        return [], set()


def _result_key(vignette_id, subgroup, variant_type, comparison_label=None):
    """Unique key for deduplication in checkpoint."""
    if comparison_label:
        return f"{vignette_id}_{subgroup}_{variant_type}_{comparison_label}"
    return f"{vignette_id}_{subgroup}_{variant_type}"


def run_evaluation(evaluator, dataset, comparisons, closed_data,
                   output_path, checkpoint_path, paraphrases=None,
                   checkpoint_freq=5, seed=42):
    """Run GPT-2 evaluation over all variants with checkpointing.

    Evaluates:
    1. Demographic: 8 subgroups x N vignettes
    2. Fixed sentence: per unique original subgroup x N vignettes
    3. Paraphrase: per comparison x N vignettes (if paraphrases provided)

    All variants for the same vignette share identical closed prompts.
    """
    results, completed_ids = load_eval_checkpoint(checkpoint_path)
    total_done = len(completed_ids)

    rng = random.Random(seed)

    for record in dataset:
        vid = record["vignette_id"]
        ctx = record["context"]

        # Build shared closed prompts using the longest possible open prompt
        # (fixed sentence prefix + longest vignette + longest question) to ensure
        # ALL variants fit within the context window.
        # This must run unconditionally (even when all evaluations for this vignette
        # are checkpointed) to maintain RNG determinism on resume.
        all_open_prompts = []
        for sg in SUBGROUPS:
            # Fixed sentence variant is the longest for each subgroup
            ft = FIXED_SENTENCE_PREFIX + record["texts"][sg]
            all_open_prompts.append(build_open_prompt(ft, record["questions"][sg]))
        ref_open = max(all_open_prompts, key=len)
        closed_a, closed_b = build_closed_prompts_for_vignette(
            record["vignette_idx_in_context"], ctx, closed_data, rng,
            open_prompt=ref_open, tokenizer=evaluator.tokenizer,
        )

        # 1. Demographic evaluations
        for subgroup in SUBGROUPS:
            key = _result_key(vid, subgroup, "demographic")
            if key in completed_ids:
                continue
            open_p = build_open_prompt(
                record["texts"][subgroup], record["questions"][subgroup],
            )
            full_prompt = closed_a + closed_b + open_p
            prob_no, prob_yes = evaluator.evaluate_prompt(full_prompt)
            results.append({
                "vignette_id": vid, "context": ctx,
                "subgroup": subgroup, "variant_type": "demographic",
                "prob_no": prob_no, "prob_yes": prob_yes,
            })
            completed_ids.add(key)

        # 2. Fixed sentence evaluations (deduplicate by subgroup)
        seen_fixed = set()
        for comp in comparisons:
            orig = comp["original"]
            if orig in seen_fixed:
                continue
            seen_fixed.add(orig)
            key = _result_key(vid, orig, "fixed_sentence")
            if key in completed_ids:
                continue
            fixed_text = FIXED_SENTENCE_PREFIX + record["texts"][orig]
            open_p = build_open_prompt(fixed_text, record["questions"][orig])
            full_prompt = closed_a + closed_b + open_p
            prob_no, prob_yes = evaluator.evaluate_prompt(full_prompt)
            results.append({
                "vignette_id": vid, "context": ctx,
                "subgroup": orig, "variant_type": "fixed_sentence",
                "prob_no": prob_no, "prob_yes": prob_yes,
            })
            completed_ids.add(key)

        # 3. Paraphrase evaluations
        if paraphrases:
            for comp in comparisons:
                label = f"{comp['original']}_vs_{comp['swapped']}"
                key = _result_key(vid, comp["original"], "paraphrase", label)
                if key in completed_ids:
                    continue
                para_key = f"{label}_{vid}"
                if para_key not in paraphrases:
                    continue
                para_text = paraphrases[para_key]["paraphrase"]
                open_p = build_open_prompt(para_text, record["questions"][comp["original"]])
                full_prompt = closed_a + closed_b + open_p
                prob_no, prob_yes = evaluator.evaluate_prompt(full_prompt)
                results.append({
                    "vignette_id": vid, "context": ctx,
                    "subgroup": comp["original"],
                    "variant_type": "paraphrase",
                    "comparison": label,
                    "prob_no": prob_no, "prob_yes": prob_yes,
                })
                completed_ids.add(key)

        if len(completed_ids) - total_done >= checkpoint_freq:
            save_eval_checkpoint(checkpoint_path, results, completed_ids)
            total_done = len(completed_ids)

    save_eval_checkpoint(checkpoint_path, results, completed_ids)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
