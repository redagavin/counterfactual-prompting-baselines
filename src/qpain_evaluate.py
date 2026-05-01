# ABOUTME: Qwen3 evaluation with logit extraction for Q-Pain race bias vignette classification
# ABOUTME: Extracts log-odds from "Yes" and "No" token logits using Qwen3 in non-thinking mode

import json
import os
import pickle
import shutil
import tempfile

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """Vignette: {vignette_text}

{core_question} Respond with only Yes or No.

Answer (Yes or No):"""


def format_prompt(vignette_text, core_question):
    """Format a vignette and question into the classification prompt."""
    return PROMPT_TEMPLATE.format(vignette_text=vignette_text, core_question=core_question)


def compute_log_odds(logit_yes, logit_no):
    """Compute log-odds from raw logits: logit_yes - logit_no.

    Positive = treatment favored, negative = treatment not favored.
    Equivalent to log(P(Yes)/P(No)) since softmax normalization cancels.
    """
    return logit_yes - logit_no


class QpainEvaluator:
    """Evaluates Q-Pain vignettes using Qwen3 in non-thinking mode."""

    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        # Validate Yes/No are single tokens
        self.token_id_yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        self.token_id_no = self.tokenizer.encode("No", add_special_tokens=False)
        assert len(self.token_id_yes) == 1
        assert len(self.token_id_no) == 1
        self.token_id_yes = self.token_id_yes[0]
        self.token_id_no = self.token_id_no[0]

    def _format_chat(self, prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _logits_to_log_odds(self, logits):
        last_logits = logits[0, -1, :]
        return compute_log_odds(
            last_logits[self.token_id_yes].item(),
            last_logits[self.token_id_no].item(),
        )

    @torch.no_grad()
    def extract_log_odds(self, vignette_text, core_question):
        prompt = format_prompt(vignette_text, core_question)
        chat_text = self._format_chat(prompt)
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return self._logits_to_log_odds(outputs.logits)

    def evaluate_sample(self, sample):
        """Evaluate one vignette across all text variants.

        Output fields match bib_analysis.py expectations:
        bio_id = vignette_id, logit_original = White, logit_swapped = Black.
        logit_asian added when asian_text is present.
        """
        core_question = sample["core_question"]
        black_question = core_question.replace(sample["white_name"], sample["black_name"])

        result = {
            "bio_id": sample["vignette_id"],
            "swap_direction": sample["swap_direction"],
            "logit_original": self.extract_log_odds(sample["white_text"], core_question),
            "logit_swapped": self.extract_log_odds(sample["black_text"], black_question),
            "logit_fixed_sentence": self.extract_log_odds(sample["fixed_sentence_text"], core_question),
        }
        if "paraphrase_text" in sample and pd.notna(sample.get("paraphrase_text")):
            result["logit_paraphrase"] = self.extract_log_odds(sample["paraphrase_text"], core_question)
        if "asian_text" in sample:
            asian_question = core_question.replace(sample["white_name"], sample["asian_name"])
            result["logit_asian"] = self.extract_log_odds(sample["asian_text"], asian_question)
        if "asian_paraphrase_text" in sample and pd.notna(sample.get("asian_paraphrase_text")):
            result["logit_asian_paraphrase"] = self.extract_log_odds(
                sample["asian_paraphrase_text"], core_question
            )
        return result


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


def run_evaluation(evaluator, samples, output_path, checkpoint_path,
                   checkpoint_freq=5):
    """Run evaluation over samples with periodic checkpointing."""
    results, completed_ids = load_eval_checkpoint(checkpoint_path)
    print(f"Resuming from checkpoint: {len(completed_ids)} completed")

    for i, sample in enumerate(samples):
        vignette_id = sample["vignette_id"]
        if vignette_id in completed_ids:
            continue
        result = evaluator.evaluate_sample(sample)
        results.append(result)
        completed_ids.add(vignette_id)
        if (i + 1) % checkpoint_freq == 0:
            save_eval_checkpoint(checkpoint_path, results, completed_ids)
            print(f"Checkpoint saved: {len(completed_ids)}/{len(samples)}")

    save_eval_checkpoint(checkpoint_path, results, completed_ids)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete: {len(results)} results saved to {output_path}")
    return results
