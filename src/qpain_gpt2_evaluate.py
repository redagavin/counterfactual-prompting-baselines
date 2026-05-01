# ABOUTME: GPT-2 Large evaluator for Q-Pain race bias experiment with few-shot prompting
# ABOUTME: Matches original Q-Pain code: 2 closed prompts (Patient C commented out), completion model

import csv
import json
import os
import pickle
import random
import shutil
import tempfile

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "physionet.org/files/q-pain/1.0.0"

CONTEXT_FILES = {
    "acute_cancer": "data_acute_cancer.csv",
    "acute_non_cancer": "data_acute_non_cancer.csv",
    "chronic_cancer": "data_chronic_cancer.csv",
    "chronic_non_cancer": "data_chronic_non_cancer.csv",
    "post_op": "data_post_op.csv",
}

# Dosage text per context (High and Low variants)
DOSAGE_HIGH = {
    "acute_cancer": "Dosage: High (1 mg)",
    "acute_non_cancer": "Dosage: High (1 mg)",
    "chronic_cancer": "Dosage: High (4 weeks)",
    "chronic_non_cancer": "Dosage: High (4 weeks)",
    "post_op": "Dosage: High (4 weeks)",
}
DOSAGE_LOW = {
    "acute_cancer": "Dosage: Low (0.5 mg)",
    "acute_non_cancer": "Dosage: Low (0.5 mg)",
    "chronic_cancer": "Dosage: Low (1 week)",
    "chronic_non_cancer": "Dosage: Low (1 week)",
    "post_op": "Dosage: Low (1 week)",
}


def standardize_closed(text, patient_name):
    """Remove demographics and rename Patient D for a closed prompt.

    Matches the original Q-Pain paper's standardize_closed:
    - Delete [race] and [gender] with trailing space
    - Replace [subjective] and [subject] with patient_name
    - Replace [possessive] with 'their'
    - Replace 'Patient D' with patient_name
    """
    text = text.replace("[gender] ", "")
    text = text.replace("[race] ", "")
    text = text.replace("[subjective]", patient_name)
    text = text.replace("[subject]", patient_name)
    text = text.replace("[possessive]", "their")
    text = text.replace("Patient D", patient_name)
    return text


def build_closed_prompt(vignette, question, answer, dosage, explanation):
    """Format a single closed (few-shot example) prompt."""
    return (
        f"Vignette: {vignette}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"{dosage}\n"
        f"Explanation: {explanation}##\n\n"
    )


def build_open_prompt(vignette, question):
    """Format the open (test) prompt ending with 'Answer:'."""
    return (
        f"Vignette: {vignette}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def assemble_few_shot_prompt(closed_a, closed_b, open_prompt):
    """Concatenate 2 closed prompts + 1 open prompt.

    Matches the original Q-Pain code where Patient C is commented out.
    """
    return closed_a + closed_b + open_prompt


def compute_log_odds(logit_yes, logit_no):
    """Compute log-odds: logit(' Yes') - logit(' No'). Positive = treatment favored."""
    return logit_yes - logit_no


def load_closed_data(data_dir=DATA_DIR):
    """Load Patient B and all Patient D vignettes per context for closed prompt construction."""
    result = {}
    for context, filename in CONTEXT_FILES.items():
        path = f"{data_dir}/{filename}"
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        patient_b = None
        yes_vignettes = []
        for row in rows:
            if "[race]" not in row["Vignette"]:
                patient_b = {
                    "vignette": row["Vignette"],
                    "question": row["Question"],
                    "answer": row["Answer"].strip(),
                    "dosage": str(row["Dosage"]),
                    "explanation": row["Explanation"],
                }
            else:
                yes_vignettes.append({
                    "vignette": row["Vignette"],
                    "question": row["Question"],
                    "answer": row["Answer"].strip(),
                    "explanation": row["Explanation"],
                })

        result[context] = {"patient_b": patient_b, "yes_vignettes": yes_vignettes}
    return result


def build_closed_prompts_for_vignette(vignette_idx, context, closed_data, rng,
                                      open_prompt, tokenizer, max_context=1024):
    """Build the 2 closed prompts for a given open vignette.

    Matches the original Q-Pain code where Patient C is commented out.
    Resamples Patient A if the total prompt exceeds GPT-2's context window.
    Returns (closed_a, closed_b) strings.
    """
    ctx_data = closed_data[context]
    available = [i for i in range(10) if i != vignette_idx]

    # Patient B is fixed per context
    pb = ctx_data["patient_b"]
    closed_b = build_closed_prompt(
        vignette=pb["vignette"], question=pb["question"],
        answer=pb["answer"], dosage=f"Dosage: {pb['dosage']}",
        explanation=pb["explanation"],
    )

    for _ in range(10):
        # Draw 2 indices to match original code's RNG consumption pattern,
        # but only use sample[0] for Patient A (sample[1] would be Patient C)
        sample = rng.sample(available, 2)

        # Patient A: Yes, High dose
        v_a = ctx_data["yes_vignettes"][sample[0]]
        closed_a = build_closed_prompt(
            vignette=standardize_closed(v_a["vignette"], "Patient A"),
            question=standardize_closed(v_a["question"], "Patient A"),
            answer=v_a["answer"],
            dosage=DOSAGE_HIGH[context],
            explanation=standardize_closed(v_a["explanation"], "Patient A"),
        )

        # Check total token count fits GPT-2 context window
        full_prompt = assemble_few_shot_prompt(closed_a, closed_b, open_prompt)
        n_tokens = len(tokenizer.encode(full_prompt))
        if n_tokens <= max_context:
            return closed_a, closed_b

    # If all resamples exceed context, use the last one (truncated by model)
    return closed_a, closed_b


class Gpt2QpainEvaluator:
    """Evaluates Q-Pain vignettes using GPT-2 Large with few-shot prompting."""

    def __init__(self, model_name="gpt2-large", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=device
        )
        self.model.eval()
        # GPT-2 uses " Yes" and " No" (with leading space) after "Answer:"
        self.token_id_yes = self.tokenizer.encode(" Yes", add_special_tokens=False)
        self.token_id_no = self.tokenizer.encode(" No", add_special_tokens=False)
        assert len(self.token_id_yes) == 1
        assert len(self.token_id_no) == 1
        self.token_id_yes = self.token_id_yes[0]
        self.token_id_no = self.token_id_no[0]

    @torch.no_grad()
    def extract_log_odds(self, prompt_text):
        """Extract log-odds from the next-token position after the full prompt."""
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :]
        return compute_log_odds(
            last_logits[self.token_id_yes].item(),
            last_logits[self.token_id_no].item(),
        )

    def evaluate_sample(self, sample, closed_a, closed_b):
        """Evaluate one vignette with its few-shot context (2 closed prompts).

        Output fields: bio_id, swap_direction, logit_original (White),
        logit_swapped (Black), logit_asian, logit_fixed_sentence,
        optionally logit_paraphrase, logit_asian_paraphrase.
        """
        # Replace Patient D in question with race-specific name (matching original code's
        # race_name_open which replaces Patient D in the entire prompt including question)
        white_question = sample["question"].replace("Patient D", sample["white_name"])
        black_question = sample["question"].replace("Patient D", sample["black_name"])

        # Build open prompts for each race variant
        white_open = build_open_prompt(sample["white_text"], white_question)
        black_open = build_open_prompt(sample["black_text"], black_question)
        fixed_open = build_open_prompt(sample["fixed_sentence_text"], white_question)

        result = {
            "bio_id": sample["vignette_id"],
            "swap_direction": sample["swap_direction"],
            "logit_original": self.extract_log_odds(
                assemble_few_shot_prompt(closed_a, closed_b, white_open)
            ),
            "logit_swapped": self.extract_log_odds(
                assemble_few_shot_prompt(closed_a, closed_b, black_open)
            ),
            "logit_fixed_sentence": self.extract_log_odds(
                assemble_few_shot_prompt(closed_a, closed_b, fixed_open)
            ),
        }

        if "asian_text" in sample:
            asian_question = sample["question"].replace("Patient D", sample["asian_name"])
            asian_open = build_open_prompt(sample["asian_text"], asian_question)
            result["logit_asian"] = self.extract_log_odds(
                assemble_few_shot_prompt(closed_a, closed_b, asian_open)
            )

        if "paraphrase_text" in sample and pd.notna(sample.get("paraphrase_text")):
            para_open = build_open_prompt(sample["paraphrase_text"], white_question)
            result["logit_paraphrase"] = self.extract_log_odds(
                assemble_few_shot_prompt(closed_a, closed_b, para_open)
            )

        if "asian_paraphrase_text" in sample and pd.notna(sample.get("asian_paraphrase_text")):
            asian_para_open = build_open_prompt(sample["asian_paraphrase_text"], white_question)
            result["logit_asian_paraphrase"] = self.extract_log_odds(
                assemble_few_shot_prompt(closed_a, closed_b, asian_para_open)
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


def run_gpt2_evaluation(evaluator, samples, closed_data, output_path,
                        checkpoint_path, checkpoint_freq=5, seed=42):
    """Run GPT-2 evaluation over samples with few-shot prompts and checkpointing."""
    results, completed_ids = load_eval_checkpoint(checkpoint_path)
    print(f"Resuming from checkpoint: {len(completed_ids)} completed")

    rng = random.Random(seed)
    # Pre-build closed prompts per vignette (deterministic with seed)
    context_vignette_idx = {}
    for sample in samples:
        ctx = sample["context"]
        if ctx not in context_vignette_idx:
            context_vignette_idx[ctx] = 0
        idx = context_vignette_idx[ctx]
        sample["_vignette_idx_in_context"] = idx
        context_vignette_idx[ctx] = idx + 1

    for sample in samples:
        vid = sample["vignette_id"]
        if vid in completed_ids:
            # Still need to consume RNG state to keep sequence aligned.
            # Call the same function (which may resample) to advance RNG identically.
            white_question = sample["question"].replace("Patient D", sample["white_name"])
            ref_open = build_open_prompt(sample["fixed_sentence_text"], white_question)
            build_closed_prompts_for_vignette(
                sample["_vignette_idx_in_context"], sample["context"], closed_data, rng,
                open_prompt=ref_open, tokenizer=evaluator.tokenizer,
            )
            continue

        # Use fixed_sentence variant for token counting — it's the longest open prompt
        white_question = sample["question"].replace("Patient D", sample["white_name"])
        ref_open = build_open_prompt(sample["fixed_sentence_text"], white_question)
        closed_a, closed_b = build_closed_prompts_for_vignette(
            sample["_vignette_idx_in_context"], sample["context"], closed_data, rng,
            open_prompt=ref_open, tokenizer=evaluator.tokenizer,
        )
        result = evaluator.evaluate_sample(sample, closed_a, closed_b)
        results.append(result)
        completed_ids.add(vid)

        if len(completed_ids) % checkpoint_freq == 0:
            save_eval_checkpoint(checkpoint_path, results, completed_ids)
            print(f"Checkpoint saved: {len(completed_ids)}/{len(samples)}")

    save_eval_checkpoint(checkpoint_path, results, completed_ids)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete: {len(results)} results saved to {output_path}")
    return results
