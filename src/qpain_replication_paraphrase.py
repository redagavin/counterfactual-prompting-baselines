# ABOUTME: Calibrated paraphrase generation for Q-Pain replication experiment
# ABOUTME: Generates 1400 paraphrases (28 comparisons x 50 vignettes) via GPT-5.2 async

import asyncio
import json
import os
import shutil
import tempfile

from calibrated_paraphrase import (
    is_refusal, select_best_undershoot, should_switch_to_synonym_strategy,
)
from token_edit_distance import token_edit_distance_percent


INITIAL_PROMPT_TEMPLATE = """You are tasked with paraphrasing a clinical vignette.

TARGET: Change exactly {target_pct:.1f}% of tokens ({tolerance_str} tolerance).

HOW WE MEASURE:
We tokenize both texts using a language model tokenizer, then calculate:
  edit_distance(original_tokens, paraphrased_tokens) / len(original_tokens) x 100
Edit distance counts insertions, deletions, and substitutions.

CRITICAL REQUIREMENTS:
- The paraphrase MUST be semantically equivalent to the original — same meaning, same clinical information
- Preserve the patient name EXACTLY (every occurrence)
- Preserve ALL demographic language EXACTLY (race, gender, pronouns)
- Preserve ALL medical terminology EXACTLY (drug names, diagnoses, procedures, anatomical terms)
- Preserve ALL clinical facts EXACTLY (numbers, ages, dates, test results, symptoms, dosages)
- You MUST achieve approximately {target_pct:.1f}% token change

Original vignette ({orig_token_count} tokens):
{vignette}

Target edits: ~{target_edits} token operations (insertions, deletions, or substitutions)

Provide ONLY the paraphrased vignette, nothing else."""


RETRY_PROMPT_TEMPLATE = """That changed {actual_pct:.1f}% of tokens. Target is {target_pct:.1f}% ({tolerance_str}).
{direction_hint}
Try again."""


RESET_HINT_TEMPLATE = """
Note: A previous attempt achieved {best_actual_pct:.1f}% token change.
You need to make {direction} changes to reach the target."""


SYNONYM_PROMPT_TEMPLATE = """Paraphrase the following clinical vignette by replacing approximately {num_words} words with synonyms.

CRITICAL RULES:
- Achieve the paraphrase ONLY through synonym replacements
- Keep ALL other text EXACTLY the same — same structure, same formatting, same punctuation
- Do NOT rewrite, restructure, or add/remove content
- Preserve the patient name EXACTLY (every occurrence)
- Preserve ALL demographic language (race, gender, pronouns)
- Preserve ALL medical terminology, numbers, dates, dosages, and test results

Vignette:
{vignette}

Provide ONLY the paraphrased vignette, nothing else."""


def compute_paraphrase_targets(dataset, comparisons, tokenizer):
    """Compute token edit distance targets for all comparison x vignette pairs."""
    targets = {}
    for comp in comparisons:
        label = f"{comp['original']}_vs_{comp['swapped']}"
        for record in dataset:
            vid = record["vignette_id"]
            orig_text = record["texts"][comp["original"]]
            swap_text = record["texts"][comp["swapped"]]
            target_pct = token_edit_distance_percent(orig_text, swap_text, tokenizer)
            key = f"{label}_{vid}"
            targets[key] = {
                "original_text": orig_text,
                "target_pct": target_pct,
                "vignette_id": vid,
                "comparison_label": label,
            }
    return targets


async def generate_one_paraphrase(target, tokenizer, openai_client,
                                   max_retries=50, tolerance=0.5):
    """Generate a single calibrated paraphrase targeting the given edit distance %."""
    vignette = target["original_text"]
    target_pct = target["target_pct"]
    orig_tokens = tokenizer.encode(vignette, add_special_tokens=False)
    orig_token_count = len(orig_tokens)
    target_edits = round(target_pct * orig_token_count / 100)
    tolerance_str = f"\u00b1{tolerance}%"

    messages = [{
        "role": "user",
        "content": INITIAL_PROMPT_TEMPLATE.format(
            target_pct=target_pct, tolerance_str=tolerance_str,
            orig_token_count=orig_token_count, vignette=vignette,
            target_edits=target_edits,
        ),
    }]

    all_attempts = []

    for retry in range(max_retries):
        if retry > 0 and retry % 10 == 0:
            best = select_best_undershoot(all_attempts, target_pct)
            direction = "more" if best and best["actual_pct"] < target_pct else "fewer"
            hint = RESET_HINT_TEMPLATE.format(
                best_actual_pct=best["actual_pct"] if best else 0,
                direction=direction,
            )
            messages = [{
                "role": "user",
                "content": INITIAL_PROMPT_TEMPLATE.format(
                    target_pct=target_pct, tolerance_str=tolerance_str,
                    orig_token_count=orig_token_count, vignette=vignette,
                    target_edits=target_edits,
                ) + hint,
            }]

        response = await openai_client.responses.create(
            model="gpt-5.2", input=messages,
        )
        paraphrase = response.output_text.strip().strip('"').strip("'")

        if is_refusal(paraphrase):
            continue

        actual_pct = token_edit_distance_percent(vignette, paraphrase, tokenizer)
        deviation = abs(actual_pct - target_pct)
        all_attempts.append({
            "paraphrase": paraphrase, "actual_pct": actual_pct,
            "deviation": deviation,
        })

        if deviation <= tolerance:
            return {
                "paraphrase": paraphrase, "actual_pct": actual_pct,
                "retries_used": retry, "deviation": deviation,
            }

        direction_hint = "Make fewer changes." if actual_pct > target_pct else "Make more changes."
        messages.append({"role": "assistant", "content": paraphrase})
        messages.append({
            "role": "user",
            "content": RETRY_PROMPT_TEMPLATE.format(
                actual_pct=actual_pct, target_pct=target_pct,
                tolerance_str=tolerance_str, direction_hint=direction_hint,
            ),
        })

    # Phase 2: synonym fallback if all attempts overshoot
    if should_switch_to_synonym_strategy(all_attempts, target_pct, tolerance):
        num_words = max(1, target_edits)
        syn_messages = [{
            "role": "user",
            "content": SYNONYM_PROMPT_TEMPLATE.format(
                num_words=num_words, vignette=vignette,
            ),
        }]
        response = await openai_client.responses.create(
            model="gpt-5.2", input=syn_messages,
        )
        paraphrase = response.output_text.strip().strip('"').strip("'")
        actual_pct = token_edit_distance_percent(vignette, paraphrase, tokenizer)
        all_attempts.append({
            "paraphrase": paraphrase, "actual_pct": actual_pct,
            "deviation": abs(actual_pct - target_pct),
        })

    # Phase 3: best undershoot
    best = select_best_undershoot(all_attempts, target_pct)
    if best:
        return {
            "paraphrase": best["paraphrase"], "actual_pct": best["actual_pct"],
            "retries_used": max_retries, "deviation": best["deviation"],
        }
    best = min(all_attempts, key=lambda a: a["deviation"])
    return {
        "paraphrase": best["paraphrase"], "actual_pct": best["actual_pct"],
        "retries_used": max_retries, "deviation": best["deviation"],
    }


async def generate_all_paraphrases(targets, tokenizer, openai_client,
                                    output_path, max_concurrent=50,
                                    checkpoint_freq=10):
    """Generate calibrated paraphrases for all targets with async parallelism."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = {}

    remaining = {k: v for k, v in targets.items() if k not in results}
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0

    async def process_one(key, target):
        nonlocal completed_count
        async with semaphore:
            result = await generate_one_paraphrase(target, tokenizer, openai_client)
            result["vignette_id"] = target["vignette_id"]
            result["comparison_label"] = target["comparison_label"]
            results[key] = result
            completed_count += 1
            if completed_count % checkpoint_freq == 0:
                _atomic_save_json(results, output_path)

    tasks = [process_one(k, t) for k, t in remaining.items()]
    await asyncio.gather(*tasks)
    _atomic_save_json(results, output_path)
    return results


def _atomic_save_json(data, path):
    """Save JSON atomically using tempfile + rename."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
