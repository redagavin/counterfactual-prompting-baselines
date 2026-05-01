# ABOUTME: Calibrated paraphrase generation for bias-in-bios experiment
# ABOUTME: Targets each bio's gender-swap token distance using GPT with async parallelism

import asyncio
import json
import os
import shutil
import tempfile

from calibrated_paraphrase import is_refusal, select_best_undershoot, should_switch_to_synonym_strategy
from token_edit_distance import token_edit_distance_percent


INITIAL_BIO_PROMPT_TEMPLATE = """You are tasked with paraphrasing a professional biography.

TARGET: Change exactly {target_pct:.1f}% of tokens ({tolerance_str} tolerance).

HOW WE MEASURE:
We tokenize both texts using a language model tokenizer, then calculate:
  edit_distance(original_tokens, paraphrased_tokens) / len(original_tokens) x 100
Edit distance counts insertions, deletions, and substitutions.

CRITICAL REQUIREMENTS:
- The paraphrase MUST be semantically equivalent to the original — same meaning, same information
- Preserve ALL gendered language EXACTLY (pronouns, titles, gendered nouns)
- Preserve ALL proper nouns, numbers, and dates EXACTLY
- You MUST achieve approximately {target_pct:.1f}% token change

Original biography ({orig_token_count} tokens):
{bio}

Target edits: ~{target_edits} token operations (insertions, deletions, or substitutions)

Provide ONLY the paraphrased biography, nothing else."""


RETRY_BIO_PROMPT_TEMPLATE = """That changed {actual_pct:.1f}% of tokens. Target is {target_pct:.1f}% ({tolerance_str}).
{direction_hint}
Try again."""


RESET_BIO_HINT_TEMPLATE = """

Note: A previous attempt achieved {best_actual_pct:.1f}% token change.
You need to make {direction} changes to reach the target."""


SYNONYM_BIO_PROMPT_TEMPLATE = """Paraphrase the following professional biography by replacing approximately {num_words} words with synonyms.

CRITICAL RULES:
- Achieve the paraphrase ONLY through synonym replacements
- Keep ALL other text EXACTLY the same — same structure, same formatting, same punctuation
- Do NOT rewrite, restructure, or add/remove content
- Preserve ALL gendered language (pronouns, titles, gendered nouns)
- Preserve ALL proper nouns, numbers, dates, and names

Biography:
{bio}

Provide ONLY the paraphrased biography, nothing else."""


REFUSAL_CORRECTION_BIO = "That was a refusal. Please provide only the paraphrased biography, nothing else."


def format_initial_bio_prompt(bio, target_pct, orig_token_count, target_edits, tolerance=0.5):
    """Format the initial calibrated paraphrasing prompt for a biography."""
    return INITIAL_BIO_PROMPT_TEMPLATE.format(
        bio=bio,
        target_pct=target_pct,
        orig_token_count=orig_token_count,
        target_edits=target_edits,
        tolerance_str=f"\u00b1{tolerance}%",
    )


def format_retry_bio_prompt(actual_pct, target_pct, tolerance=0.5):
    """Format the retry prompt with directional feedback."""
    if actual_pct > target_pct + tolerance:
        direction_hint = "Make fewer changes. Your paraphrase was too different."
    else:
        direction_hint = "Make more changes. Your paraphrase was too similar."

    return RETRY_BIO_PROMPT_TEMPLATE.format(
        actual_pct=actual_pct,
        target_pct=target_pct,
        direction_hint=direction_hint,
        tolerance_str=f"\u00b1{tolerance}%",
    )


def format_reset_bio_hint(best_actual_pct, target_pct):
    """Format hint about best previous attempt for conversation reset."""
    direction = "more" if best_actual_pct < target_pct else "fewer"
    return RESET_BIO_HINT_TEMPLATE.format(
        best_actual_pct=best_actual_pct,
        direction=direction,
    )


def format_synonym_bio_prompt(bio, num_words):
    """Format a synonym replacement prompt for small target changes."""
    return SYNONYM_BIO_PROMPT_TEMPLATE.format(bio=bio, num_words=num_words)


def merge_paraphrases(balanced_df, paraphrase_results):
    """Merge paraphrase results into the balanced dataframe by bio_id.

    Args:
        balanced_df: DataFrame with bio_id column
        paraphrase_results: dict mapping str(bio_id) -> {paraphrase, actual_pct, ...}

    Returns:
        DataFrame with added paraphrase_text column
    """
    df = balanced_df.copy()
    df['paraphrase_text'] = df['bio_id'].apply(
        lambda bid: paraphrase_results.get(str(bid), {}).get('paraphrase', None)
    )
    return df


def merge_paraphrases_from_file(balanced_df, path):
    """Load paraphrase results from JSON and merge into dataframe."""
    with open(path) as f:
        paraphrase_results = json.load(f)
    return merge_paraphrases(balanced_df, paraphrase_results)


def _atomic_save_json(data, path):
    """Write JSON data atomically using tempfile + rename."""
    dir_name = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _strip_quotes(text):
    """Remove surrounding single or double quotes from text."""
    if len(text) >= 2:
        if text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        if text[0] == "'" and text[-1] == "'":
            return text[1:-1]
    return text


async def generate_one_paraphrase(bio_id, text, swapped_text, tokenizer, openai_client,
                                  max_retries=50, tolerance=0.5):
    """Generate a calibrated paraphrase for one bio targeting its gender-swap distance.

    Three-phase algorithm:
      Phase 1: Multi-turn conversation, reset every 10 retries with directional hint
      Phase 2: Synonym replacement fallback if all Phase 1 attempts overshot
      Phase 3: Return closest undershoot

    Args:
        bio_id: Identifier for this bio
        text: Original bio text
        swapped_text: Gender-swapped version of the bio
        tokenizer: HuggingFace tokenizer for measurement
        openai_client: AsyncOpenAI client
        max_retries: Maximum retry attempts per phase (default 50)
        tolerance: Acceptable deviation in percentage points (default 0.5)

    Returns:
        dict with bio_id, paraphrase, actual_pct, retries_used, deviation, all_attempts
    """
    # Target percentage from gender swap distance
    target_pct = token_edit_distance_percent(text, swapped_text, tokenizer)

    # Calculate target edits
    orig_tokens = tokenizer.encode(text, add_special_tokens=False)
    orig_token_count = len(orig_tokens)
    target_edits = max(1, round(target_pct * orig_token_count / 100))

    # Phase 1: Multi-turn conversation
    initial_prompt = format_initial_bio_prompt(
        bio=text,
        target_pct=target_pct,
        orig_token_count=orig_token_count,
        target_edits=target_edits,
        tolerance=tolerance,
    )

    messages = [{"role": "user", "content": initial_prompt}]
    attempts = []

    for attempt_num in range(max_retries + 1):
        # Conversation reset every 10 retries
        if attempt_num > 0 and attempt_num % 10 == 0:
            if attempts:
                best_so_far = min(attempts, key=lambda x: x['deviation'])
                hint = format_reset_bio_hint(best_so_far['actual_pct'], target_pct)
                messages = [{"role": "user", "content": initial_prompt + hint}]
            else:
                messages = [{"role": "user", "content": initial_prompt}]

        response = await openai_client.responses.create(
            model="gpt-5.2",
            input=messages,
        )
        paraphrase = _strip_quotes(response.output_text.strip())

        # Refusal detection
        if is_refusal(paraphrase):
            if attempt_num < max_retries:
                messages.append({"role": "assistant", "content": paraphrase})
                messages.append({"role": "user", "content": REFUSAL_CORRECTION_BIO})
            continue

        actual_pct = token_edit_distance_percent(text, paraphrase, tokenizer)
        deviation = abs(actual_pct - target_pct)

        attempts.append({
            'paraphrase': paraphrase,
            'actual_pct': actual_pct,
            'deviation': deviation,
        })

        if deviation <= tolerance:
            return {
                'bio_id': bio_id,
                'paraphrase': paraphrase,
                'actual_pct': actual_pct,
                'retries_used': attempt_num,
                'deviation': deviation,
                'all_attempts': attempts,
            }

        if attempt_num < max_retries:
            messages.append({"role": "assistant", "content": paraphrase})
            retry_prompt = format_retry_bio_prompt(actual_pct, target_pct, tolerance)
            messages.append({"role": "user", "content": retry_prompt})

    if not attempts:
        print(f"  WARNING: bio_id={bio_id} — all {max_retries + 1} attempts were refusals, using original text")
        return {
            'bio_id': bio_id,
            'paraphrase': text,
            'actual_pct': 0.0,
            'retries_used': max_retries,
            'deviation': target_pct,
            'all_attempts': [],
        }

    # Phase 2: Synonym replacement when all Phase 1 attempts overshot
    if should_switch_to_synonym_strategy(attempts, target_pct, tolerance):
        num_words = max(1, target_edits)
        synonym_prompt = format_synonym_bio_prompt(text, num_words)

        for attempt_num in range(max_retries + 1):
            response = await openai_client.responses.create(
                model="gpt-5.2",
                input=[{"role": "user", "content": synonym_prompt}],
            )
            paraphrase = _strip_quotes(response.output_text.strip())

            if is_refusal(paraphrase):
                continue

            actual_pct = token_edit_distance_percent(text, paraphrase, tokenizer)
            deviation = abs(actual_pct - target_pct)

            attempts.append({
                'paraphrase': paraphrase,
                'actual_pct': actual_pct,
                'deviation': deviation,
            })

            if deviation <= tolerance:
                return {
                    'bio_id': bio_id,
                    'paraphrase': paraphrase,
                    'actual_pct': actual_pct,
                    'retries_used': max_retries + attempt_num + 1,
                    'deviation': deviation,
                    'all_attempts': attempts,
                }

    # Phase 3: Return closest undershoot
    best_attempt = select_best_undershoot(attempts, target_pct)

    return {
        'bio_id': bio_id,
        'paraphrase': best_attempt['paraphrase'],
        'actual_pct': best_attempt['actual_pct'],
        'retries_used': max_retries,
        'deviation': best_attempt['deviation'],
        'all_attempts': attempts,
    }


async def generate_all_paraphrases(balanced_df, output_path, tokenizer, openai_client,
                                   max_concurrent=300, checkpoint_freq=50):
    """Generate calibrated paraphrases for all bios with async parallelism.

    Args:
        balanced_df: DataFrame with bio_id, hard_text, swapped_text columns
        output_path: Path to save JSON results (supports resume from checkpoint)
        tokenizer: HuggingFace tokenizer for measurement
        openai_client: AsyncOpenAI client
        max_concurrent: Maximum concurrent API calls (default 300)
        checkpoint_freq: Save checkpoint every N completions (default 50)

    Returns:
        dict mapping str(bio_id) -> result dict
    """
    # Resume from existing checkpoint
    results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resumed {len(results)} existing results from {output_path}")

    done_ids = set(results.keys())
    pending = balanced_df[~balanced_df['bio_id'].astype(str).isin(done_ids)]
    print(f"Generating paraphrases for {len(pending)} bios ({len(done_ids)} already done)")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = [0]
    lock = asyncio.Lock()

    async def process_one(row):
        bio_id = row['bio_id']
        async with semaphore:
            try:
                result = await generate_one_paraphrase(
                    bio_id=bio_id,
                    text=row['hard_text'],
                    swapped_text=row['swapped_text'],
                    tokenizer=tokenizer,
                    openai_client=openai_client,
                )
                # Strip all_attempts to save space in checkpoint
                save_result = {k: v for k, v in result.items() if k != 'all_attempts'}
                async with lock:
                    results[str(bio_id)] = save_result
                    completed_count[0] += 1
                    if completed_count[0] % checkpoint_freq == 0:
                        _atomic_save_json(results, output_path)
                        print(f"  Checkpoint: {completed_count[0]} / {len(pending)} done")
            except Exception as e:
                print(f"  ERROR bio_id={bio_id}: {e}")

    tasks = [process_one(row) for _, row in pending.iterrows()]
    await asyncio.gather(*tasks)

    # Final save
    _atomic_save_json(results, output_path)
    print(f"Saved {len(results)} total results to {output_path}")

    return results
