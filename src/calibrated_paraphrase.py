# ABOUTME: Calibrated paraphrasing for controlled perturbation experiments
# ABOUTME: Generates paraphrases targeting specific token change percentages

REFUSAL_PHRASES = [
    "i can't help with", "i cannot help with",
    "i can't help you", "i cannot help you",
    "i can't help engineer", "i cannot help engineer",
    "i can't help generate", "i cannot help generate",
    "i can't assist", "i cannot assist",
    "i can't do what", "i cannot do what",
    "i can't do that", "i cannot do that",
    "i cannot fulfill this", "i can't fulfill this",
    "i cannot fulfill the", "i can't fulfill the",
    "i cannot fulfill your", "i can't fulfill your",
    "i can't comply with the", "i cannot comply with the",
    "i can't comply with request", "i cannot comply with request",
    "i can't produce a paraphrase", "i cannot produce a paraphrase",
    "i can't meet the requirement", "i cannot meet the requirement",
    "i can't generate a paraphrase", "i cannot generate a paraphrase",
    "as an ai", "as a language model",
]


REFUSAL_CORRECTION = "That was a refusal. Please provide only the paraphrased question, nothing else."


def is_refusal(text):
    """
    Check if text is a model refusal message.

    Args:
        text: Response text from GPT

    Returns:
        bool: True if text appears to be a refusal
    """
    normalized = text.strip().lower().replace('\u2019', "'")
    return any(normalized.startswith(phrase) for phrase in REFUSAL_PHRASES)


def select_best_undershoot(attempts, target_pct):
    """
    Select the attempt closest to target that doesn't exceed it.

    Args:
        attempts: List of attempt dicts with 'actual_pct' and 'deviation' keys
        target_pct: Target token change percentage

    Returns:
        dict: Best undershoot attempt, or closest overall if no undershoots exist
    """
    undershoots = [a for a in attempts if a['actual_pct'] <= target_pct]
    if not undershoots:
        return min(attempts, key=lambda x: x['deviation'])
    return min(undershoots, key=lambda x: x['deviation'])


INITIAL_PROMPT_TEMPLATE = """You are tasked with paraphrasing a medical question.

TARGET: Change exactly {target_pct:.1f}% of tokens (±0.5% tolerance).

HOW WE MEASURE:
We tokenize both texts using a language model tokenizer, then calculate:
  edit_distance(original_tokens, paraphrased_tokens) / len(original_tokens) × 100
Edit distance counts insertions, deletions, and substitutions.

CRITICAL REQUIREMENTS:
- The paraphrase MUST be semantically equivalent to the original — same meaning, same information, same intent
- Preserve ALL medical terminology EXACTLY (drug names, diagnoses, procedures, anatomical terms)
- Preserve ALL factual/clinical information EXACTLY (numbers, dates, test results, symptoms)
- You MUST achieve approximately {target_pct:.1f}% token change

Original question ({orig_token_count} tokens):
{question}

Target edits: ~{target_edits} token operations (insertions, deletions, or substitutions)

Provide ONLY the paraphrased question, nothing else."""


def format_initial_prompt(question, target_pct, orig_token_count, target_edits):
    """
    Format the initial calibrated paraphrasing prompt.

    Args:
        question: Original question text
        target_pct: Target token change percentage
        orig_token_count: Number of tokens in original
        target_edits: Target number of edit operations

    Returns:
        str: Formatted prompt
    """
    return INITIAL_PROMPT_TEMPLATE.format(
        question=question,
        target_pct=target_pct,
        orig_token_count=orig_token_count,
        target_edits=target_edits
    )


RETRY_PROMPT_TEMPLATE = """That changed {actual_pct:.1f}% of tokens. Target is {target_pct:.1f}% (±0.5%).
{direction_hint}
Try again."""


def format_retry_prompt(actual_pct, target_pct):
    """
    Format the retry prompt for calibrated paraphrasing.

    Args:
        actual_pct: Actual token change percentage achieved
        target_pct: Target token change percentage

    Returns:
        str: Formatted retry prompt
    """
    if actual_pct > target_pct + 0.5:
        direction_hint = "Make fewer changes. Your paraphrase was too different."
    else:
        direction_hint = "Make more changes. Your paraphrase was too similar."

    return RETRY_PROMPT_TEMPLATE.format(
        actual_pct=actual_pct,
        target_pct=target_pct,
        direction_hint=direction_hint
    )


RESET_HINT_TEMPLATE = """

Note: A previous attempt achieved {best_actual_pct:.1f}% token change.
You need to make {direction} changes to reach the target."""


def format_reset_hint(best_actual_pct, target_pct):
    """
    Format hint about best previous attempt for conversation reset.

    Args:
        best_actual_pct: Actual percentage achieved by best previous attempt
        target_pct: Target token change percentage

    Returns:
        str: Formatted hint to append to initial prompt
    """
    direction = "more" if best_actual_pct < target_pct else "fewer"
    return RESET_HINT_TEMPLATE.format(
        best_actual_pct=best_actual_pct,
        direction=direction
    )


SYNONYM_PROMPT_TEMPLATE = """Paraphrase the following medical text by replacing approximately {num_words} words with synonyms.

CRITICAL RULES:
- Achieve the paraphrase ONLY through synonym replacements
- Keep ALL other text EXACTLY the same — same structure, same formatting, same punctuation
- Do NOT rewrite, restructure, or add/remove content
- Preserve ALL medical terminology, numbers, dates, and proper nouns

Text:
{question}

Provide ONLY the paraphrased text, nothing else."""


def format_synonym_prompt(question, num_words):
    """
    Format a synonym replacement prompt for small target changes.

    Args:
        question: Original text
        num_words: Number of words to replace with synonyms

    Returns:
        str: Formatted prompt
    """
    return SYNONYM_PROMPT_TEMPLATE.format(question=question, num_words=num_words)


def should_switch_to_synonym_strategy(attempts, target_pct, tolerance=0.5):
    """
    Detect when all paraphrasing attempts overshot beyond tolerance.

    Args:
        attempts: List of attempt dicts with 'actual_pct' key
        target_pct: Target token change percentage
        tolerance: Acceptable deviation from target

    Returns:
        bool: True if every attempt overshoots by more than tolerance
    """
    if not attempts:
        return False
    return all(a['actual_pct'] > target_pct + tolerance for a in attempts)


from token_edit_distance import token_edit_distance_percent


def generate_calibrated_paraphrase(question, target_pct, tokenizer, openai_client,
                                   max_retries=50, tolerance=0.5):
    """
    Generate a paraphrase targeting a specific token change percentage.

    Uses multi-turn conversation with GPT to iteratively adjust until
    the target percentage is achieved or max retries reached.

    Args:
        question: Original question text
        target_pct: Target token change percentage
        tokenizer: HuggingFace tokenizer for measuring token changes
        openai_client: OpenAI client for GPT API calls
        max_retries: Maximum number of retry attempts (default 50)
        tolerance: Acceptable deviation from target in percentage points (default 0.5)

    Returns:
        dict: {
            'paraphrase': str - best paraphrase found,
            'actual_pct': float - actual token change percentage,
            'retries_used': int - number of retries (0 if first attempt succeeded),
            'deviation': float - absolute deviation from target,
            'all_attempts': list - all attempts with their percentages
        }
    """
    # Calculate target edits
    orig_tokens = tokenizer.encode(question, add_special_tokens=False)
    orig_token_count = len(orig_tokens)
    target_edits = max(1, round(target_pct * orig_token_count / 100))

    # Build conversation
    initial_prompt = format_initial_prompt(
        question=question,
        target_pct=target_pct,
        orig_token_count=orig_token_count,
        target_edits=target_edits
    )

    messages = [{"role": "user", "content": initial_prompt}]
    attempts = []

    for attempt_num in range(max_retries + 1):
        # Conversation reset every 10 retries
        if attempt_num > 0 and attempt_num % 10 == 0:
            if attempts:
                best_so_far = min(attempts, key=lambda x: x['deviation'])
                hint = format_reset_hint(best_so_far['actual_pct'], target_pct)
                messages = [{"role": "user", "content": initial_prompt + hint}]
            else:
                messages = [{"role": "user", "content": initial_prompt}]

        # Call GPT
        response = openai_client.responses.create(
            model="gpt-5.2",
            input=messages
        )
        paraphrase = response.output_text.strip()

        # Remove quotes if present
        if paraphrase.startswith('"') and paraphrase.endswith('"'):
            paraphrase = paraphrase[1:-1]
        if paraphrase.startswith("'") and paraphrase.endswith("'"):
            paraphrase = paraphrase[1:-1]

        # Refusal detection
        if is_refusal(paraphrase):
            if attempt_num < max_retries:
                messages.append({"role": "assistant", "content": paraphrase})
                messages.append({"role": "user", "content": REFUSAL_CORRECTION})
            continue

        # Calculate actual percentage
        actual_pct = token_edit_distance_percent(question, paraphrase, tokenizer)
        deviation = abs(actual_pct - target_pct)

        attempts.append({
            'paraphrase': paraphrase,
            'actual_pct': actual_pct,
            'deviation': deviation
        })

        # Check if within tolerance
        if deviation <= tolerance:
            return {
                'paraphrase': paraphrase,
                'actual_pct': actual_pct,
                'retries_used': attempt_num,
                'deviation': deviation,
                'all_attempts': attempts
            }

        # If not last attempt, add retry feedback to conversation
        if attempt_num < max_retries:
            messages.append({"role": "assistant", "content": paraphrase})
            retry_prompt = format_retry_prompt(actual_pct, target_pct)
            messages.append({"role": "user", "content": retry_prompt})

    if not attempts:
        print(f"  WARNING: All {max_retries + 1} paraphrasing attempts were refusals — using original question as baseline")
        return {
            'paraphrase': question,
            'actual_pct': 0.0,
            'retries_used': max_retries,
            'deviation': target_pct,
            'all_attempts': []
        }

    # Second pass: synonym replacement when all paraphrase attempts overshot
    if should_switch_to_synonym_strategy(attempts, target_pct, tolerance):
        num_words = max(1, target_edits)
        synonym_prompt = format_synonym_prompt(question, num_words)

        for attempt_num in range(max_retries + 1):
            response = openai_client.responses.create(
                model="gpt-5.2",
                input=[{"role": "user", "content": synonym_prompt}]
            )
            paraphrase = response.output_text.strip()

            if paraphrase.startswith('"') and paraphrase.endswith('"'):
                paraphrase = paraphrase[1:-1]
            if paraphrase.startswith("'") and paraphrase.endswith("'"):
                paraphrase = paraphrase[1:-1]

            if is_refusal(paraphrase):
                continue

            actual_pct = token_edit_distance_percent(question, paraphrase, tokenizer)
            deviation = abs(actual_pct - target_pct)

            attempts.append({
                'paraphrase': paraphrase,
                'actual_pct': actual_pct,
                'deviation': deviation
            })

            if deviation <= tolerance:
                return {
                    'paraphrase': paraphrase,
                    'actual_pct': actual_pct,
                    'retries_used': max_retries + attempt_num + 1,
                    'deviation': deviation,
                    'all_attempts': attempts
                }

    best_attempt = select_best_undershoot(attempts, target_pct)

    return {
        'paraphrase': best_attempt['paraphrase'],
        'actual_pct': best_attempt['actual_pct'],
        'retries_used': max_retries,
        'deviation': best_attempt['deviation'],
        'all_attempts': attempts
    }


async def generate_calibrated_paraphrase_async(question, target_pct, tokenizer, openai_client,
                                                max_retries=50, tolerance=0.5):
    """
    Async version of generate_calibrated_paraphrase.

    Generate a paraphrase targeting a specific token change percentage.

    Uses multi-turn conversation with GPT to iteratively adjust until
    the target percentage is achieved or max retries reached.

    Args:
        question: Original question text
        target_pct: Target token change percentage
        tokenizer: HuggingFace tokenizer for measuring token changes
        openai_client: AsyncOpenAI client for GPT API calls
        max_retries: Maximum number of retry attempts (default 50)
        tolerance: Acceptable deviation from target in percentage points (default 0.5)

    Returns:
        dict: {
            'paraphrase': str - best paraphrase found,
            'actual_pct': float - actual token change percentage,
            'retries_used': int - number of retries (0 if first attempt succeeded),
            'deviation': float - absolute deviation from target,
            'all_attempts': list - all attempts with their percentages
        }
    """
    # Calculate target edits
    orig_tokens = tokenizer.encode(question, add_special_tokens=False)
    orig_token_count = len(orig_tokens)
    target_edits = max(1, round(target_pct * orig_token_count / 100))

    # Build conversation
    initial_prompt = format_initial_prompt(
        question=question,
        target_pct=target_pct,
        orig_token_count=orig_token_count,
        target_edits=target_edits
    )

    messages = [{"role": "user", "content": initial_prompt}]
    attempts = []

    for attempt_num in range(max_retries + 1):
        # Conversation reset every 10 retries
        if attempt_num > 0 and attempt_num % 10 == 0:
            if attempts:
                best_so_far = min(attempts, key=lambda x: x['deviation'])
                hint = format_reset_hint(best_so_far['actual_pct'], target_pct)
                messages = [{"role": "user", "content": initial_prompt + hint}]
            else:
                messages = [{"role": "user", "content": initial_prompt}]

        # Call GPT
        response = await openai_client.responses.create(
            model="gpt-5.2",
            input=messages
        )
        paraphrase = response.output_text.strip()

        # Remove quotes if present
        if paraphrase.startswith('"') and paraphrase.endswith('"'):
            paraphrase = paraphrase[1:-1]
        if paraphrase.startswith("'") and paraphrase.endswith("'"):
            paraphrase = paraphrase[1:-1]

        # Refusal detection
        if is_refusal(paraphrase):
            if attempt_num < max_retries:
                messages.append({"role": "assistant", "content": paraphrase})
                messages.append({"role": "user", "content": REFUSAL_CORRECTION})
            continue

        # Calculate actual percentage
        actual_pct = token_edit_distance_percent(question, paraphrase, tokenizer)
        deviation = abs(actual_pct - target_pct)

        attempts.append({
            'paraphrase': paraphrase,
            'actual_pct': actual_pct,
            'deviation': deviation
        })

        # Check if within tolerance
        if deviation <= tolerance:
            return {
                'paraphrase': paraphrase,
                'actual_pct': actual_pct,
                'retries_used': attempt_num,
                'deviation': deviation,
                'all_attempts': attempts
            }

        # If not last attempt, add retry feedback to conversation
        if attempt_num < max_retries:
            messages.append({"role": "assistant", "content": paraphrase})
            retry_prompt = format_retry_prompt(actual_pct, target_pct)
            messages.append({"role": "user", "content": retry_prompt})

    if not attempts:
        print(f"  WARNING: All {max_retries + 1} paraphrasing attempts were refusals — using original question as baseline")
        return {
            'paraphrase': question,
            'actual_pct': 0.0,
            'retries_used': max_retries,
            'deviation': target_pct,
            'all_attempts': []
        }

    # Second pass: synonym replacement when all paraphrase attempts overshot
    if should_switch_to_synonym_strategy(attempts, target_pct, tolerance):
        num_words = max(1, target_edits)
        synonym_prompt = format_synonym_prompt(question, num_words)

        for attempt_num in range(max_retries + 1):
            response = await openai_client.responses.create(
                model="gpt-5.2",
                input=[{"role": "user", "content": synonym_prompt}]
            )
            paraphrase = response.output_text.strip()

            if paraphrase.startswith('"') and paraphrase.endswith('"'):
                paraphrase = paraphrase[1:-1]
            if paraphrase.startswith("'") and paraphrase.endswith("'"):
                paraphrase = paraphrase[1:-1]

            if is_refusal(paraphrase):
                continue

            actual_pct = token_edit_distance_percent(question, paraphrase, tokenizer)
            deviation = abs(actual_pct - target_pct)

            attempts.append({
                'paraphrase': paraphrase,
                'actual_pct': actual_pct,
                'deviation': deviation
            })

            if deviation <= tolerance:
                return {
                    'paraphrase': paraphrase,
                    'actual_pct': actual_pct,
                    'retries_used': max_retries + attempt_num + 1,
                    'deviation': deviation,
                    'all_attempts': attempts
                }

    best_attempt = select_best_undershoot(attempts, target_pct)

    return {
        'paraphrase': best_attempt['paraphrase'],
        'actual_pct': best_attempt['actual_pct'],
        'retries_used': max_retries,
        'deviation': best_attempt['deviation'],
        'all_attempts': attempts
    }
