# ABOUTME: Age bracket swap for precision sanity check
# ABOUTME: Extracts and replaces patient age in clinical text across 9+ formats

import os
import re
import sys
import json
import random
import hashlib
import argparse

import pandas as pd


# Ordered by specificity — most specific patterns first to avoid partial matches
AGE_PATTERNS = [
    # "in my late/early/mid 30s" format
    (r'\b((?:late|early|mid)\s+\d0s)\b', 'decade'),
    # "19 year(s) old" / "19-year-old" / "19 year-old"
    (r'\b(\d{1,3}\s*-?\s*years?\s*-?\s*old)\b', 'year_old'),
    # "26(M)" / "26(f)" paren format — must come before compact_gender
    (r'\b(\d{1,3})\(([MFmf])\)', 'paren_gender'),
    # "F24" / "M17" / "m17" — gender letter before digits
    (r'\b([MFmf])(\d{1,3})\b', 'gender_prefix'),
    # "28M" / "35f" / "25 M" compact (digit optionally followed by space then M/F/m/f)
    (r'\b(\d{1,3})(\s?)([MFmf])\b', 'compact_gender'),
    # "Age = 28" format (may be followed by M/F)
    (r'[Aa]ge\s*[=:]\s*(\d{1,3})', 'age_equals'),
    # "Female, 23" / "Male, 45" — gender then comma then age
    (r'(?:[Ff]emale|[Mm]ale)\s*,\s*(\d{1,3})\b', 'gender_comma_age'),
    # "21 Male" / "21 Female"
    (r'\b(\d{1,3})\s+(?:[Mm]ale|[Ff]emale)\b', 'age_gender'),
    # "28 this year" — age followed by "this year"
    (r'\b(\d{1,3})\s+this\s+year\b', 'this_year'),
    # "I'm 25" / "I am 30" / "I'm a 19" — pronoun + age (first-person)
    (r"(?:I'?m|I am)\s+(?:a\s+)?(\d{1,3})\b", 'im_age'),
]

# Minimum plausible patient age — patterns require gender/age context, so
# low ages (e.g. 3M, 7 year old) are real pediatric patients, not noise
MIN_AGE = 1
MAX_AGE = 110


def extract_age(text):
    """Extract patient age from clinical text.

    Tries patterns in order of specificity, returning the first match
    with a plausible age value (1-110).

    Args:
        text: Clinical context string

    Returns:
        tuple: (age_int, matched_string, match_start) or None if no age found.
            match_start is the character offset of matched_string in text.
    """
    for pattern, pat_type in AGE_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue

        if pat_type == 'decade':
            decade_str = match.group(1)
            decade_match = re.search(r'(\d)0s', decade_str)
            decade = int(decade_match.group(1)) * 10
            if 'early' in decade_str:
                age = decade + 2
            elif 'late' in decade_str:
                age = decade + 7
            else:  # mid
                age = decade + 5
            if MIN_AGE <= age <= MAX_AGE:
                return (age, decade_str, match.start(1))

        elif pat_type == 'year_old':
            full_match = match.group(0)
            age_str = re.match(r'(\d+)', full_match).group(1)
            age = int(age_str)
            if MIN_AGE <= age <= MAX_AGE:
                return (age, full_match, match.start())

        elif pat_type == 'paren_gender':
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(0), match.start())

        elif pat_type == 'gender_prefix':
            age = int(match.group(2))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(0), match.start())

        elif pat_type == 'compact_gender':
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(0), match.start())

        elif pat_type == 'age_equals':
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(1), match.start(1))

        elif pat_type in ('gender_comma_age', 'age_gender', 'this_year', 'im_age'):
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return (age, match.group(1), match.start(1))

    return None


def compute_target_age(original_age, seed):
    """Compute target age via bracket swap with randomness.

    Under 50 -> random in [60, 80], >=50 -> random in [18, 35].
    Seeded by input for reproducibility.

    Args:
        original_age: Original patient age
        seed: Integer seed for deterministic randomness

    Returns:
        int: Target age (guaranteed >=10 year change)
    """
    rng = random.Random(seed)

    if original_age < 50:
        return rng.randint(60, 80)
    else:
        return rng.randint(18, 35)


def replace_age(text, matched_string, new_age, match_start=None):
    """Replace the matched age string with new age, preserving format.

    Args:
        text: Original clinical text
        matched_string: The exact string matched by extract_age
        new_age: Integer target age
        match_start: Character offset of matched_string in text (from extract_age).
            Used for positional replacement when matched_string is a bare number
            that could appear elsewhere in the text.

    Returns:
        str: Text with age replaced
    """
    # Decade format: "late 30s" / "early 20s" -> just the number
    if re.match(r'(?:late|early|mid)\s+\d0s', matched_string):
        return text.replace(matched_string, str(new_age), 1)

    # Paren gender format: "26(M)" / "26(f)" -> "65(M)" / "65(f)"
    paren_match = re.match(r'^(\d+)\(([MFmf])\)$', matched_string)
    if paren_match:
        gender_char = paren_match.group(2)
        return text.replace(matched_string, f"{new_age}({gender_char})", 1)

    # Gender prefix format: "F24" / "M17" -> "F65" / "M65"
    prefix_match = re.match(r'^([MFmf])(\d+)$', matched_string)
    if prefix_match:
        gender_char = prefix_match.group(1)
        return text.replace(matched_string, f"{gender_char}{new_age}", 1)

    # Compact gender format: "28M" / "35f" / "25 M" -> "65M" / "65f" / "65 M"
    compact_match = re.match(r'^(\d+)(\s?)([MFmf])$', matched_string)
    if compact_match:
        space = compact_match.group(2)
        gender_char = compact_match.group(3)
        return text.replace(matched_string, f"{new_age}{space}{gender_char}", 1)

    # Year-old format: "19 year old" -> "65 year old", "35-year-old" -> "20-year-old"
    year_old_match = re.match(r'^(\d+)(\s*-?\s*years?\s*-?\s*old)$', matched_string)
    if year_old_match:
        suffix = year_old_match.group(2)
        return text.replace(matched_string, f"{new_age}{suffix}", 1)

    # Numeric only: positional replacement to avoid replacing wrong occurrence
    if match_start is not None:
        end = match_start + len(matched_string)
        return text[:match_start] + str(new_age) + text[end:]
    return text.replace(matched_string, str(new_age), 1)


def context_id_to_seed(context_id):
    """Derive a deterministic seed from context_id string.

    Args:
        context_id: String identifier (e.g., "N75")

    Returns:
        int: Positive integer seed
    """
    return int(hashlib.md5(context_id.encode()).hexdigest(), 16) % (2**31)


def apply_age_perturbation(dataset_path, compute_new_age, tokenizer=None):
    """Apply an age perturbation to all original non-conversational samples.

    Args:
        dataset_path: Path to data_with_baselines.csv
        compute_new_age: Callable(original_age, context_id) -> int
        tokenizer: Optional HuggingFace tokenizer for token edit distance

    Returns:
        list of dicts with keys: context_id, dataset, original_text,
            age_swapped_text, original_age, new_age, token_change_pct,
            age_extraction_failed
    """
    # Late import to avoid requiring src/ in path at module load time
    if tokenizer is not None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        from token_edit_distance import token_edit_distance_percent

    df = pd.read_csv(dataset_path)
    df = df[df['dataset'] != 'conversational']
    df = df[df['dataset_id'] == 1]

    results = []
    for _, row in df.iterrows():
        context_id = row['context_id']
        text = row['clinical_context']
        dataset = row['dataset']

        extraction = extract_age(text)

        if extraction is None:
            results.append({
                'context_id': context_id,
                'dataset': dataset,
                'original_text': text,
                'age_swapped_text': None,
                'original_age': None,
                'new_age': None,
                'token_change_pct': None,
                'age_extraction_failed': True,
            })
            continue

        original_age, matched_string, match_start = extraction
        new_age = compute_new_age(original_age, context_id)
        swapped_text = replace_age(text, matched_string, new_age, match_start=match_start)
        # Replace any remaining occurrences of the original age (e.g., in EHR headers
        # or body text that the primary pattern-based replacement missed).
        # Negative lookahead (?!\.\d) avoids corrupting decimal numbers like "27.0" in lab values.
        swapped_text = re.sub(r'\b' + str(original_age) + r'\b(?!\.\d)', str(new_age), swapped_text)

        token_change_pct = None
        if tokenizer is not None:
            token_change_pct = token_edit_distance_percent(text, swapped_text, tokenizer)

        results.append({
            'context_id': context_id,
            'dataset': dataset,
            'original_text': text,
            'age_swapped_text': swapped_text,
            'original_age': original_age,
            'new_age': new_age,
            'token_change_pct': token_change_pct,
            'age_extraction_failed': False,
        })

    return results


def run_age_swap(dataset_path, tokenizer=None):
    """Apply age bracket swap to all original non-conversational samples.

    Args:
        dataset_path: Path to data_with_baselines.csv
        tokenizer: Optional HuggingFace tokenizer for token edit distance

    Returns:
        list of dicts with keys: context_id, dataset, original_text,
            age_swapped_text, original_age, new_age, token_change_pct,
            age_extraction_failed
    """
    def bracket_swap(original_age, context_id):
        seed = context_id_to_seed(context_id)
        return compute_target_age(original_age, seed)
    return apply_age_perturbation(dataset_path, bracket_swap, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Age bracket swap for precision sanity check")
    parser.add_argument('--dataset', type=str, default='data_with_baselines.csv',
                        help='Path to data_with_baselines.csv')
    parser.add_argument('--output', type=str, default='results/precision_check_age_swap.json',
                        help='Output JSON path')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer (token edit distance)')
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = run_age_swap(args.dataset, tokenizer=tokenizer)

    # Print stats
    total = len(results)
    failed = sum(1 for r in results if r['age_extraction_failed'])
    successful = total - failed
    print(f"Total samples: {total}")
    print(f"Successful age swaps: {successful}")
    print(f"Failed age extraction: {failed}")

    if successful > 0:
        pcts = [r['token_change_pct'] for r in results if r['token_change_pct'] is not None]
        if pcts:
            print(f"Token change %: mean={sum(pcts)/len(pcts):.2f}, "
                  f"min={min(pcts):.2f}, max={max(pcts):.2f}")

    # Save JSON (exclude original_text to save space — can re-join by context_id)
    output_records = []
    for r in results:
        record = {k: v for k, v in r.items() if k != 'original_text'}
        output_records.append(record)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_records, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
