# ABOUTME: Data loading, balanced set construction, and gender swap for bias-in-bios experiment
# ABOUTME: Implements balanced professor/nurse subset following Marks et al. 2024

import re

import pandas as pd


PROFESSOR_ID = 21
NURSE_ID = 13
MALE = 0
FEMALE = 1


def load_bias_in_bios(split):
    """Load bias_in_bios dataset from HuggingFace for the given split."""
    from datasets import load_dataset
    ds = load_dataset("LabHC/bias_in_bios", split=split)
    return pd.DataFrame(ds)


def build_balanced_set(df, seed=42):
    """Construct balanced subset with equal counts per (profession, gender) group.

    Follows Marks et al. 2024: n = min(count of all 4 groups).
    """
    groups = {
        (PROFESSOR_ID, MALE): df[(df["profession"] == PROFESSOR_ID) & (df["gender"] == MALE)],
        (PROFESSOR_ID, FEMALE): df[(df["profession"] == PROFESSOR_ID) & (df["gender"] == FEMALE)],
        (NURSE_ID, MALE): df[(df["profession"] == NURSE_ID) & (df["gender"] == MALE)],
        (NURSE_ID, FEMALE): df[(df["profession"] == NURSE_ID) & (df["gender"] == FEMALE)],
    }
    n = min(len(g) for g in groups.values())
    sampled = [g.sample(n=n, random_state=seed) for g in groups.values()]
    result = pd.concat(sampled, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


# Explicit mappings in both directions. NOT auto-reversed because some mappings
# are asymmetric: "His"→"Her" (possessive) and "Him"→"Her" (object) both map to
# "Her", which cannot be recovered by inverting a single dict.
FEMALE_TO_MALE = {
    # Titles
    "Ms.": "Mr.", "ms.": "mr.", "Ms": "Mr", "ms": "mr",
    "Mrs.": "Mr.", "mrs.": "mr.", "Mrs": "Mr", "mrs": "mr",
    "Miss": "Mr.",
    "Ma'am": "Sir", "ma'am": "sir",
    "Madam": "Sir", "madam": "sir",
    # Pronouns
    "She": "He", "she": "he",
    "Her": "His", "her": "his",
    "Hers": "His", "hers": "his",
    "Herself": "Himself", "herself": "himself",
    # Contractions
    "She's": "He's", "she's": "he's",
    "She'd": "He'd", "she'd": "he'd",
    "She'll": "He'll", "she'll": "he'll",
    # Descriptors
    "Female": "Male", "female": "male",
    "Woman": "Man", "woman": "man",
    "Women": "Men", "women": "men",
    "Girl": "Boy", "girl": "boy",
    "Lady": "Gentleman", "lady": "gentleman",
    "Ladies": "Gentlemen", "ladies": "gentlemen",
    # Family
    "Mother": "Father", "mother": "father",
    "Daughter": "Son", "daughter": "son",
    "Sister": "Brother", "sister": "brother",
    "Wife": "Husband", "wife": "husband",
    "Girlfriend": "Boyfriend", "girlfriend": "boyfriend",
    "Grandmother": "Grandfather", "grandmother": "grandfather",
    "Granddaughter": "Grandson", "granddaughter": "grandson",
    "Aunt": "Uncle", "aunt": "uncle",
    "Niece": "Nephew", "niece": "nephew",
    # Professional
    "Actress": "Actor", "actress": "actor",
    "Waitress": "Waiter", "waitress": "waiter",
    "Businesswoman": "Businessman", "businesswoman": "businessman",
    "Spokeswoman": "Spokesman", "spokeswoman": "spokesman",
    "Congresswoman": "Congressman", "congresswoman": "congressman",
    "Chairwoman": "Chairman", "chairwoman": "chairman",
    "Heroine": "Hero", "heroine": "hero",
}

MALE_TO_FEMALE = {
    # Titles
    "Mr.": "Ms.", "mr.": "ms.", "Mr": "Ms", "mr": "ms",
    "Sir": "Ma'am", "sir": "ma'am",
    # Pronouns — "His"→"Her" (possessive) and "Him"→"Her" (object) are both
    # explicit to avoid the auto-reverse collision where "His" would become "Hers".
    "He": "She", "he": "she",
    "His": "Her", "his": "her",
    "Him": "Her", "him": "her",
    "Himself": "Herself", "himself": "herself",
    # Contractions
    "He's": "She's", "he's": "she's",
    "He'd": "She'd", "he'd": "she'd",
    "He'll": "She'll", "he'll": "she'll",
    # Descriptors
    "Male": "Female", "male": "female",
    "Man": "Woman", "man": "woman",
    "Men": "Women", "men": "women",
    "Boy": "Girl", "boy": "girl",
    "Gentleman": "Lady", "gentleman": "lady",
    "Gentlemen": "Ladies", "gentlemen": "ladies",
    # Family
    "Father": "Mother", "father": "mother",
    "Son": "Daughter", "son": "daughter",
    "Brother": "Sister", "brother": "sister",
    "Husband": "Wife", "husband": "wife",
    "Boyfriend": "Girlfriend", "boyfriend": "girlfriend",
    "Grandfather": "Grandmother", "grandfather": "grandmother",
    "Grandson": "Granddaughter", "grandson": "granddaughter",
    "Uncle": "Aunt", "uncle": "aunt",
    "Nephew": "Niece", "nephew": "niece",
    # Professional
    "Actor": "Actress", "actor": "actress",
    "Waiter": "Waitress", "waiter": "waitress",
    "Businessman": "Businesswoman", "businessman": "businesswoman",
    "Spokesman": "Spokeswoman", "spokesman": "spokeswoman",
    "Congressman": "Congresswoman", "congressman": "congresswoman",
    "Chairman": "Chairwoman", "chairman": "chairwoman",
    "Hero": "Heroine", "hero": "heroine",
}


# Terms that should NOT remain after swapping from the given gender
MALE_INDICATORS = [r'\bhe\b', r'\bhis\b', r'\bhim\b', r'\bhimself\b', r'\bMr\.\b',
                   r"\bhe's\b", r"\bhe'd\b", r"\bhe'll\b"]
FEMALE_INDICATORS = [r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bherself\b', r'\bMs\.\b', r'\bMrs\.\b',
                     r"\bshe's\b", r"\bshe'd\b", r"\bshe'll\b"]
# Note: period-less titles (Mr, Ms, Mrs) are handled in the swap mappings
# but not in validation indicators because case-insensitive matching would
# false-positive on "MS" (Master of Science) and "MS" (Mississippi).


def apply_gender_swap(text, original_gender):
    """Replace all gendered terms in text with their opposite-gender equivalents.

    Longer terms are replaced first to prevent substring collisions
    (e.g., "Grandmother" before "Mother").
    """
    mapping = FEMALE_TO_MALE if original_gender == FEMALE else MALE_TO_FEMALE
    for source in sorted(mapping, key=len, reverse=True):
        escaped = re.escape(source)
        # \b after a period fails because '.' is non-word and the next char
        # (space/EOL) is also non-word, so no boundary exists. For terms ending
        # in '.', the period itself is a sufficient delimiter — only assert \b
        # at the start.
        if source.endswith("."):
            pattern = r"\b" + escaped
        else:
            pattern = r"\b" + escaped + r"\b"
        text = re.sub(pattern, mapping[source], text)
    return text


def validate_gender_swap(original, swapped, original_gender):
    """Check for residual gendered terms after a swap.
    Returns list of issue descriptions. Empty list = clean swap.
    """
    residual_patterns = MALE_INDICATORS if original_gender == MALE else FEMALE_INDICATORS
    issues = []
    for pattern in residual_patterns:
        matches = re.findall(pattern, swapped, re.IGNORECASE)
        if matches:
            gender_label = 'male' if original_gender == MALE else 'female'
            issues.append(f"Residual term '{matches[0]}' found after swapping from {gender_label}")
    return issues


FIXED_SENTENCE = "This profile has been viewed"


def apply_fixed_sentence(bio_text):
    """Prepend the fixed sentence baseline to a bio."""
    return f"{FIXED_SENTENCE} {bio_text}"


def get_swap_direction(original_gender):
    """Encode gender swap direction for regression.
    +1 = swapped to female (original was male)
    -1 = swapped to male (original was female)
    """
    return 1 if original_gender == MALE else -1


_SPAM_PATTERNS = re.compile(
    r"\bfucks?\b|\bfucking\b|\bpussy\b|\bcock\s|\bcocks\b"
    r"|\bporn\b|\bxxx\b|\bnudes?\s|\borgasm|\bdildo|\bblowjob|\bcunt\b"
    r"|xhamster|pornunlim|alphaporno|porn\s+tube"
    r"|\bpantyhose\s+ripped|\bbreasts\s+massage|\bbig\s+tits\b",
    re.IGNORECASE,
)


def _is_spam(text):
    """Detect pornographic spam that was misclassified as a professional bio."""
    return bool(_SPAM_PATTERNS.search(text))


def _has_gendered_language(text, gender):
    """Check if a bio contains any gendered language that can be swapped."""
    mapping = FEMALE_TO_MALE if gender == FEMALE else MALE_TO_FEMALE
    for source in mapping:
        if source.endswith("."):
            pattern = r"\b" + re.escape(source)
        else:
            pattern = r"\b" + re.escape(source) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def prepare_split(split, seed=42, paraphrase_path=None):
    """Load, balance, and apply gender swap for one split.

    Filters out bios with no gendered language before balancing, so only
    bios where the swap produces an actual change are included.

    Args:
        split: 'train' or 'test'
        seed: random seed for balanced sampling
        paraphrase_path: optional path to paraphrase JSON.
            If provided, merges paraphrase_text into the DataFrame.

    Returns DataFrame with columns:
        bio_id, hard_text, profession, gender, swapped_text,
        fixed_sentence_text, swap_direction, and optionally paraphrase_text
    """
    df = load_bias_in_bios(split)
    # Fix known typo in bias_in_bios dataset: "SHe" → "She" (word boundary only)
    df["hard_text"] = df["hard_text"].str.replace(r"\bSHe\b", "She", regex=True)
    # Filter out spam and bios without gendered language
    is_clean = ~df["hard_text"].apply(_is_spam)
    has_gender = df.apply(lambda r: _has_gendered_language(r["hard_text"], r["gender"]), axis=1)
    df_filtered = df[is_clean & has_gender].copy()
    balanced = build_balanced_set(df_filtered, seed=seed)
    balanced["bio_id"] = range(len(balanced))
    balanced["swapped_text"] = balanced.apply(
        lambda r: apply_gender_swap(r["hard_text"], r["gender"]), axis=1
    )
    balanced["fixed_sentence_text"] = balanced["hard_text"].apply(apply_fixed_sentence)
    balanced["swap_direction"] = balanced["gender"].apply(get_swap_direction)

    if paraphrase_path is not None:
        from bib_paraphrase import merge_paraphrases_from_file
        balanced = merge_paraphrases_from_file(balanced, paraphrase_path)

    return balanced


def validate_all_swaps(df):
    """Run validation on all bios. df must have 'hard_text', 'gender', 'swapped_text' columns."""
    total = len(df)
    issue_count = 0
    issue_details = []
    for _, row in df.iterrows():
        issues = validate_gender_swap(row["hard_text"], row["swapped_text"], row["gender"])
        if issues:
            issue_count += 1
            issue_details.append({"index": row.name, "issues": issues})
    return {"total": total, "clean": total - issue_count, "issues": issue_count, "details": issue_details}
