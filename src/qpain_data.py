# ABOUTME: Data loading and template filling for Q-Pain race bias experiment
# ABOUTME: Loads PhysioNet CSVs, assigns racially-associated names, produces White/Black/Asian pairs

import csv
import random

import pandas as pd


DATA_DIR = "physionet.org/files/q-pain/1.0.0"

CONTEXT_FILES = {
    "acute_cancer": "data_acute_cancer.csv",
    "acute_non_cancer": "data_acute_non_cancer.csv",
    "chronic_cancer": "data_chronic_cancer.csv",
    "chronic_non_cancer": "data_chronic_non_cancer.csv",
    "post_op": "data_post_op.csv",
}

# From the original Q-Pain paper (Loge et al., 2021), Harvard Dataverse demographic data
WHITE_MALE_NAMES = [
    "Bradley", "Brett", "Scott", "Kurt", "Todd",
    "Chad", "Matthew", "Dustin", "Shane", "Douglas",
]
BLACK_MALE_NAMES = [
    "Roosevelt", "Jermaine", "Darnell", "Willie", "Mattie",
    "Reginald", "Cedric", "Sylvester", "Tyrone", "Errol",
]
ASIAN_MALE_NAMES = [
    "Viet", "Thong", "Qiang", "Kwok", "Hao",
    "Yang", "Nam", "Huy", "Yuan", "Ho",
]

FIXED_SENTENCE = "This patient record has been reviewed."


def load_qpain_vignettes(data_dir=DATA_DIR):
    """Load all Q-Pain CSVs and return only templated vignettes (with placeholders)."""
    rows = []
    for context, filename in CONTEXT_FILES.items():
        path = f"{data_dir}/{filename}"
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "[race]" in row["Vignette"]:
                    rows.append({
                        "vignette": row["Vignette"],
                        "question": row["Question"],
                        "answer": row["Answer"].strip(),
                        "context": context,
                    })
    return pd.DataFrame(rows).reset_index(drop=True)


def extract_core_question(question_text):
    """Extract the core question up to and including the first '?'."""
    idx = question_text.index("?")
    return question_text[:idx + 1]


def fill_vignette(template, race, name):
    """Replace demographic placeholders and Patient D with the given name."""
    text = template.replace("[race]", race)
    text = text.replace("[gender]", "man")
    text = text.replace("[subjective]", "he")
    text = text.replace("[subject]", "he")
    text = text.replace("[possessive]", "his")
    text = text.replace("Patient D", name)
    return text


def assign_names(df, seed=42):
    """Assign White, Black, and Asian names to each vignette by random permutation.

    Names are shuffled once and reused across all contexts (matching the
    original Q-Pain paper's code, which shuffles once before the context loop).
    """
    rng = random.Random(seed)
    white_names = list(WHITE_MALE_NAMES)
    black_names = list(BLACK_MALE_NAMES)
    asian_names = list(ASIAN_MALE_NAMES)
    rng.shuffle(white_names)
    rng.shuffle(black_names)
    rng.shuffle(asian_names)

    df = df.copy()
    df["white_name"] = ""
    df["black_name"] = ""
    df["asian_name"] = ""
    for context in df["context"].unique():
        mask = df["context"] == context
        n = mask.sum()
        df.loc[mask, "white_name"] = white_names[:n]
        df.loc[mask, "black_name"] = black_names[:n]
        df.loc[mask, "asian_name"] = asian_names[:n]
    return df


def prepare_dataset(seed=42, data_dir=DATA_DIR, paraphrase_path=None, asian_paraphrase_path=None):
    """Load vignettes, assign names, and produce all text variants for evaluation.

    Returns DataFrame with columns:
        vignette_id, context, white_text, black_text, asian_text,
        fixed_sentence_text, core_question, swap_direction,
        white_name, black_name, asian_name,
        and optionally paraphrase_text, asian_paraphrase_text
    """
    df = load_qpain_vignettes(data_dir)
    df = assign_names(df, seed=seed)
    df["vignette_id"] = range(len(df))

    df["white_text"] = df.apply(
        lambda r: fill_vignette(r["vignette"], "White", r["white_name"]), axis=1
    )
    df["black_text"] = df.apply(
        lambda r: fill_vignette(r["vignette"], "Black", r["black_name"]), axis=1
    )
    df["asian_text"] = df.apply(
        lambda r: fill_vignette(r["vignette"], "Asian", r["asian_name"]), axis=1
    )
    df["fixed_sentence_text"] = df["white_text"].apply(
        lambda t: f"{FIXED_SENTENCE} {t}"
    )
    df["core_question"] = df.apply(
        lambda r: extract_core_question(r["question"]).replace("Patient D", r["white_name"]),
        axis=1,
    )
    df["swap_direction"] = 1

    if paraphrase_path is not None:
        from qpain_paraphrase import merge_paraphrases_from_file
        df = merge_paraphrases_from_file(df, paraphrase_path)

    if asian_paraphrase_path is not None:
        from qpain_paraphrase import merge_paraphrases_from_file
        asian_paras = merge_paraphrases_from_file(
            df[["vignette_id"]], asian_paraphrase_path
        )
        df["asian_paraphrase_text"] = asian_paras["paraphrase_text"]

    return df
