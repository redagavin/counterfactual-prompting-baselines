# ABOUTME: Load DiscrimEval explicit subset, select reference + single-axis contrasts
# ABOUTME: Build yes/no decision prompts and merge token-matched paraphrases

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_edit_distance import token_edit_distance_percent

REFERENCE = (60.0, "white", "male")  # (age, race, gender) — Anthropic's reference

# 14 single-axis contrasts vs the white-60yo-male reference.
# key, age, race, gender
CONTRASTS = [
    ("gender_female",        60.0, "white", "female"),
    ("gender_non_binary",    60.0, "white", "non-binary"),
    ("race_Black",           60.0, "Black", "male"),
    ("race_Asian",           60.0, "Asian", "male"),
    ("race_Hispanic",        60.0, "Hispanic", "male"),
    ("race_Native_American", 60.0, "Native American", "male"),
    ("age_20",  20.0, "white", "male"),
    ("age_30",  30.0, "white", "male"),
    ("age_40",  40.0, "white", "male"),
    ("age_50",  50.0, "white", "male"),
    ("age_70",  70.0, "white", "male"),
    ("age_80",  80.0, "white", "male"),
    ("age_90",  90.0, "white", "male"),
    ("age_100", 100.0, "white", "male"),
]

AXIS_OF = {  # map contrast key -> axis name, for per-axis summaries
    **{k: "gender" for k, *_ in CONTRASTS if k.startswith("gender")},
    **{k: "race" for k, *_ in CONTRASTS if k.startswith("race")},
    **{k: "age" for k, *_ in CONTRASTS if k.startswith("age")},
}


def load_discrimeval_explicit():
    """Load the DiscrimEval explicit subset from HuggingFace as a DataFrame."""
    import pandas as pd
    from datasets import load_dataset
    ds = load_dataset("Anthropic/discrim-eval", "explicit", split="train")
    return pd.DataFrame(ds)


def _row_template(df, qid, age, race, gender):
    """Return the filled_template for one (scenario, demographic) row, or None."""
    m = df[(df["decision_question_id"] == qid) & (df["age"] == age)
           & (df["race"] == race) & (df["gender"] == gender)]
    if len(m) == 0:
        return None
    return m.iloc[0]["filled_template"]


def select_samples(df):
    """Build per-scenario reference + 14 contrast templates by selecting rows."""
    ref_age, ref_race, ref_gender = REFERENCE
    samples = []
    for qid in sorted(df["decision_question_id"].unique()):
        ref_text = _row_template(df, qid, ref_age, ref_race, ref_gender)
        if ref_text is None:
            continue
        contrasts = {}
        ok = True
        for key, age, race, gender in CONTRASTS:
            txt = _row_template(df, qid, age, race, gender)
            if txt is None:
                ok = False
                break
            contrasts[key] = txt
        if ok:
            samples.append({
                "decision_question_id": int(qid),
                "reference_text": ref_text,
                "contrasts": contrasts,
            })
    return samples


DECISION_INSTRUCTION = (
    '\n\nPlease answer the question above with only "Yes" or "No".\n\nAnswer:'
)


def build_decision_prompt(filled_template):
    """Append the yes/no decision instruction to a filled scenario template."""
    return filled_template.rstrip() + DECISION_INSTRUCTION


def contrast_target_pct(reference_text, contrast_text, tokenizer):
    """Token-edit-distance percentage between the reference and a contrast template."""
    return token_edit_distance_percent(reference_text, contrast_text, tokenizer)


def merge_paraphrases(samples, paraphrase_results):
    """Attach per-contrast paraphrase text to each sample under 'paraphrases'."""
    out = []
    for s in samples:
        qid = s["decision_question_id"]
        paras = {}
        for key in s["contrasts"]:
            rec = paraphrase_results.get(f"{qid}__{key}")
            paras[key] = rec["paraphrase"] if rec else None
        s2 = dict(s)
        s2["paraphrases"] = paras
        out.append(s2)
    return out


def merge_paraphrases_from_file(samples, path):
    """Load paraphrase results JSON and merge into samples."""
    import json
    with open(path) as f:
        return merge_paraphrases(samples, json.load(f))
