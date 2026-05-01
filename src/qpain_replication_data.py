# ABOUTME: Data loading and template filling for Q-Pain replication experiment
# ABOUTME: Handles 4 races x 2 genders with few-shot prompt construction for GPT-2

import copy
import csv
import random
import warnings
from itertools import combinations

DATA_DIR = "physionet.org/files/q-pain/1.0.0"

CONTEXT_FILES = {
    "acute_cancer": "data_acute_cancer.csv",
    "acute_non_cancer": "data_acute_non_cancer.csv",
    "chronic_cancer": "data_chronic_cancer.csv",
    "chronic_non_cancer": "data_chronic_non_cancer.csv",
    "post_op": "data_post_op.csv",
}

# Pandas default NA strings that get converted to NaN
_PANDAS_NA_VALUES = {"", "N/A", "NA", "n/a", "na", "NaN", "nan", "NULL", "null", "None"}


def _pandas_compatible_dosage(raw_value):
    """Convert CSV dosage string to match pandas str(NaN) behavior.

    The original code uses pandas which reads 'N/A' as NaN, then
    str(NaN) -> 'nan'. csv.DictReader returns the raw string, so
    we replicate the pandas conversion.
    """
    if raw_value.strip() in _PANDAS_NA_VALUES:
        return "nan"
    return raw_value


def load_context_data(data_dir=DATA_DIR):
    """Load all 5 Q-Pain CSVs, separating Yes/No vignettes per context.

    Identifies Patient B (No vignette) by absence of [race] placeholder,
    matching the original paper's convention that Patient B vignettes are
    non-templated.
    """
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
                    "dosage": _pandas_compatible_dosage(row["Dosage"]),
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


RACES = ["Black", "White", "Asian", "Hispanic"]
GENDERS = ["man", "woman"]
PRONOUNS = {
    "subject": {"man": "he", "woman": "she"},
    "possessive": {"man": "his", "woman": "her"},
}

NAMES = {
    "Black": {
        "man": ["Roosevelt", "Jermaine", "Darnell", "Willie", "Mattie",
                "Reginald", "Cedric", "Sylvester", "Tyrone", "Errol"],
        "woman": ["Latonya", "Tamika", "Ebony", "Latasha", "Keisha",
                  "Lillie", "Minnie", "Gwendolyn", "Bessie", "Marva"],
    },
    "White": {
        "man": ["Bradley", "Brett", "Scott", "Kurt", "Todd", "Chad",
                "Matthew", "Dustin", "Shane", "Douglas"],
        "woman": ["Beth", "Megan", "Kristin", "Jill", "Erin", "Colleen",
                  "Kathleen", "Heather", "Holly", "Laurie"],
    },
    "Asian": {
        "man": ["Viet", "Thong", "Qiang", "Kwok", "Hao", "Yang",
                "Nam", "Huy", "Yuan", "Ho"],
        "woman": ["Zhen", "Nga", "Lien", "Lam", "Hui", "Wing",
                  "Hoa", "Wai", "Min", "Huong"],
    },
    "Hispanic": {
        "man": ["Rigoberto", "Santos", "Javier", "Efrain", "Juan",
                "Ramiro", "Jesus", "Humberto", "Gonzalo", "Hector"],
        "woman": ["Guadalupe", "Marisela", "Guillermina", "Rocio",
                  "Yesenia", "Blanca", "Rosalba", "Elvia", "Alejandra", "Mayra"],
    },
}

SUBGROUPS = [f"{r}_{g}" for r in RACES for g in GENDERS]


def fill_vignette(vignette_text, race, gender, name):
    """Fill a vignette template with demographics. Does NOT handle [subjective] typo."""
    text = vignette_text
    text = text.replace("[race]", race)
    text = text.replace("[gender]", gender)
    text = text.replace("[possessive]", PRONOUNS["possessive"][gender])
    text = text.replace("[subject]", PRONOUNS["subject"][gender])
    text = text.replace("Patient D", name)
    return text


def fill_question(question_text, name):
    """Replace Patient D with the assigned name in the question."""
    return question_text.replace("Patient D", name)


def assign_names_for_context(seed=42):
    """Shuffle names with seed, returning a fresh shuffled copy.

    Re-initializes from the original NAMES constant to ensure identical
    permutation regardless of prior state (shuffle is in-place).
    Loop order matches the original Q-Pain code: gender outer, race inner.
    """
    rng = random.Random(seed)
    names = copy.deepcopy(NAMES)
    for g in GENDERS:
        for r in RACES:
            rng.shuffle(names[r][g])
    return names


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
    """Remove demographics and rename Patient D for a closed prompt."""
    text = text.replace("[gender] ", "")
    text = text.replace("[race] ", "")
    text = text.replace("[subjective]", patient_name)
    text = text.replace("[subject]", patient_name)
    text = text.replace("[possessive]", "their")
    text = text.replace("Patient D", patient_name)
    return text


def build_open_prompt(vignette, question):
    """Format the open (test) prompt ending with 'Answer:'."""
    return (
        f"Vignette: {vignette}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_closed_prompt(vignette, question, answer, dosage, explanation):
    """Format a single closed (few-shot example) prompt."""
    return (
        f"Vignette: {vignette}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"{dosage}\n"
        f"Explanation: {explanation}##\n\n"
    )


def build_closed_prompts_for_vignette(vignette_idx, context, closed_data, rng,
                                      open_prompt, tokenizer, max_context=1024):
    """Build 2 closed prompts for a given vignette. Resample on overflow.

    Returns (closed_a, closed_b) — Patient A (Yes+High) and Patient B (No).
    Patient C (Yes+Low) is omitted, matching the original code where it was
    commented out. The RNG still samples 2 indices to match the original
    code's consumption pattern.
    """
    ctx_data = closed_data[context]
    available = [i for i in range(10) if i != vignette_idx]

    pb = ctx_data["patient_b"]
    closed_b = build_closed_prompt(
        vignette=pb["vignette"], question=pb["question"],
        answer=pb["answer"], dosage=f"Dosage: {pb['dosage']}",
        explanation=pb["explanation"],
    )

    for _ in range(100):
        # Sample 2 to match original code's RNG consumption pattern,
        # but only use sample[0] — Patient C was commented out in the original.
        sample = rng.sample(available, 2)

        v_a = ctx_data["yes_vignettes"][sample[0]]
        closed_a = build_closed_prompt(
            vignette=standardize_closed(v_a["vignette"], "Patient A"),
            question=standardize_closed(v_a["question"], "Patient A"),
            answer=v_a["answer"],
            dosage=DOSAGE_HIGH[context],
            explanation=standardize_closed(v_a["explanation"], "Patient A"),
        )

        if tokenizer is None or max_context is None:
            return closed_a, closed_b

        full_prompt = closed_a + closed_b + open_prompt
        n_tokens = len(tokenizer.encode(full_prompt))
        if n_tokens <= max_context:
            return closed_a, closed_b

    warnings.warn(
        f"All 100 resample attempts exceeded {max_context} tokens for "
        f"vignette {vignette_idx} in {context} (last attempt: {n_tokens} tokens). "
        f"Returning last attempt -- GPT-2 may truncate.",
    )
    return closed_a, closed_b


def prepare_dataset(data_dir=DATA_DIR, seed=42):
    """Build the full dataset: 50 vignettes x 8 subgroup texts each."""
    closed_data = load_context_data(data_dir)
    dataset = []
    vignette_id = 0

    for context in CONTEXT_FILES:
        names = assign_names_for_context(seed=seed)
        yes_vignettes = closed_data[context]["yes_vignettes"]

        for q in range(len(yes_vignettes)):
            v = yes_vignettes[q]
            texts = {}
            questions = {}
            for race in RACES:
                for gender in GENDERS:
                    subgroup = f"{race}_{gender}"
                    name = names[race][gender][q]
                    texts[subgroup] = fill_vignette(v["vignette"], race, gender, name)
                    questions[subgroup] = fill_question(v["question"], name)
            dataset.append({
                "vignette_id": vignette_id,
                "context": context,
                "vignette_idx_in_context": q,
                "raw_question": v["question"],
                "texts": texts,
                "questions": questions,
                "names": {f"{r}_{g}": names[r][g][q]
                          for r in RACES for g in GENDERS},
            })
            vignette_id += 1
    return dataset


def compute_comparisons(seed=99):
    """Generate all 28 pairwise subgroup comparisons with random original/swapped."""
    rng = random.Random(seed)
    comparisons = []
    for a, b in combinations(SUBGROUPS, 2):
        if rng.random() < 0.5:
            comparisons.append({"original": a, "swapped": b})
        else:
            comparisons.append({"original": b, "swapped": a})
    return comparisons
