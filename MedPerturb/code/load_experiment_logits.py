# ABOUTME: Loads real logits and binary answers from MedPerturb experiment JSON files.
# ABOUTME: Converts P(Yes) to logits with clamping for extreme probabilities.

import json
import math

import numpy as np


CLAMP_EPSILON = 1e-6


def load_condition(json_path: str, question: str) -> dict:
    """Load z_i (logits) and y_orig (majority-vote binary answers) for one condition.

    Args:
        json_path: Path to experiment result JSON (list of dicts).
        question: One of "MANAGE", "VISIT", "RESOURCE".

    Returns:
        dict with:
            z_i: numpy float64 array of logits, log(p / (1-p))
            y_orig: numpy int array of majority-vote binary answers
    """
    with open(json_path) as f:
        data = json.load(f)

    key = f"original_{question}"
    z_i_list = []
    y_orig_list = []

    for entry in data:
        record = entry[key]

        p = record["logit_probs"]
        p_clamped = max(CLAMP_EPSILON, min(1.0 - CLAMP_EPSILON, p))
        z_i_list.append(math.log(p_clamped / (1.0 - p_clamped)))

        votes = record["binary_answers"]
        y_orig_list.append(1 if sum(votes) >= 2 else 0)

    return {
        "z_i": np.array(z_i_list, dtype=np.float64),
        "y_orig": np.array(y_orig_list, dtype=np.int64),
    }
