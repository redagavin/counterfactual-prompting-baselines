# ABOUTME: Pre-flight gates for DiscrimEval: V1 selection, V2 calibration, V0 yes/no mass, V3 raw-effect
# ABOUTME: V1/V2/V3 are CPU/JSON checks; V0 needs a GPU (run via srun)

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from discrimeval_data import load_discrimeval_explicit, select_samples, CONTRASTS


def v1_selection():
    """V1: real DiscrimEval data yields scenarios each with all 14 contrasts."""
    samples = select_samples(load_discrimeval_explicit())
    assert len(samples) > 0, "no scenarios selected"
    expected_keys = {c[0] for c in CONTRASTS}
    for s in samples:
        assert set(s["contrasts"].keys()) == expected_keys, \
            f"scenario {s['decision_question_id']} has wrong contrast keys"
    print(f"V1 PASS: {len(samples)} scenarios, 14 contrasts each")
    return samples


def v0_mass(model_name, k=5):
    """V0: yes/no tokens capture >= 0.99 of the full-vocab softmax mass (GPU)."""
    from discrimeval_evaluate import DiscrimEvalEvaluator
    samples = select_samples(load_discrimeval_explicit())[:k]
    ev = DiscrimEvalEvaluator(model_name)
    print(f"yes_id={ev.yes_id}, no_id={ev.no_id}")
    masses = [ev.mass(s["reference_text"]) for s in samples]
    for s, m in zip(samples, masses):
        print(f"  scenario {s['decision_question_id']}: p_yes+p_no = {m:.4f}")
    n_ok = sum(1 for m in masses if m >= 0.99)
    assert n_ok == len(masses), \
        f"V0 FAIL: only {n_ok}/{len(masses)} scenarios have yes/no mass >= 0.99 (wrong token variant or prompt format)"
    print(f"V0 PASS: yes/no mass >= 0.99 on {n_ok}/{len(masses)} scenarios")


# Reference demographics (white, 60-year-old, male) that every paraphrase must preserve.
# Race ("white") and age ("60") are rendered as literal tokens in every DiscrimEval
# reference, so we require them present. Gender is sometimes encoded only via pronouns
# ("he"/"his") or gender-neutral phrasing ("they"), so we do NOT require the literal
# token "male"; instead the ABSENT set below catches any drift to a *different* gender.
_REF_PRESENT = [r"\bwhite\b", r"\b60\b"]
_REF_ABSENT = [r"\bblack\b", r"\basian\b", r"\bhispanic\b", r"\bnative american\b",
               r"\bfemale\b", r"\bnon-?binary\b"]


def v2_calibration(paraphrases_path):
    """V2: each paraphrase is within tolerance or a conservative undershoot, and
    keeps the reference demographics (white / 60 / male) unchanged."""
    with open(paraphrases_path) as f:
        paras = json.load(f)
    assert paras, "no paraphrases in file"
    n_calib_ok = n_demo_ok = 0
    for pid, rec in paras.items():
        calib_ok = rec["deviation"] <= 0.5 or rec["actual_pct"] <= rec["target_pct"]
        n_calib_ok += int(calib_ok)
        text = rec["paraphrase"].lower()
        present = all(re.search(p, text) for p in _REF_PRESENT)
        absent = not any(re.search(p, text) for p in _REF_ABSENT)
        if present and absent:
            n_demo_ok += 1
        else:
            print(f"  WARN {pid}: demographic drift (ref-present={present}, no-foreign={absent})")
        print(f"  {pid}: target={rec['target_pct']:.2f}% actual={rec['actual_pct']:.2f}% "
              f"dev={rec['deviation']:.2f} calib_ok={calib_ok}")
    assert n_calib_ok == len(paras), \
        f"V2 FAIL: {len(paras) - n_calib_ok} paraphrases neither within tol nor undershoot"
    assert n_demo_ok == len(paras), \
        f"V2 FAIL: {len(paras) - n_demo_ok} paraphrases altered reference demographics"
    print(f"V2 PASS: {len(paras)} paraphrases — all calibrated and demographics preserved")


def v3_raw_effect(eval_path):
    """V3 (diagnostic): raw per-contrast mean(logit_contrast - logit_reference) is
    finite and not all identically zero (catches an instrumentation bug)."""
    import numpy as np
    with open(eval_path) as f:
        results = json.load(f)
    print("V3 raw per-contrast effect mean(logit_contrast - logit_reference):")
    any_nonzero = False
    for key, *_ in CONTRASTS:
        diffs = [r["contrasts"][key]["logit_contrast"] - r["logit_reference"]
                 for r in results if key in r["contrasts"]]
        if not diffs:
            continue
        mean = float(np.mean(diffs))
        assert np.isfinite(mean), f"V3 FAIL: non-finite raw effect for {key}"
        if abs(mean) > 1e-9:
            any_nonzero = True
        print(f"  {key:22s}: {mean:+.4f}  (n={len(diffs)})")
    assert any_nonzero, "V3 FAIL: all raw effects identically zero (suspicious)"
    print("V3 PASS: raw effects finite and not all zero")


def main():
    ap = argparse.ArgumentParser(description="DiscrimEval pre-flight validation gates")
    ap.add_argument("--v1", action="store_true", help="selection sanity (CPU + HF download)")
    ap.add_argument("--v0", action="store_true", help="yes/no mass gate (needs GPU + --model)")
    ap.add_argument("--v2", action="store_true", help="paraphrase calibration (needs --paraphrases)")
    ap.add_argument("--v3", action="store_true", help="raw-effect sanity (needs --eval)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--paraphrases")
    ap.add_argument("--eval")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if not (args.v0 or args.v1 or args.v2 or args.v3):
        args.v1 = True  # default to the GPU-free gate
    if args.v1:
        v1_selection()
    if args.v0:
        v0_mass(args.model, args.k)
    if args.v2:
        assert args.paraphrases, "--v2 requires --paraphrases"
        v2_calibration(args.paraphrases)
    if args.v3:
        assert args.eval, "--v3 requires --eval"
        v3_raw_effect(args.eval)


if __name__ == "__main__":
    main()
