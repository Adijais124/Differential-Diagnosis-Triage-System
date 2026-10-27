"""Analyze evidence distribution across the 4 conditions on DDXPlus train set.

Calculates P(evidence | condition), spread (max - min), and flags (leaky, rare, constant).
Saves results to results/candidate_evidence_spread.json.
"""

import ast
import json
import os
import pandas as pd
from src.data.load_ddxplus import load_evidences_metadata, TARGET_CONDITIONS


def analyze_candidate_nodes(
    data_path: str = "data/processed/train_filtered.csv.gz",
    out_file: str = "results/candidate_evidence_spread.json",
) -> list[dict]:
    """Compute evidence spread and characteristics across the 4 conditions."""
    print("Loading filtered train dataset...")
    df = pd.read_csv(data_path, compression="gzip")
    ev_meta = load_evidences_metadata()

    # Total counts per condition
    cond_counts = df["CONDITION"].value_counts().to_dict()
    total_patients = len(df)

    print("Parsing evidence presence per patient...")
    # Map patient index to set of raw evidence strings
    patient_ev_sets = []
    for ev_str in df["EVIDENCES"]:
        try:
            items = set(ast.literal_eval(ev_str))
        except Exception:
            items = set()
        patient_ev_sets.append(items)

    df["EV_SET_RAW"] = patient_ev_sets

    # Identify all observed base codes and pain multi-valued codes
    base_codes_seen = set()
    pain_codes_seen = set()

    for ev_set in df["EV_SET_RAW"]:
        for item in ev_set:
            base = item.split("_@_")[0]
            base_codes_seen.add(base)
            if base in ["E_54", "E_57", "E_55"]:
                pain_codes_seen.add(item)

    print(f"Total base codes seen: {len(base_codes_seen)}; pain sub-codes: {len(pain_codes_seen)}")

    candidates = []

    # Pre-partition evidence sets by condition for speed
    subsets = {}
    for c in TARGET_CONDITIONS:
        subsets[c] = df[df["CONDITION"] == c]["EV_SET_RAW"].tolist()

    # 1. Evaluate base evidence codes
    for code in sorted(base_codes_seen):
        meta = ev_meta.get(code, {})
        q_en = meta.get("question_en", f"Code {code}")

        p_cond = {}
        for c in TARGET_CONDITIONS:
            ev_list = subsets[c]
            count = sum(1 for ev_s in ev_list if any(i.split("_@_")[0] == code for i in ev_s))
            p_cond[c] = count / cond_counts[c]

        probs = list(p_cond.values())
        max_p = max(probs)
        min_p = min(probs)
        spread = max_p - min_p
        overall_prev = sum(p_cond[c] * cond_counts[c] for c in TARGET_CONDITIONS) / total_patients

        flags = []
        if max_p > 0.98 and min_p > 0.98:
            flags.append("constant_high")
        if max_p < 0.005 and min_p < 0.005:
            flags.append("rare")
        if "pericarditis" in q_en.lower() or code == "E_3":
            flags.append("leaky")

        candidates.append({
            "code": code,
            "type": "binary",
            "question_en": q_en,
            "p_cond": {k: round(v, 4) for k, v in p_cond.items()},
            "spread": round(spread, 4),
            "overall_prevalence": round(overall_prev, 4),
            "flags": flags,
        })

    # Sort by spread descending
    candidates.sort(key=lambda x: x["spread"], reverse=True)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"Saved {len(candidates)} candidate evaluations to {out_file}")
    return candidates


if __name__ == "__main__":
    analyze_candidate_nodes()
