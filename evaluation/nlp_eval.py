"""Evaluation script for Task 6: NLP benchmark against rule-only baseline.

Compares:
(a) Rule-only baseline (keyword search + naive 25-char window negation)
(b) Full clinical NLP pipeline (scispaCy PhraseMatcher + clause-aware negation + variant generator)

Saves per-node precision, recall, and F1 metrics to results/nlp_eval.json.
"""

import json
import os
import re
from src.nlp.extractor import extract_findings
from src.nlp.synonyms import build_synonym_dictionary

# Baseline keyword dictionary based on Milestone2_.ipynb cell 2139
BASELINE_KEYWORDS = {
    "shortness_of_breath": ["shortness of breath", "breathless", "breathing difficulty", "can't catch my breath", "dyspnea"],
    "pain_character": ["chest pressure", "heavy pressure in the centre of his chest", "chest tightness", "burning", "sharp"],
    "sweating": ["sweating", "sweaty", "diaphoresis", "cold sweat"],
    "racing_heart": ["palpitations", "racing heart", "heart racing"],
    "feeling_of_impending_doom": ["anxiety", "anxious", "panic attack", "panic", "dying"],
    "burning_sensation": ["heartburn", "burning in the throat", "acid reflux"],
    "worse_after_eating": ["worse after eating", "after meals"],
    "worse_lying_down": ["worse lying down", "relieved sitting up"],
    "pleuritic_pain": ["hurts to breathe", "worse on deep breath"],
    "recent_viral_infection": ["viral infection", "cold", "flu"],
    "prior_heart_attack_angina": ["heart attack", "angina", "myocardial infarction"],
    "smoker": ["smoker", "smoking"],
    "high_blood_pressure": ["high blood pressure", "hypertension"],
}


def rule_only_baseline(text: str) -> dict:
    """Milestone 2 rule-only baseline: keyword matching with naive 25-character negation window."""
    low = text.lower()
    out = {}
    for node, phrases in BASELINE_KEYWORDS.items():
        for p in phrases:
            for m in re.finditer(re.escape(p), low):
                window = low[max(0, m.start() - 25):m.start()]
                neg = bool(re.search(r"\b(no|denies|denied|without|not)\b", window))
                if node == "pain_character":
                    if not neg:
                        if "pressure" in p or "tightness" in p:
                            out[node] = "heavy_pressure"
                        elif "burning" in p:
                            out[node] = "burning"
                        elif "sharp" in p:
                            out[node] = "sharp_knife"
                else:
                    out[node] = "no" if neg else "yes"
    return out


def evaluate_extractor(extractor_fn, test_cases: list[dict]) -> dict:
    """Compute micro and per-node precision, recall, and F1 against gold annotations."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_node_stats = {}

    for case in test_cases:
        text = case["text"]
        gold = case["gold_findings"]

        preds = extractor_fn(text)

        all_keys = set(gold.keys()) | set(preds.keys())

        for k in all_keys:
            if k not in per_node_stats:
                per_node_stats[k] = {"tp": 0, "fp": 0, "fn": 0}

            g_val = gold.get(k)
            p_val = preds.get(k)

            if g_val is not None and p_val is not None:
                if g_val == p_val:
                    per_node_stats[k]["tp"] += 1
                    total_tp += 1
                else:
                    per_node_stats[k]["fp"] += 1
                    per_node_stats[k]["fn"] += 1
                    total_fp += 1
                    total_fn += 1
            elif g_val is None and p_val is not None:
                per_node_stats[k]["fp"] += 1
                total_fp += 1
            elif g_val is not None and p_val is None:
                per_node_stats[k]["fn"] += 1
                total_fn += 1

    def calc_metrics(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    micro_prec, micro_rec, micro_f1 = calc_metrics(total_tp, total_fp, total_fn)

    per_node = {}
    for node, s in sorted(per_node_stats.items()):
        p, r, f = calc_metrics(s["tp"], s["fp"], s["fn"])
        per_node[node] = {"precision": p, "recall": r, "f1": f, "support": s["tp"] + s["fn"]}

    return {
        "overall": {
            "precision": micro_prec,
            "recall": micro_rec,
            "f1_score": micro_f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_node": per_node,
    }


def run_task6_nlp_evaluation(
    cases_path: str = "evaluation/nlp_cases.json",
    out_file: str = "results/nlp_eval.json",
) -> dict:
    """Run comparative evaluation of baseline vs pipeline on nlp_cases.json."""
    print("Loading NLP test cases...")
    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data["cases"]
    print(f"Loaded {len(test_cases)} test cases.")

    print("Evaluating Rule-Only Baseline...")
    baseline_metrics = evaluate_extractor(rule_only_baseline, test_cases)
    print(f"Baseline F1: {baseline_metrics['overall']['f1_score']:.4f}")

    print("Evaluating Full Clinical NLP Pipeline...")
    def pipeline_fn(t):
        return extract_findings(t)["findings"]

    pipeline_metrics = evaluate_extractor(pipeline_fn, test_cases)
    print(f"Pipeline F1: {pipeline_metrics['overall']['f1_score']:.4f}")

    results = {
        "status": "draft gold set, not yet reviewed",
        "needs_human_review": True,
        "num_test_cases": len(test_cases),
        "comparison": {
            "baseline": baseline_metrics,
            "full_pipeline": pipeline_metrics,
        },
        "summary": {
            "baseline_f1": baseline_metrics["overall"]["f1_score"],
            "pipeline_f1": pipeline_metrics["overall"]["f1_score"],
            "f1_gain": round(pipeline_metrics["overall"]["f1_score"] - baseline_metrics["overall"]["f1_score"], 4),
            "pipeline_beats_baseline": pipeline_metrics["overall"]["f1_score"] > baseline_metrics["overall"]["f1_score"],
        },
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation metrics to {out_file}")
    return results


if __name__ == "__main__":
    run_task6_nlp_evaluation()
