"""Evaluation script for Task 3: Bayesian Network.

Evaluates test-split accuracy and per-class recall (especially ACS recall)
for both empirical and uniform prior networks.
Saves comprehensive metrics to results/bn_eval.json.
"""

import ast
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.bayesian.build_model import (
    build_model,
    load_nodes_config,
    extract_node_states_from_patient,
)
from src.bayesian.inference import posterior
from src.data.load_ddxplus import TARGET_CONDITIONS


def evaluate_split(
    model,
    test_df: pd.DataFrame,
    nodes: list[dict],
) -> dict:
    """Evaluate Bayesian model on a patient split."""
    y_true = test_df["CONDITION"].tolist()
    y_pred = []

    print(f"Evaluating {len(test_df)} cases...")
    for ev_str in test_df["EVIDENCES"]:
        try:
            ev_set = set(ast.literal_eval(ev_str))
        except Exception:
            ev_set = set()

        # In testing against ground truth patient profiles, observable nodes
        # match the patient's state profile
        evidence = extract_node_states_from_patient(ev_set, nodes)
        post = posterior(model, evidence, use_fast_bayes=True)
        top_cond = max(post.items(), key=lambda x: x[1])[0]
        y_pred.append(top_cond)

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_class = {}
    for c in TARGET_CONDITIONS:
        c_stats = report.get(c, {})
        per_class[c] = {
            "precision": round(c_stats.get("precision", 0.0), 4),
            "recall": round(c_stats.get("recall", 0.0), 4),
            "f1_score": round(c_stats.get("f1-score", 0.0), 4),
            "support": int(c_stats.get("support", 0)),
        }

    return {
        "accuracy": round(acc, 4),
        "acs_recall": per_class["ACS"]["recall"],
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "per_class": per_class,
    }


def run_task3_evaluation(
    train_path: str = "data/processed/train_filtered.csv.gz",
    test_path: str = "data/processed/test_filtered.csv.gz",
    config_path: str = "config/nodes.json",
    out_file: str = "results/bn_eval.json",
    model_save_path: str = "results/bn_model.pkl",
) -> dict:
    """Train models with empirical and uniform priors, evaluate on test, and save metrics."""
    print("Loading datasets and configuration...")
    train_df = pd.read_csv(train_path, compression="gzip")
    test_df = pd.read_csv(test_path, compression="gzip")
    nodes_config = load_nodes_config(config_path)

    # 1. Empirical Prior Model
    print("\n--- Training Model with Empirical Prior ---")
    model_empirical = build_model(train_df, nodes_config, alpha=1.0, prior_type="empirical")
    print("Evaluating Empirical Prior Model on Test Split...")
    eval_empirical = evaluate_split(model_empirical, test_df, nodes_config["nodes"])
    print(f"Empirical Prior Accuracy: {eval_empirical['accuracy']:.4f}, ACS Recall: {eval_empirical['acs_recall']:.4f}")

    # 2. Uniform Prior Model
    print("\n--- Training Model with Uniform Prior ---")
    model_uniform = build_model(train_df, nodes_config, alpha=1.0, prior_type="uniform")
    print("Evaluating Uniform Prior Model on Test Split...")
    eval_uniform = evaluate_split(model_uniform, test_df, nodes_config["nodes"])
    print(f"Uniform Prior Accuracy: {eval_uniform['accuracy']:.4f}, ACS Recall: {eval_uniform['acs_recall']:.4f}")

    results = {
        "dataset_splits": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "prior_comparison": {
            "empirical_prior": eval_empirical,
            "uniform_prior": eval_uniform,
        },
        "discussion": (
            "Empirical prior reflects DDXPlus dataset prevalence where ACS accounts for 44.6% of presentations. "
            "Uniform prior (25% each) eliminates class prevalence bias, making likelihoods solely driven by patient findings. "
            "ACS recall remains the critical clinical safety metric."
        ),
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_file}")

    # Save trained empirical model to disk for reuse in tasks 4, 7, 8
    with open(model_save_path, "wb") as f:
        pickle.dump(model_empirical, f)
    print(f"Saved trained Bayesian Network to {model_save_path}")

    return results


if __name__ == "__main__":
    run_task3_evaluation()
