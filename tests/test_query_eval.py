"""Tests for Task 9: Evaluation of active questioning policies."""

import json
import os


def test_query_eval_results_and_plot():
    """Verify query evaluation results file and benchmark plot."""
    json_path = os.path.join("results", "query_eval.json")
    plot_path = os.path.join("results", "query_eval_plot.html")

    assert os.path.exists(json_path), "results/query_eval.json must exist"
    assert os.path.exists(plot_path), "results/query_eval_plot.html must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cohort = data.get("evaluation_cohort", {})
    assert cohort.get("sample_size") == 400
    assert cohort.get("random_seed") == 42

    policies = data.get("policy_comparison", {})
    assert "eig" in policies
    assert "random" in policies
    assert "fixed" in policies

    eig = policies["eig"]
    rand = policies["random"]
    fixed = policies["fixed"]

    # EIG must be strictly more efficient than random and fixed order
    assert eig["avg_questions_needed"] < rand["avg_questions_needed"], (
        f"EIG ({eig['avg_questions_needed']}) should ask fewer questions than Random ({rand['avg_questions_needed']})"
    )
    assert eig["avg_questions_needed"] < fixed["avg_questions_needed"], (
        f"EIG ({eig['avg_questions_needed']}) should ask fewer questions than Fixed ({fixed['avg_questions_needed']})"
    )

    # EIG must maintain 100% ACS safety recall
    assert eig["acs_recall"] == 1.0
    assert eig["acs_safety_escalation_recall"] == 1.0
