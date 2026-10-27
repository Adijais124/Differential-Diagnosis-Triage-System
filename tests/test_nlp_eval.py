"""Tests for Task 6: NLP Evaluation and Gold Set."""

import json
import os


def test_nlp_cases_valid():
    """Verify nlp_cases.json contains 40-60 sentences and is marked for human review."""
    path = os.path.join("evaluation", "nlp_cases.json")
    assert os.path.exists(path), "evaluation/nlp_cases.json must exist"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("needs_human_review") is True, "Must be flagged with needs_human_review: true"
    cases = data.get("cases", [])
    assert 40 <= len(cases) <= 60, f"Expected 40-60 test cases, found {len(cases)}"


def test_nlp_evaluation_results():
    """Verify that results/nlp_eval.json exists and pipeline outperforms baseline."""
    res_path = os.path.join("results", "nlp_eval.json")
    assert os.path.exists(res_path), "results/nlp_eval.json must exist"

    with open(res_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report.get("needs_human_review") is True
    assert "draft gold set, not yet reviewed" in report.get("status", "")
    summary = report.get("summary", {})
    assert summary.get("pipeline_beats_baseline") is True, "Full pipeline must outperform baseline"
    assert summary.get("pipeline_f1", 0.0) > summary.get("baseline_f1", 0.0)
