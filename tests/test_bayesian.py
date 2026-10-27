"""Tests for Task 3: Bayesian Network (CPD normalization, smoothing, inference)."""

import os
import pickle
import numpy as np
import pytest
from src.bayesian.build_model import load_nodes_config
from src.bayesian.inference import posterior


@pytest.fixture(scope="module")
def model():
    """Load the trained Bayesian Network from results/bn_model.pkl."""
    model_path = os.path.join("results", "bn_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("Trained model results/bn_model.pkl not found; run bn_eval first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def test_cpt_columns_sum_to_one(model):
    """Test 1: Every column of every CPT sums to 1.0."""
    for cpd in model.get_cpds():
        vals = cpd.values
        if cpd.variable == "Condition":
            # Prior vector
            assert np.isclose(np.sum(vals), 1.0, atol=1e-5), "Condition prior must sum to 1.0"
        else:
            # Conditional CPT: sum down states for each condition parent
            col_sums = np.sum(vals, axis=0)
            assert np.allclose(col_sums, 1.0, atol=1e-5), (
                f"CPD for {cpd.variable} columns do not sum to 1.0: {col_sums}"
            )


def test_no_cpt_entry_is_zero_or_one(model):
    """Test 2: Laplace smoothing ensures no CPT entry is exactly 0.0 or 1.0."""
    for cpd in model.get_cpds():
        if cpd.variable == "Condition":
            continue
        vals = cpd.values
        assert not np.any(vals <= 0.0), f"Found non-positive value in CPD for {cpd.variable}"
        assert not np.any(vals >= 1.0), f"Found value >= 1.0 in CPD for {cpd.variable}"


def test_classic_acs_presentation_ranks_acs_first(model):
    """Test 3: Classic ACS evidence set ranks ACS first."""
    acs_evidence = {
        "pain_character": "heavy_pressure",
        "pain_radiation": "arm_shoulder",
        "sweating": "yes",
        "prior_heart_attack_angina": "yes",
        "high_blood_pressure": "yes",
    }
    post = posterior(model, acs_evidence)
    top_cond = max(post.items(), key=lambda x: x[1])[0]
    assert top_cond == "ACS", f"Expected ACS to rank first, got {top_cond} with probs {post}"
    assert post["ACS"] > 0.8, f"Expected high ACS confidence, got {post['ACS']}"


def test_classic_gerd_presentation_ranks_gerd_first(model):
    """Test 4: Classic GERD evidence set ranks GERD first."""
    gerd_evidence = {
        "pain_character": "burning",
        "burning_sensation": "yes",
        "worse_after_eating": "yes",
        "sweating": "no",
        "prior_heart_attack_angina": "no",
    }
    post = posterior(model, gerd_evidence)
    top_cond = max(post.items(), key=lambda x: x[1])[0]
    assert top_cond == "GERD", f"Expected GERD to rank first, got {top_cond} with probs {post}"
    assert post["GERD"] > 0.8, f"Expected high GERD confidence, got {post['GERD']}"


def test_contradictory_evidence_does_not_crash(model):
    """Test 5: Contradictory evidence does not crash and returns valid distribution."""
    contradictory_evidence = {
        "pain_character": "heavy_pressure",  # suggests ACS
        "burning_sensation": "yes",          # suggests GERD
        "pleuritic_pain": "yes",             # suggests Pericarditis
        "feeling_of_impending_doom": "yes",  # suggests Panic
    }
    post = posterior(model, contradictory_evidence)
    assert len(post) == 4
    assert np.isclose(sum(post.values()), 1.0, atol=1e-5), "Posterior must sum to 1.0"
    for prob in post.values():
        assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"


def test_fast_bayes_matches_pgmpy_variable_elimination(model):
    """Verify that vectorized Bayes inference matches pgmpy VariableElimination."""
    evidence = {
        "pain_character": "heavy_pressure",
        "sweating": "yes",
        "shortness_of_breath": "yes",
    }
    post_fast = posterior(model, evidence, use_fast_bayes=True)
    post_ve = posterior(model, evidence, use_fast_bayes=False)

    for cond in ["ACS", "GERD", "Panic", "Pericarditis"]:
        assert np.isclose(post_fast[cond], post_ve[cond], atol=1e-4), (
            f"Mismatch on {cond}: fast={post_fast[cond]} vs VE={post_ve[cond]}"
        )
