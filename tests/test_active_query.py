"""Tests for Task 4: EIG Engine and Active Querying Policy."""

import os
import pickle
import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

from src.active_query.eig import expected_info_gain, shannon_entropy
from src.active_query.policy import next_step


def create_toy_network():
    """Build the exact 3-condition toy network from Milestone2_.ipynb Part 4."""
    model = BayesianNetwork([
        ("Condition", "Sweating"),
        ("Condition", "Exertional"),
        ("Condition", "Nausea"),
    ])
    states = ["ACS", "GERD", "Panic"]
    cpd_c = TabularCPD("Condition", 3, [[0.30], [0.40], [0.30]], state_names={"Condition": states})

    def sym(name, p_yes):
        return TabularCPD(
            name, 2, [[1 - p for p in p_yes], p_yes],
            evidence=["Condition"], evidence_card=[3],
            state_names={name: ["no", "yes"], "Condition": states},
        )

    cpd_s = sym("Sweating", [0.70, 0.10, 0.40])
    cpd_e = sym("Exertional", [0.80, 0.05, 0.10])
    cpd_n = sym("Nausea", [0.50, 0.30, 0.20])
    model.add_cpds(cpd_c, cpd_s, cpd_e, cpd_n)
    return model


def test_toy_network_eig_reproduction():
    """Verify that EIG reproduces Milestone2 Exertional 0.407 / Nausea 0.062 bits."""
    toy_model = create_toy_network()
    evidence = {"Sweating": "yes"}

    eig_exertional_ve = expected_info_gain(toy_model, "Exertional", evidence, use_fast_bayes=False)
    eig_nausea_ve = expected_info_gain(toy_model, "Nausea", evidence, use_fast_bayes=False)

    eig_exertional_fast = expected_info_gain(toy_model, "Exertional", evidence, use_fast_bayes=True)
    eig_nausea_fast = expected_info_gain(toy_model, "Nausea", evidence, use_fast_bayes=True)

    # Exact check against notebook numbers: 0.407 and 0.062
    assert round(eig_exertional_ve, 3) == 0.407, f"Expected 0.407 bits, got {eig_exertional_ve}"
    assert round(eig_nausea_ve, 3) == 0.062, f"Expected 0.062 bits, got {eig_nausea_ve}"

    # Fast Bayes calculation must match pgmpy VariableElimination
    assert round(eig_exertional_fast, 3) == 0.407
    assert round(eig_nausea_fast, 3) == 0.062


def test_answered_questions_never_asked_again():
    """Verify that answered questions are excluded from subsequent questioning."""
    with open("results/bn_model.pkl", "rb") as f:
        model = pickle.load(f)

    evidence = {"sweating": "yes"}
    answered = {"sweating"}

    step1 = next_step(model, evidence, answered)
    assert step1["action"] == "ask"
    assert step1["question"] != "sweating", "Answered question must not be re-asked"

    answered.add(step1["question"])
    evidence[step1["question"]] = "no"

    step2 = next_step(model, evidence, answered)
    if step2["action"] == "ask":
        assert step2["question"] not in answered, "Previously answered question was re-asked!"


def test_max_question_cap():
    """Verify that reaching max_questions (8) forces an immediate stop."""
    with open("results/bn_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Simulate 8 answered questions
    answered = {
        "sweating", "shortness_of_breath", "racing_heart", "worse_after_eating",
        "burning_sensation", "worse_lying_down", "pleuritic_pain", "recent_viral_infection"
    }
    evidence = {q: "no" for q in answered}

    step = next_step(model, evidence, answered, cfg={"max_questions": 8})
    assert step["action"] == "stop", "Must stop when max_questions cap is reached"
    assert step["question"] is None
    assert step["triage"] in ["URGENT", "SEMI-URGENT", "ROUTINE"]
    assert "Maximum question cap" in step["reason"]


def test_acs_safety_escalation():
    """Verify safety rule: P(ACS) >= 0.25 forces URGENT triage."""
    with open("results/bn_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Evidence giving high ACS risk
    evidence = {
        "prior_heart_attack_angina": "yes",
        "pain_character": "heavy_pressure",
    }
    answered = {"prior_heart_attack_angina", "pain_character"}

    # Simulate stopping by reaching max questions
    step = next_step(model, evidence, answered, cfg={"max_questions": 2, "acs_risk_threshold": 0.25})
    assert step["action"] == "stop"
    assert step["triage"] == "URGENT", f"Expected URGENT for high ACS risk, got {step['triage']}"
    assert "SAFETY ESCALATION" in step["reason"] or "ACS" in step["reason"]
