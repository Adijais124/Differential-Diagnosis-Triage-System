"""Bayesian Network construction and CPT learning using pgmpy.

Constructs a Naive Bayes network: Condition -> [Symptom nodes].
Applies Laplace smoothing (alpha=1.0) so no CPT entry is exactly 0 or 1.
Supports both empirical and uniform class priors.
"""

import ast
import json
import os
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

from src.data.load_ddxplus import TARGET_CONDITIONS


def load_nodes_config(config_path: str = "config/nodes.json") -> dict:
    """Load Bayesian nodes configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_node_states_from_patient(
    ev_set_raw: set[str],
    nodes: list[dict],
) -> dict[str, str]:
    """Map a patient's raw DDXPlus evidences to discrete node states.

    In training/simulator semantics (PROJECT_BRIEF Section 6):
    - Evidence absent from patient list -> 'no' for binary nodes,
      or default state ('none'/'nowhere') for multi-valued nodes.
    """
    states = {}
    for node in nodes:
        name = node["name"]
        code = node["ddxplus_code"]
        allowed_states = node["states"]

        if "value_mapping" in node:
            # Multi-valued node (e.g. pain character E_54, radiation E_57)
            val_mapping = node["value_mapping"]
            default_st = node.get("default_state", allowed_states[0])
            matched_state = None

            for item in ev_set_raw:
                if item.startswith(f"{code}_@_"):
                    val_id = item.split("_@_")[1]
                    if val_id in val_mapping:
                        matched_state = val_mapping[val_id]
                        break

            states[name] = matched_state if matched_state else default_st
        else:
            # Binary node: check if code is present
            present = any(i.split("_@_")[0] == code for i in ev_set_raw)
            states[name] = "yes" if present else "no"

    return states


def build_model(
    train_df: pd.DataFrame,
    nodes_config: dict,
    alpha: float = 1.0,
    prior_type: str = "empirical",
) -> BayesianNetwork:
    """Build and parameterize the Bayesian Network using Laplace smoothing.

    Args:
        train_df: Filtered training dataframe with 'CONDITION' and 'EVIDENCES' columns.
        nodes_config: Dict with 'nodes' specification list.
        alpha: Laplace smoothing parameter (default 1.0).
        prior_type: 'empirical' (from data proportions) or 'uniform' (0.25 each).

    Returns:
        Trained pgmpy.models.BayesianNetwork instance.
    """
    nodes = nodes_config["nodes"]
    node_names = [n["name"] for n in nodes]

    # Naive-Bayes structure: Condition -> each symptom node
    edges = [("Condition", name) for name in node_names]
    model = BayesianNetwork(edges)

    total_train = len(train_df)
    cond_counts = train_df["CONDITION"].value_counts().to_dict()

    # 1. Prior CPD for Condition
    num_conds = len(TARGET_CONDITIONS)
    if prior_type == "uniform":
        prior_values = [[1.0 / num_conds] for _ in range(num_conds)]
    else:
        # Empirical prior
        prior_values = [[cond_counts[c] / total_train] for c in TARGET_CONDITIONS]

    condition_cpd = TabularCPD(
        variable="Condition",
        variable_card=num_conds,
        values=prior_values,
        state_names={"Condition": TARGET_CONDITIONS},
    )
    model.add_cpds(condition_cpd)

    # 2. Extract patient states for all rows
    print("Extracting patient node states across training set...")
    raw_ev_sets = [
        set(ast.literal_eval(s)) if isinstance(s, str) else set()
        for s in train_df["EVIDENCES"]
    ]

    # Pre-compute count tables: count[node][condition][state]
    node_cpds = []
    conditions_series = train_df["CONDITION"].tolist()

    for node in nodes:
        name = node["name"]
        states = node["states"]
        k_states = len(states)
        state_to_idx = {st: idx for idx, st in enumerate(states)}

        # Count matrix: shape (K states, 4 conditions)
        counts = np.zeros((k_states, num_conds), dtype=np.float64)

        for i, ev_set in enumerate(raw_ev_sets):
            c_name = conditions_series[i]
            c_idx = TARGET_CONDITIONS.index(c_name)
            p_state = extract_node_states_from_patient(ev_set, [node])[name]
            st_idx = state_to_idx[p_state]
            counts[st_idx, c_idx] += 1.0

        # Apply Laplace smoothing: (count + alpha) / (N_cond + K * alpha)
        # N_cond is sum of counts over all states for that condition
        n_cond = counts.sum(axis=0, keepdims=True)
        smoothed_cpt = (counts + alpha) / (n_cond + k_states * alpha)

        # TabularCPD expects values of shape (variable_card, product of evidence_card)
        cpd = TabularCPD(
            variable=name,
            variable_card=k_states,
            values=smoothed_cpt.tolist(),
            evidence=["Condition"],
            evidence_card=[num_conds],
            state_names={
                name: states,
                "Condition": TARGET_CONDITIONS,
            },
        )
        node_cpds.append(cpd)
        model.add_cpds(cpd)

    # Validate model CPD consistency
    assert model.check_model(), "Bayesian Network model check failed!"
    return model
