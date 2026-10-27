"""Bayesian Network inference engine and posterior calculation.

Implements posterior(model, evidence) per PROJECT_BRIEF Section 7.1.
Supports both pgmpy VariableElimination and fast direct Bayes-rule evaluation.
Unobserved variables are omitted from evidence (marginalised out, not assumed 'no').
"""

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from src.data.load_ddxplus import TARGET_CONDITIONS


def posterior(
    model: BayesianNetwork,
    evidence: dict[str, str],
    use_fast_bayes: bool = True,
) -> dict[str, float]:
    """Calculate posterior condition probabilities given observed findings.

    Args:
        model: Trained pgmpy BayesianNetwork instance.
        evidence: Dict of observed symptoms e.g. {"sweating": "yes", "pain_character": "heavy_pressure"}.
                  Nodes not in evidence are UNOBSERVED (marginalised over).
        use_fast_bayes: If True, uses direct Naive-Bayes vectorized computation.
                        If False, uses pgmpy VariableElimination.

    Returns:
        Dict mapping condition to probability, summing to 1.0.
        e.g. {"ACS": 0.65, "GERD": 0.15, "Panic": 0.10, "Pericarditis": 0.10}
    """
    # Filter evidence to valid variables in model
    valid_evidence = {
        k: v for k, v in evidence.items()
        if k in model.nodes() and k != "Condition"
    }

    if not use_fast_bayes:
        # Standard pgmpy VariableElimination inference
        infer = VariableElimination(model)
        query_res = infer.query(
            variables=["Condition"],
            evidence=valid_evidence,
            joint=False,
            show_progress=False,
        )["Condition"]
        probs = query_res.values
        st_names = query_res.state_names["Condition"]
        return {st: float(probs[i]) for i, st in enumerate(st_names)}

    # Fast direct Naive Bayes calculation:
    # P(C | e) \propto P(C) * \prod_{X \in e} P(X = e_X | C)
    cond_cpd = model.get_cpds("Condition")
    cond_states = cond_cpd.state_names["Condition"]
    prior = np.array(cond_cpd.values, dtype=np.float64)  # shape (4,)

    # Log-scale computation to prevent numerical underflow
    log_posterior = np.log(prior)

    for var_name, obs_state in valid_evidence.items():
        cpd = model.get_cpds(var_name)
        var_states = cpd.state_names[var_name]

        if obs_state not in var_states:
            # Skip invalid state or ignore
            continue

        st_idx = var_states.index(obs_state)
        # cpd.values has shape (variable_card, condition_card)
        p_x_given_c = cpd.values[st_idx, :]  # shape (4,)
        log_posterior += np.log(p_x_given_c)

    # Convert back from log-space with log-sum-exp normalization
    max_log = np.max(log_posterior)
    exp_unnorm = np.exp(log_posterior - max_log)
    normalized = exp_unnorm / np.sum(exp_unnorm)

    return {cond: float(normalized[i]) for i, cond in enumerate(cond_states)}
