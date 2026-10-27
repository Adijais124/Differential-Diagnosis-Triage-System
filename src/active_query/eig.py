"""Expected Information Gain (EIG) calculation engine.

Calculates Shannon entropy and EIG for candidate clinical questions.
Supports both fast direct Naive-Bayes computation and pgmpy VariableElimination.
"""

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from src.bayesian.inference import posterior


def shannon_entropy(probs: list[float] | np.ndarray) -> float:
    """Calculate Shannon entropy H in bits (log2).

    H(P) = - sum_i p_i * log2(p_i) for p_i > 0.
    """
    p = np.array(list(probs), dtype=np.float64)
    p = p[p > 0.0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def expected_info_gain(
    model: BayesianNetwork,
    question: str,
    evidence: dict[str, str],
    use_fast_bayes: bool = True,
) -> float:
    """Calculate Expected Information Gain (in bits) for asking 'question'.

    EIG(Q; C | e) = H(C | e) - sum_a P(Q=a | e) * H(C | e, Q=a)
    """
    # Current posterior and entropy
    current_post = posterior(model, evidence, use_fast_bayes=use_fast_bayes)
    h_now = shannon_entropy(list(current_post.values()))

    # Possible answers/states for the question node
    cpd = model.get_cpds(question)
    q_states = cpd.state_names[question]
    cond_states = model.get_cpds("Condition").state_names["Condition"]

    if not use_fast_bayes:
        # Standard pgmpy VariableElimination
        infer = VariableElimination(model)
        valid_ev = {k: v for k, v in evidence.items() if k in model.nodes() and k != "Condition"}
        pq = infer.query([question], evidence=valid_ev, show_progress=False)

        exp_h = 0.0
        for i, ans in enumerate(pq.state_names[question]):
            p_ans = float(pq.values[i])
            if p_ans > 0.0:
                post_with_ans = posterior(
                    model,
                    {**evidence, question: ans},
                    use_fast_bayes=False,
                )
                exp_h += p_ans * shannon_entropy(list(post_with_ans.values()))

        eig = max(0.0, h_now - exp_h)
        return float(eig)

    # Fast direct Naive Bayes calculation:
    # P(Q=a | e) = sum_c P(Q=a | C=c) * P(C=c | e)
    post_vec = np.array([current_post[c] for c in cond_states], dtype=np.float64)  # shape (4,)

    # cpd.values has shape (variable_card, condition_card)
    p_q_given_c = cpd.values  # shape (K, 4)

    # Marginal probability of each answer given current evidence: shape (K,)
    p_answers = np.dot(p_q_given_c, post_vec)
    # Ensure numerical normalization
    p_answers_sum = np.sum(p_answers)
    if p_answers_sum > 0:
        p_answers /= p_answers_sum

    exp_h = 0.0
    for st_idx, ans in enumerate(q_states):
        p_ans = p_answers[st_idx]
        if p_ans > 1e-9:
            hypo_evidence = dict(evidence)
            hypo_evidence[question] = ans
            hypo_post = posterior(model, hypo_evidence, use_fast_bayes=True)
            exp_h += p_ans * shannon_entropy(list(hypo_post.values()))

    eig = max(0.0, h_now - exp_h)
    return float(eig)


def rank_unanswered_questions(
    model: BayesianNetwork,
    evidence: dict[str, str],
    answered: set[str],
    candidate_nodes: list[dict] | None = None,
    use_fast_bayes: bool = True,
) -> list[dict]:
    """Compute and rank EIG for all unanswered candidate nodes.

    Returns list of dicts sorted descending by EIG:
        [{"name": "sweating", "eig": 0.407, "question_text": ...}, ...]
    """
    if candidate_nodes is not None:
        candidates = [n for n in candidate_nodes if n["name"] not in answered and n["name"] not in evidence]
    else:
        all_nodes = [n for n in model.nodes() if n != "Condition"]
        candidates = [{"name": n} for n in all_nodes if n not in answered and n not in evidence]

    scored = []
    for node in candidates:
        name = node["name"]
        eig_val = expected_info_gain(model, name, evidence, use_fast_bayes=use_fast_bayes)
        scored.append({
            "name": name,
            "eig": round(eig_val, 4),
            "question_text": node.get("question_text", f"Question about {name}"),
        })

    scored.sort(key=lambda x: x["eig"], reverse=True)
    return scored
