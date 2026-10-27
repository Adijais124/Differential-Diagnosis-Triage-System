"""Active querying policy and clinical safety escalation rules.

Implements next_step() per PROJECT_BRIEF Section 7.1.
Enforces stopping criteria:
1. Max 8 questions
2. Best remaining EIG < 0.02 bits
3. High confidence (top posterior >= 0.90), EXCEPT when ACS safety rule applies.

Safety Rule: Triage is risk-based, not merely argmax.
If P(ACS) >= acs_risk_threshold (default 0.25), output is URGENT.
"""

from pgmpy.models import BayesianNetwork
from src.active_query.eig import rank_unanswered_questions
from src.bayesian.build_model import load_nodes_config
from src.bayesian.inference import posterior

DEFAULT_POLICY_CONFIG = {
    "max_questions": 8,
    "min_eig": 0.02,
    "confidence_stop": 0.90,
    "acs_risk_threshold": 0.25,  # Assumption tuned on validate split
    "pericarditis_urgent_threshold": 0.70,
}


def determine_triage(
    posterior_dist: dict[str, float],
    cfg: dict,
) -> tuple[str, str]:
    """Determine clinical triage severity and accompanying plain-English rationale.

    Triage levels:
    - URGENT: High acute risk requiring immediate emergency evaluation (ACS risk or high-prob Pericarditis).
    - SEMI-URGENT: Moderate acuity requiring timely medical assessment (GERD/Panic with ACS safely excluded).
    - ROUTINE: Low acuity suitable for scheduled primary care outpatient review.
    """
    p_acs = posterior_dist.get("ACS", 0.0)
    top_cond, top_prob = max(posterior_dist.items(), key=lambda x: x[1])

    acs_thresh = cfg.get("acs_risk_threshold", 0.25)
    peri_thresh = cfg.get("pericarditis_urgent_threshold", 0.70)

    # Safety rule: ACS risk escalation takes absolute precedence
    if p_acs >= acs_thresh:
        return (
            "URGENT",
            f"SAFETY ESCALATION: P(ACS) is {p_acs:.1%} (exceeds safety threshold of {acs_thresh:.1%}). "
            f"Immediate emergency cardiac evaluation required regardless of leading diagnosis.",
        )

    if top_cond == "Pericarditis":
        if top_prob >= peri_thresh:
            return (
                "URGENT",
                f"Probable acute pericarditis ({top_prob:.1%}). Urgent clinical evaluation "
                f"required to evaluate effusion risk and start anti-inflammatory therapy.",
            )
        else:
            return (
                "SEMI-URGENT",
                f"Possible pericarditis ({top_prob:.1%}). Requires same-day medical assessment.",
            )

    if top_cond == "GERD":
        if top_prob >= 0.85:
            return (
                "ROUTINE",
                f"High confidence Gastroesophageal Reflux Disease ({top_prob:.1%}) with ACS ruled out ({p_acs:.1%}). "
                f"Suitable for outpatient primary care management and acid suppression trial.",
            )
        else:
            return (
                "SEMI-URGENT",
                f"Likely GERD ({top_prob:.1%}) with low ACS risk ({p_acs:.1%}). Recommend medical review.",
            )

    if top_cond == "Panic":
        return (
            "SEMI-URGENT",
            f"Probable panic attack / anxiety presentation ({top_prob:.1%}) with cardiac ischemia ruled out ({p_acs:.1%}). "
            f"Supportive care and outpatient mental health follow-up recommended.",
        )

    return ("SEMI-URGENT", f"Leading condition {top_cond} ({top_prob:.1%}). Further clinical assessment advised.")


def next_step(
    model: BayesianNetwork,
    evidence: dict[str, str],
    answered: set[str],
    cfg: dict | None = None,
) -> dict:
    """Determine next action in the active questioning loop.

    Returns:
        dict matching Section 7.1 contract:
        {
          "action": "ask" | "stop",
          "question": node_name | None,
          "eig": float,
          "posterior": dict[str, float],
          "triage": "URGENT" | "SEMI-URGENT" | "ROUTINE" | None,
          "reason": str,
        }
    """
    config = dict(DEFAULT_POLICY_CONFIG)
    if cfg is not None:
        config.update(cfg)

    # 1. Compute current posterior
    current_post = posterior(model, evidence, use_fast_bayes=True)
    top_cond, top_prob = max(current_post.items(), key=lambda x: x[1])
    p_acs = current_post.get("ACS", 0.0)

    # Load candidate node definitions
    try:
        nodes_cfg = load_nodes_config()
        candidate_nodes = nodes_cfg.get("nodes", [])
    except Exception:
        candidate_nodes = None

    # 2. Check stopping rule: max questions reached
    if len(answered) >= config["max_questions"]:
        triage_level, triage_reason = determine_triage(current_post, config)
        return {
            "action": "stop",
            "question": None,
            "eig": 0.0,
            "posterior": current_post,
            "triage": triage_level,
            "reason": f"Maximum question cap ({config['max_questions']}) reached. {triage_reason}",
        }

    # 3. Rank remaining unanswered questions by EIG
    ranked = rank_unanswered_questions(
        model,
        evidence=evidence,
        answered=answered,
        candidate_nodes=candidate_nodes,
        use_fast_bayes=True,
    )

    # If no remaining candidate questions
    if not ranked:
        triage_level, triage_reason = determine_triage(current_post, config)
        return {
            "action": "stop",
            "question": None,
            "eig": 0.0,
            "posterior": current_post,
            "triage": triage_level,
            "reason": f"All available questions have been answered. {triage_reason}",
        }

    best_q = ranked[0]
    best_eig = best_q["eig"]

    # 4. Check stopping rule: EIG diminishing returns (< min_eig)
    if best_eig < config["min_eig"]:
        triage_level, triage_reason = determine_triage(current_post, config)
        return {
            "action": "stop",
            "question": None,
            "eig": best_eig,
            "posterior": current_post,
            "triage": triage_level,
            "reason": f"Information gain is negligible (best EIG {best_eig:.3f} < {config['min_eig']:.3f} bits). {triage_reason}",
        }

    # 5. Check stopping rule: high confidence stop (top_prob >= 0.90)
    # SAFETY OVERRIDE: Never stop on confidence alone if P(ACS) is elevated
    is_acs_risk = p_acs >= config["acs_risk_threshold"]
    if top_prob >= config["confidence_stop"] and not is_acs_risk:
        triage_level, triage_reason = determine_triage(current_post, config)
        return {
            "action": "stop",
            "question": None,
            "eig": best_eig,
            "posterior": current_post,
            "triage": triage_level,
            "reason": f"High confidence reached for {top_cond} ({top_prob:.1%}) with ACS ruled out ({p_acs:.1%}). {triage_reason}",
        }

    # 6. Action: ASK next best question
    return {
        "action": "ask",
        "question": best_q["name"],
        "eig": best_eig,
        "posterior": current_post,
        "triage": None,
        "reason": f"Asking most informative question '{best_q['name']}' (EIG = {best_eig:.3f} bits).",
    }
