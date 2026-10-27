"""Tests for Task 7: TriageSession pipeline glue."""

from src.pipeline.session import TriageSession


def test_session_lifecycle_acs_case():
    """Verify end-to-end active questioning session on classic ACS narrative."""
    session = TriageSession()
    text = "55-year-old male smoker with heavy chest pressure and diaphoresis."
    init_res = session.start_from_text(text)

    assert "pain_character" in init_res["findings"]
    assert init_res["findings"]["pain_character"] == "heavy_pressure"
    assert init_res["findings"]["sweating"] == "yes"
    assert init_res["attributes"]["age"] == 55

    # Run active questioning until completion
    turns = 0
    while not session.is_finished and turns < 10:
        step = session.get_current_step()
        if step["action"] == "stop":
            break
        q_name = step["question"]
        assert q_name not in session.evidence
        # Answer affirmative or negative
        session.submit_answer(q_name, "no")
        turns += 1

    assert session.is_finished is True
    assert session.final_result is not None
    assert session.final_result["triage"] == "URGENT"
    assert len(session.final_result["reason"]) > 0


def test_session_skip_question():
    """Verify that answering 'unknown' or 'skip' does not insert evidence."""
    session = TriageSession()
    text = "She has heartburn worse after meals."
    session.start_from_text(text)

    step = session.get_current_step()
    if step["action"] == "ask":
        q_name = step["question"]
        session.submit_answer(q_name, "don't know")
        assert q_name in session.answered
        assert q_name not in session.evidence, "Skipped question must remain unobserved in evidence"
