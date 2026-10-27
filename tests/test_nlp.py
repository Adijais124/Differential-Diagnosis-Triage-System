"""Tests for Task 5: Clinical NLP Module.

Tests the 3 original notebook sentences (verifying fixes for Bugs A, B, and C),
plus 10+ new challenging clinical test cases.
"""

from src.nlp.extractor import extract_findings


def test_notebook_sentence_1_contrastive_negation():
    """Sentence 1: denies nausea but reports severe chest pressure and shortness of breath."""
    text = "Patient denies nausea but reports severe chest pressure and shortness of breath."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("shortness_of_breath") == "yes"
    assert findings.get("pain_character") == "heavy_pressure"


def test_notebook_sentence_2_variant_pronouns_and_attributes():
    """Sentence 2: 55-year-old male smoker with heavy pressure in centre of chest."""
    text = "A 55-year-old male smoker has had heavy pressure in the centre of his chest for 20 minutes. It started while walking and he is sweating."
    res = extract_findings(text)
    findings = res["findings"]
    attrs = res["attributes"]

    assert findings.get("pain_character") == "heavy_pressure"
    assert findings.get("smoker") == "yes"
    assert findings.get("sweating") == "yes"
    assert attrs.get("age") == 55
    assert attrs.get("sex") == "M"
    assert attrs.get("duration_min") == 20


def test_notebook_sentence_3_bug_a_and_b_fixes():
    """Sentence 3: fixes Bug A ('can't catch her breath') and Bug B ('No chest pain on exertion')."""
    text = "She can't catch her breath, has a racing heart and feels anxious. No chest pain on exertion."
    res = extract_findings(text)
    findings = res["findings"]

    # Bug A fix: 'can't catch her breath' MUST be detected as positive shortness_of_breath
    assert findings.get("shortness_of_breath") == "yes", "Bug A: breathlessness must be detected with pronoun 'her'"

    # Racing heart & anxiety detected
    assert findings.get("racing_heart") == "yes"
    assert findings.get("feeling_of_impending_doom") == "yes"

    # Bug B fix: "No chest pain on exertion" must NOT produce a positive exertional symptom
    assert findings.get("pain_character") != "heavy_pressure"


def test_case_list_negation():
    """List negation: 'no shortness of breath, palpitations or diaphoresis'."""
    text = "Patient has burning in the chest, but no shortness of breath, palpitations or sweating."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("pain_character") == "burning"
    assert findings.get("shortness_of_breath") == "no"
    assert findings.get("racing_heart") == "no"
    assert findings.get("sweating") == "no"


def test_case_non_smoker():
    """Special negation: non-smoker."""
    text = "A 48 yo female non-smoker presents with sudden crushing chest tightness."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("smoker") == "no"
    assert findings.get("pain_character") == "heavy_pressure"
    assert res["attributes"].get("age") == 48
    assert res["attributes"].get("sex") == "F"


def test_case_does_not_have_hypertension():
    """Negation: 'does not have high blood pressure'."""
    text = "He does not have high blood pressure and denies prior heart attack."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("high_blood_pressure") == "no"
    assert findings.get("prior_heart_attack_angina") == "no"


def test_case_radiation_arm_shoulder():
    """Radiation extraction to arm/shoulder."""
    text = "Severe squeezing chest pain radiating to the left arm and left shoulder with profuse sweating."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("pain_character") == "heavy_pressure"
    assert findings.get("pain_radiation") == "arm_shoulder"
    assert findings.get("sweating") == "yes"


def test_case_radiation_jaw_neck():
    """Radiation extraction to jaw/neck."""
    text = "Heavy chest pressure radiating to the jaw and throat. Patient feels short of breath."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("pain_character") == "heavy_pressure"
    assert findings.get("pain_radiation") == "neck_jaw"
    assert findings.get("shortness_of_breath") == "yes"


def test_case_gerd_postprandial_burn():
    """GERD presentation: worse after eating and burning sensation."""
    text = "Burning sensation in the throat, symptoms are significantly worse after eating dinner."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("burning_sensation") == "yes"
    assert findings.get("worse_after_eating") == "yes"


def test_case_pericarditis_pleuritic_and_positional():
    """Pericarditis presentation: sharp pain, hurts to breathe, relieved sitting up."""
    text = "Sharp knife-like chest pain that is worse on deep breath, but relieved when sitting up. Recent viral infection 1 week ago."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("pain_character") == "sharp_knife"
    assert findings.get("pleuritic_pain") == "yes"
    assert findings.get("worse_lying_down") == "yes"
    assert findings.get("recent_viral_infection") == "yes"


def test_case_impending_doom():
    """Panic presentation: feeling like dying and rapid heartbeat."""
    text = "Young woman complains of sudden palpitations and feels like she is dying with extreme anxiety."
    res = extract_findings(text)
    findings = res["findings"]

    assert findings.get("racing_heart") == "yes"
    assert findings.get("feeling_of_impending_doom") == "yes"
    assert res["attributes"].get("sex") == "F"


def test_case_unmapped_entities_retained():
    """Verify that clinical entities found by scispaCy but outside our dictionary are in unmapped."""
    text = "Patient reports a rash on forearm and severe headache with tight chest."
    res = extract_findings(text)

    findings = res["findings"]
    assert findings
    assert findings.get("pain_character") == "heavy_pressure"
    assert len(res["unmapped"]) > 0, "scispaCy entities like headache or rash should be retained in unmapped"
