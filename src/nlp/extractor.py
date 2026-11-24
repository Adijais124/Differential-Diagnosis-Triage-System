"""Clinical text extractor using scispaCy and phrase matching.

Extracts structured symptoms, patient demographics, and records unmapped entities.
Implements extract_findings(text: str) -> dict matching PROJECT_BRIEF Section 7.1.
"""

import re
import spacy
from spacy.matcher import PhraseMatcher

from src.nlp.negation import is_mention_negated
from src.nlp.synonyms import build_synonym_dictionary

_NLP_MODEL = None
_MATCHER = None
_SYN_DICT = None

TYPO_CORRECTIONS = {
    r"\btightnes\b": "tightness",
    r"\bsweting\b": "sweating",
    r"\bsweeting\b": "sweating",
    r"\bshortnes\b": "shortness",
    r"\bbreth\b": "breath",
    r"\bpresure\b": "pressure",
    r"\bhart\b": "heart",
    r"\bnausia\b": "nausea",
    r"\bdizzyiness\b": "dizziness",
    r"\bradiatin\b": "radiating",
    r"\bpalpatations\b": "palpitations",
    r"\bpalpitation\b": "palpitations",
}

CHEST_CARDIO_KEYWORDS = {
    "chest", "heart", "cardiac", "pressure", "tightness", "tight", "heavy",
    "crushing", "squeezing", "burning", "heartburn", "acid", "reflux",
    "breath", "breathing", "dyspnea", "winded", "sweat", "sweating",
    "palpitation", "palpitations", "racing", "doom", "substernal",
    "breastbone", "sternum", "ribs", "angina", "arm", "shoulder",
    "jaw", "neck", "throat", "pleuritic", "pericarditis", "postprandial",
}


def normalize_clinical_text(text: str) -> str:
    """Correct frequent patient misspellings and typos."""
    normalized = text
    for pattern, repl in TYPO_CORRECTIONS.items():
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return normalized


def check_domain_relevance(text: str, findings: dict, unmapped: list[str]) -> tuple[bool, str]:
    """Check if presentation contains symptoms within cardiovascular/chest triage scope.

    Returns:
        (is_out_of_domain: bool, reason: str)
    """
    clean_text = text.strip()
    if not clean_text:
        return True, "No clinical symptoms were entered."

    low = clean_text.lower()
    has_cardio_keyword = any(re.search(rf"\b{kw}\b", low) for kw in CHEST_CARDIO_KEYWORDS)
    has_findings = len([k for k in findings if k != "has_chest_pain" or findings[k] == "yes"]) > 0

    if has_findings or has_cardio_keyword:
        return False, "Relevant thoracic/cardiovascular presentation."

    if unmapped:
        complaint = ", ".join(f"'{u}'" for u in unmapped[:3])
        return True, f"Non-cardiovascular complaint detected ({complaint})."

    return True, "No cardiovascular or chest-pain symptoms identified."



def get_nlp_pipeline():
    """Lazily load scispaCy pipeline and initialize PhraseMatcher."""
    global _NLP_MODEL, _MATCHER, _SYN_DICT
    if _NLP_MODEL is None:
        _NLP_MODEL = spacy.load("en_core_sci_md")
        _SYN_DICT = build_synonym_dictionary()
        _MATCHER = PhraseMatcher(_NLP_MODEL.vocab, attr="LOWER")

        for key, phrases in _SYN_DICT.items():
            patterns = [_NLP_MODEL.make_doc(p) for p in phrases]
            _MATCHER.add(key, patterns)

    return _NLP_MODEL, _MATCHER, _SYN_DICT


def extract_attributes(text: str) -> dict:
    """Extract age, sex, and duration from patient narrative via regex."""
    attrs = {}
    low = text.lower()

    # 1. Age (e.g. "55-year-old", "62 yo", "age 45")
    age_match = re.search(r"\b(\d{1,3})\s*[- ]?(?:year[ -]?old|yo|y/o|years of age)\b", low)
    if not age_match:
        age_match = re.search(r"\bage\s*[:]?\s*(\d{1,3})\b", low)
    if age_match:
        attrs["age"] = int(age_match.group(1))

    # 2. Sex ("male", "female", "man", "woman", "gentleman", "lady")
    if re.search(r"\b(female|woman|lady|she|her)\b", low):
        attrs["sex"] = "F"
    elif re.search(r"\b(male|man|gentleman|he|his|him)\b", low):
        attrs["sex"] = "M"

    # 3. Duration in minutes
    dur_match = re.search(r"\b(\d+)\s*(?:minutes|mins|min)\b", low)
    if dur_match:
        attrs["duration_min"] = int(dur_match.group(1))
    else:
        hour_match = re.search(r"\b(\d+)\s*(?:hours|hrs|hr|hour)\b", low)
        if hour_match:
            attrs["duration_min"] = int(hour_match.group(1)) * 60

    return attrs


def extract_findings(text: str) -> dict:
    """Extract clinical findings from free-text patient presentation.

    Returns:
        dict:
        {
          "findings": {"sweating": "yes", "shortness_of_breath": "yes", "pain_character": "heavy_pressure"},
          "attributes": {"age": 55, "sex": "M", "duration_min": 20},
          "unmapped": ["entities found by scispaCy but not in dictionary"],
          "is_out_of_domain": bool,
          "domain_reason": str,
        }
    """
    norm_text = normalize_clinical_text(text)
    nlp, matcher, syn_dict = get_nlp_pipeline()
    doc = nlp(norm_text)

    findings = {}
    matched_spans = []

    # 1. Match custom synonyms with PhraseMatcher
    matches = matcher(doc)
    for match_id, start, end in matches:
        key = nlp.vocab.strings[match_id]
        span = doc[start:end]
        matched_spans.append((span.start_char, span.end_char, key, span.text))

    # Also check special direct regexes (e.g. non-smoker, exertional chest pain)
    low_text = norm_text.lower()
    if re.search(r"\b(non-smoker|non smoker|denies smoking|never smoked)\b", low_text):
        findings["smoker"] = "no"

    # 2. Resolve negation and map to Bayesian nodes
    for start_char, end_char, key, span_text in matched_spans:
        negated = is_mention_negated(norm_text, start_char, end_char)
        state_str = "no" if negated else "yes"

        # Multi-valued mappings
        if key.startswith("pain_character_"):
            sub_state = key.replace("pain_character_", "")
            if not negated:
                findings["pain_character"] = sub_state
        elif key.startswith("pain_radiation_"):
            sub_state = key.replace("pain_radiation_", "")
            if not negated:
                findings["pain_radiation"] = sub_state
        elif key == "anxiety":
            findings["feeling_of_impending_doom"] = state_str
        elif key == "has_chest_pain":
            findings["has_chest_pain"] = state_str
        else:
            findings[key] = state_str

    # 3. Collect unmapped entities found by pretrained scispaCy NER
    unmapped = []
    for ent in doc.ents:
        # Check if ent overlaps with any of our matched spans
        overlaps = False
        for start_char, end_char, _, _ in matched_spans:
            if not (ent.end_char <= start_char or ent.start_char >= end_char):
                overlaps = True
                break
        if not overlaps:
            ent_clean = ent.text.strip().lower()
            # Filter out non-clinical stopwords or single characters
            if len(ent_clean) > 2 and ent_clean not in ["patient", "reports", "denies", "has", "feels"]:
                unmapped.append(ent.text)

    # 4. Extract demographic attributes
    attributes = extract_attributes(norm_text)

    # 5. Check domain relevance
    sorted_unmapped = sorted(list(set(unmapped)))
    is_ood, ood_reason = check_domain_relevance(text, findings, sorted_unmapped)

    return {
        "findings": findings,
        "attributes": attributes,
        "unmapped": sorted_unmapped,
        "is_out_of_domain": is_ood,
        "domain_reason": ood_reason,
    }
