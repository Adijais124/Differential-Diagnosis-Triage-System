"""Clause-aware clinical negation detector.

Fixes Bug B: ensures negation applies per-mention within its syntactic clause,
stopping at contrastive conjunctions ('but', 'however') and affirming verbs ('reports').
Handles list negation ('no X, Y or Z') and idiomatic affirmative expressions ('can't catch breath').
"""

import re


# Cues indicating that subsequent mentions in the clause are negated
NEGATION_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdenying\b",
    r"\bwithout\b",
    r"\bnever\b",
    r"\bfree of\b",
    r"\bnegative for\b",
    r"\brules out\b",
    r"\bruled out\b",
    r"\bnon-smoker\b",
    r"\bnon smoker\b",
    r"\bdoesn't have\b",
    r"\bdoes not have\b",
    r"\bdid not have\b",
    r"\bhas no\b",
    r"\bhave no\b",
]

# Patterns that break/terminate an active negation scope
CLAUSE_TERMINATORS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\byet\b",
    r"\bexcept\b",
    r"\breports\b",
    r"\breported\b",
    r"\breporting\b",
    r"\bcomplains of\b",
    r"\bcomplaining of\b",
    r"\bwith\b",
    r"\bdeveloped\b",
    r"\bnoted\b",
    r"\bnow has\b",
]

# Idiomatic phrases containing negation words that actually express POSITIVE symptoms
POSITIVE_IDIOMS = [
    "can't catch",
    "cannot catch",
    "unable to catch",
    "can't breathe",
    "cannot breathe",
    "unable to breathe",
    "can't lie flat",
    "cannot lie flat",
]


def is_mention_negated(
    full_text: str,
    match_start: int,
    match_end: int,
) -> bool:
    """Determine whether a specific entity match is negated within its clause.

    Args:
        full_text: The complete input text.
        match_start: Start character index of the matched symptom phrase.
        match_end: End character index of the matched symptom phrase.

    Returns:
        True if the mention is negated, False if affirmative.
    """
    matched_phrase = full_text[match_start:match_end].lower()

    # 1. Check positive idioms: phrases like "can't catch her breath" express POSITIVE dyspnea
    for idiom in POSITIVE_IDIOMS:
        if idiom in matched_phrase:
            return False

    # 2. Extract preceding window up to sentence or strong punctuation boundary
    prefix = full_text[:match_start]
    # Split on sentence boundaries (. ; ! ?)
    sentence_chunks = re.split(r"[.;!?\n]", prefix)
    current_clause = sentence_chunks[-1]

    # 3. Check for phrase-level negation directly inside or immediately adjacent (e.g. non-smoker)
    immediate_pre = current_clause[-20:].lower()
    if "non-smoker" in immediate_pre or "non smoker" in immediate_pre:
        return True

    # 4. If contrastive conjunction or affirming verb appears between a negation cue and the match,
    # the negation is terminated.
    # Split current_clause on contrastive/affirming terminators
    clause_parts = re.split("|".join(CLAUSE_TERMINATORS), current_clause, flags=re.IGNORECASE)
    active_scope = clause_parts[-1].lower()

    # 5. Check if active_scope contains an uncancelled negation cue
    for cue_pattern in NEGATION_CUES:
        if re.search(cue_pattern, active_scope):
            return True

    return False
