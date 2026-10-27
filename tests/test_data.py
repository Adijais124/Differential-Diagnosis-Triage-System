"""Tests for Task 1: Data module (loading, filtering, evidence parsing)."""

import json
import os
import pytest
import pandas as pd
from src.data.load_ddxplus import (
    CONDITION_MAP,
    TARGET_CONDITIONS,
    load_conditions_metadata,
    load_evidences_metadata,
)
from src.data.build_evidence_table import parse_evidence_string


def test_metadata_loads():
    """Verify that conditions and evidences metadata load correctly."""
    conditions = load_conditions_metadata()
    evidences = load_evidences_metadata()
    assert len(conditions) >= 49, "Must contain all 49 DDXPlus conditions"
    assert len(evidences) >= 223, "Must contain all 223 DDXPlus evidence codes"


def test_condition_map_pathologies():
    """Verify that all mapped pathologies exist in DDXPlus conditions metadata."""
    conditions = load_conditions_metadata()
    for path_name, target in CONDITION_MAP.items():
        assert path_name in conditions, f"Pathology '{path_name}' not found in DDXPlus"
        assert target in TARGET_CONDITIONS, f"Target '{target}' not in 4 target conditions"


def test_no_unknown_pathology_in_processed():
    """Verify that processed data contains only the 4 allowed conditions."""
    train_path = os.path.join("data", "processed", "train_filtered.csv.gz")
    if not os.path.exists(train_path):
        pytest.skip("Processed train file does not exist yet; run pipeline first.")

    df = pd.read_csv(train_path, compression="gzip")
    unique_conditions = set(df["CONDITION"].unique())
    assert unique_conditions.issubset(set(TARGET_CONDITIONS)), (
        f"Found unknown conditions: {unique_conditions - set(TARGET_CONDITIONS)}"
    )
    assert not df["CONDITION"].isna().any(), "Found NaN in CONDITION"


def test_train_counts_match_brief():
    """Verify that train counts exactly match Section 4 of PROJECT_BRIEF.md."""
    results_path = os.path.join("results", "train_counts.json")
    if not os.path.exists(results_path):
        pytest.skip("results/train_counts.json does not exist yet; run pipeline first.")

    with open(results_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    train_stats = stats.get("train", {})
    assert train_stats.get("filtered_rows") == 133282, "Train count must be 133,282"
    counts = train_stats.get("counts", {})
    assert counts.get("ACS") == 59499, "ACS count must be 59,499"
    assert counts.get("GERD") == 25979, "GERD count must be 25,979"
    assert counts.get("Panic") == 25019, "Panic count must be 25,019"
    assert counts.get("Pericarditis") == 22785, "Pericarditis count must be 22,785"


def test_parse_evidence_multivalued():
    """Verify that multi-valued evidence keeps values for pain character & radiation."""
    sample = "['E_91', 'E_54_@_V_183', 'E_57_@_V_123']"
    base_codes, multivalued = parse_evidence_string(sample)

    assert "E_91" in base_codes
    assert "E_54" in base_codes
    assert "E_57" in base_codes
    assert multivalued.get("E_54") == "V_183", "E_54 pain character must preserve value V_183"
    assert multivalued.get("E_57") == "V_123", "E_57 radiation must preserve value V_123"
