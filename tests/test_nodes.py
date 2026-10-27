"""Tests for Task 2: Candidate node configuration."""

import json
import os


def test_nodes_config_valid():
    """Verify config/nodes.json structure and rules."""
    path = os.path.join("config", "nodes.json")
    assert os.path.exists(path), "config/nodes.json must exist"

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg.get("status") == "PROVISIONAL", "Status must be PROVISIONAL per instructions"
    nodes = cfg.get("nodes", [])

    # Range must be 12-15
    assert 12 <= len(nodes) <= 15, f"Node count must be 12-15, found {len(nodes)}"

    # Required fields per node
    names = set()
    codes = set()
    for node in nodes:
        assert "name" in node
        assert "ddxplus_code" in node
        assert "states" in node
        assert "question_text" in node
        assert "nlp_phrases_key" in node
        assert len(node["states"]) >= 2
        names.add(node["name"])
        codes.add(node["ddxplus_code"])

    assert len(names) == len(nodes), "Node names must be unique"

    # Pain character and radiation must exist
    assert "pain_character" in names
    assert "pain_radiation" in names

    # Leaky code E_3 must NOT be present
    assert "E_3" not in codes, "Leaky code E_3 (history of pericarditis) must not be included"
