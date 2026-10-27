"""Tests for Task 8: Streamlit Web Application."""

import os
import py_compile


def test_streamlit_app_compiles():
    """Verify that app/streamlit_app.py compiles without syntax errors."""
    app_path = os.path.join("app", "streamlit_app.py")
    assert os.path.exists(app_path), "app/streamlit_app.py must exist"
    py_compile.compile(app_path, doraise=True)


def test_streamlit_app_has_disclaimer():
    """Verify that app/streamlit_app.py contains the mandatory educational disclaimer."""
    app_path = os.path.join("app", "streamlit_app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Educational prototype" in content or "EDUCATIONAL PROTOTYPE" in content
    assert "Not medical advice" in content or "NOT MEDICAL ADVICE" in content
