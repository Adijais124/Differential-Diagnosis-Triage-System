"""Data loading and filtering utilities for DDXPlus dataset.

Maps raw DDXPlus pathologies to our 4 target conditions:
- ACS (merging 'Unstable angina', 'Stable angina', 'Possible NSTEMI / STEMI')
- GERD ('GERD')
- Panic ('Panic attack')
- Pericarditis ('Pericarditis')
"""

import json
import os
import zipfile
import pandas as pd

CONDITION_MAP = {
    "Unstable angina": "ACS",
    "Stable angina": "ACS",
    "Possible NSTEMI / STEMI": "ACS",
    "GERD": "GERD",
    "Panic attack": "Panic",
    "Pericarditis": "Pericarditis",
}

TARGET_CONDITIONS = ["ACS", "GERD", "Panic", "Pericarditis"]


def load_conditions_metadata(raw_dir: str = "data/raw") -> dict:
    """Load release_conditions.json."""
    path = os.path.join(raw_dir, "release_conditions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evidences_metadata(raw_dir: str = "data/raw") -> dict:
    """Load release_evidences.json."""
    path = os.path.join(raw_dir, "release_evidences.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_patients(split: str = "train", raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load raw patient dataframe for a given split (train/validate/test).

    Supports both .zip and .csv raw formats. Only loads columns needed for
    clinical differential diagnosis.
    """
    needed_cols = ["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"]

    # 1. Check for zipped train file
    zip_path = os.path.join(raw_dir, f"release_{split}_patients.zip")
    csv_path = os.path.join(raw_dir, f"release_{split}_patients.csv")
    no_ext_path = os.path.join(raw_dir, f"release_{split}_patients")

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            inner_name = f"release_{split}_patients"
            if inner_name in z.namelist():
                with z.open(inner_name) as f:
                    return pd.read_csv(f, usecols=lambda c: c in needed_cols)
            else:
                first = z.namelist()[0]
                with z.open(first) as f:
                    return pd.read_csv(f, usecols=lambda c: c in needed_cols)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path, usecols=lambda c: c in needed_cols)
    elif os.path.exists(no_ext_path):
        return pd.read_csv(no_ext_path, usecols=lambda c: c in needed_cols)
    else:
        raise FileNotFoundError(f"Could not find patient file for split '{split}' in {raw_dir}")


def filter_to_target_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows to the target pathologies and map to the 4 conditions.

    Merges Unstable angina, Stable angina, and Possible NSTEMI / STEMI into ACS.
    """
    mask = df["PATHOLOGY"].isin(CONDITION_MAP)
    filtered = df[mask].copy()
    filtered["CONDITION"] = filtered["PATHOLOGY"].map(CONDITION_MAP)
    return filtered
