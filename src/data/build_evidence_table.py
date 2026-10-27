"""Build processed evidence tables from filtered DDXPlus data.

Preserves multi-valued evidences (e.g. pain character E_54, radiation E_57,
pain location E_55, onset E_59) to prevent loss of diagnostic signal.
Saves processed splits and logs exact counts to results/train_counts.json.
"""

import ast
import json
import os
import pandas as pd
from src.data.load_ddxplus import (
    load_raw_patients,
    filter_to_target_conditions,
    load_evidences_metadata,
)


def parse_evidence_string(ev_str: str) -> tuple[set[str], dict[str, str]]:
    """Parse DDXPlus evidence string into base codes and multi-valued mapping.

    Returns:
        (base_codes, multivalued_dict):
          - base_codes: set of evidence IDs present (e.g. {'E_91', 'E_54', ...})
          - multivalued_dict: dict mapping evidence ID to specific value ID
                              (e.g. {'E_54': 'V_183', 'E_57': 'V_123'})
    """
    if not isinstance(ev_str, str) or not ev_str.strip():
        return set(), {}

    try:
        items = ast.literal_eval(ev_str)
    except Exception:
        # Fallback to simple string parsing if malformed
        items = [i.strip(" '\"[]") for i in ev_str.split(",") if i.strip(" '\"[]")]

    base_codes = set()
    multivalued = {}

    for item in items:
        if "_@_" in item:
            parts = item.split("_@_")
            code = parts[0]
            val = parts[1]
            base_codes.add(code)
            # Store the specific value (or comma-separated if multiple locations)
            if code in multivalued:
                multivalued[code] = multivalued[code] + "|" + val
            else:
                multivalued[code] = val
        else:
            base_codes.add(item)

    return base_codes, multivalued


def process_split(
    split: str = "train",
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
) -> tuple[pd.DataFrame, dict]:
    """Load, filter, parse, and save a patient split."""
    print(f"Loading raw patients for {split}...")
    raw_df = load_raw_patients(split=split, raw_dir=raw_dir)
    initial_len = len(raw_df)

    print(f"Filtering to 4 target conditions (ACS, GERD, Panic, Pericarditis)...")
    filtered_df = filter_to_target_conditions(raw_df)
    filtered_len = len(filtered_df)

    counts = filtered_df["CONDITION"].value_counts().to_dict()

    print(f"Split {split}: {initial_len} -> {filtered_len} rows.")
    print("Class distribution:", counts)

    # Save filtered split to processed directory (.csv.gz)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{split}_filtered.csv.gz")
    filtered_df.to_csv(out_file, index=False, compression="gzip")
    print(f"Saved processed {split} data to {out_file}")

    stats = {
        "split": split,
        "initial_rows": initial_len,
        "filtered_rows": filtered_len,
        "counts": counts,
    }
    return filtered_df, stats


def run_task1_data_pipeline(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    results_dir: str = "results",
) -> dict:
    """Run data pipeline for train, validate, and test.

    Saves verification counts to results/train_counts.json.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_stats = {}

    for split in ["train", "validate", "test"]:
        _, stats = process_split(split=split, raw_dir=raw_dir, out_dir=out_dir)
        all_stats[split] = stats

    results_file = os.path.join(results_dir, "train_counts.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved split statistics to {results_file}")

    return all_stats


if __name__ == "__main__":
    run_task1_data_pipeline()
