# Project Progress Log

## Task 0: Repository Setup & Verification
- **What was done:** Initialized modular repository structure, Python 3.11 virtual environment (.venv), pytest configuration, requirements.txt with pinned dependencies, and baseline documentation/tests.
- **Key numbers:** 1 smoke test passed in 0.15s; scispaCy biomedical model `en_core_sci_md` v0.5.4 loaded successfully.
- **Any concern:** None; DDXPlus dataset files placed in `data/raw/` and confirmed gitignored.

## Task 1: Data Ingestion & Preprocessing
- **What was done:** Implemented DDXPlus data loader, 4-condition filtering with ACS merge, multi-valued evidence value preservation, and exported train/validate/test splits.
- **Key numbers:** Train rows: 133,282 (ACS: 59,499 [44.64%], GERD: 25,979, Panic: 25,019, Pericarditis: 22,785) — exact match to Section 4; Val: 17,726; Test: 18,202.
- **Any concern:** None; all 6 pytest tests passed; numbers verified against results/train_counts.json.

## Task 2: Candidate Node Table Selection
- **What was done:** Computed P(E|C) spread and flags for all candidate evidences; selected 14 discriminative, askable nodes in config/nodes.json (status: PROVISIONAL).
- **Key numbers:** 14 nodes (12 binary + 2 multi-valued: pain character & radiation); excluded leaky E_3 (pericarditis history, spread 0.837) and awkward codes.
- **Any concern:** None; provisional list documented in docs/task2_nodes.md for future team review.


