# Differential Diagnosis Triage System (Chest Pain Prototype)

> **Important Limitation / Disclaimer:**  
> DDXPlus is a synthetic differential-diagnosis dataset. Probabilities learned from it are for demonstrating the method and are not clinical estimates. The system is an educational prototype for decision support and must not be used for real patient care.

---

## 1. Project Overview

This repository contains an academic prototype for differential diagnosis chest-pain triage developed for the course *"Intelligent Model Design Using AI"*.

The pipeline:
1. **Extracts structured clinical findings** from free text using biomedical NLP (scispaCy `en_core_sci_md` + phrase matching and clause-aware negation).
2. **Estimates disease probabilities** across 4 conditions (**ACS**, **GERD**, **Panic**, **Pericarditis**) using a Bayesian Network (`pgmpy`).
3. **Engages in active questioning** by ranking unasked symptoms via **Expected Information Gain (EIG)**.
4. **Applies explicit stopping & safety policies** (e.g. ACS risk threshold escalation) to classify triage severity (`URGENT`, `SEMI-URGENT`, `ROUTINE`).
5. **Interactive UI** provided via Streamlit.

---

## 2. Directory Layout

```
project/
  README.md
  requirements.txt
  PROJECT_BRIEF.md
  config/nodes.json            # Shared Bayesian node table
  data/raw/                    # Raw DDXPlus files (gitignored)
  data/processed/              # Processed & filtered datasets (gitignored)
  src/
    data/                      # DDXPlus loading and evidence matrix construction
    bayesian/                  # Bayesian network construction and inference
    active_query/              # EIG calculation, stopping rule & triage escalation
    nlp/                       # Entity extraction, phrase matching, clause-based negation
    pipeline/                  # End-to-end interactive diagnostic session
    mocks/                     # Fast mock interfaces for parallel development
  app/                         # Streamlit demo application
  evaluation/                  # Benchmarking (NLP baseline vs pipeline, BN eval, query eval)
  tests/                       # Pytest test suite
  results/                     # Evaluation metrics and artifact logs
  docs/                        # Plain-English viva explanations for each module
  notebooks/Milestone2_.ipynb  # Exploration & reference notebook
```

---

## 3. Environment Setup

### Prerequisites
- Python **3.11**

### Setup Steps
```bash
# 1. Create and activate virtual environment
py -3.11 -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify test runner
pytest
```
