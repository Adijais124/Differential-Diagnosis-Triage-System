# Task 1: Data Module (DDXPlus Ingestion & Preprocessing)

## Plain-English Viva Explanation

### What was done in this task?
1. **Loaded the Official DDXPlus Splits**:
   - Ingested the raw DDXPlus dataset from `data/raw/` (1,025,602 total rows across 49 pathologies).
   - Ingested validation and test splits (`validate.csv`, `test.csv`) to ensure strict split hygiene (Rule: learn CPTs on train, tune on validate, report on test).

2. **Condition Mapping & Merging**:
   - Filtered 49 pathologies down to our 4 target conditions:
     - **ACS (Acute Coronary Syndrome)**: Clinically, patients presenting with chest pain from ischemic origin may have *Unstable angina*, *Stable angina*, or *Possible NSTEMI / STEMI*. DDXPlus models them as separate diseases, but in an emergency triage setting, all three require immediate cardiac workup. Therefore, they are merged into a single ACS class.
     - **GERD**: Gastroesophageal reflux disease, representing non-cardiac gastrointestinal chest/epigastric discomfort.
     - **Panic**: Panic attack, representing anxiety-induced adrenergic chest tightness, palpitations, and breathlessness.
     - **Pericarditis**: Pericardial inflammation, replacing musculoskeletal conditions because DDXPlus contains no clean musculoskeletal label.
   - Resulting train slice: **133,282 patients** (ACS: 59,499 [44.6%], GERD: 25,979, Panic: 25,019, Pericarditis: 22,785).

3. **Multi-Valued Evidence Value Preservation**:
   - Solved Bug E: standard base-code collapsing (`split('_@_')[0]`) removes critical diagnostic signal because almost all chest pain patients have pain present (`E_54=1.00`). By keeping the exact values (e.g. `E_54_@_V_183` for heavy pressure vs `V_181` for burning vs `V_192` for sharp pain), the Bayesian model can differentiate ACS (pressure/crushing) from GERD (burning) and Pericarditis (sharp/knife-like).

4. **Saved Processed Data**:
   - Stored fast Parquet files in `data/processed/` and verified exact match with `PROJECT_BRIEF.md` counts.

### Viva Q&A Cheat Sheet
- **Q: Why merge Unstable Angina, Stable Angina, and NSTEMI/STEMI into ACS?**
  *A: In initial emergency triage, differentiating NSTEMI from unstable angina requires serial troponins and ECGs that cannot be determined purely from patient intake history. All require urgent escalation under the ACS umbrella.*
- **Q: What is the difference between simulator semantics and live clinical semantics?**
  *A: In the DDXPlus simulator, if an evidence code is not in a synthetic patient's list, the patient does not have it (simulated negative). In the live app, an unasked symptom is UNOBSERVED and must be marginalised over by the Bayesian Network, never assumed to be negative.*
