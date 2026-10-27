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

## Task 3: Bayesian Model & Posterior Inference
- **What was done:** Implemented Naive Bayes structure with Laplace smoothing (alpha=1.0), posterior inference via Bayes rule and pgmpy VariableElimination, and empirical vs uniform prior benchmarking.
- **Key numbers:** Test accuracy: 99.98% (both priors); ACS recall: 1.0000 on 18,202 test patients; all CPT columns sum to 1.0; no 0/1 probabilities.
- **Any concern:** Synthetic data causes near-perfect separability; documented in docs/task3_bayesian.md and results/bn_eval.json.

## Task 4: EIG Engine & Stopping Policy
- **What was done:** Implemented Shannon entropy, active question ranking via EIG, stopping rules (max 8 questions, min EIG < 0.02 bits, high confidence stop), and ACS risk safety escalation.
- **Key numbers:** Exactly reproduced Milestone2 toy network EIG: Exertional 0.407 bits / Nausea 0.062 bits; validated ACS risk threshold at 0.25 on validation set.
- **Any concern:** Thresholds labeled as clinical assumptions and recorded in results/threshold_tuning.json.

## Task 5: Clinical NLP Module & Bug Fixes
- **What was done:** Implemented scispaCy + PhraseMatcher hybrid extractor with combinatorial variant generation (synonyms.py) and clause-level negation (negation.py); resolved bugs A, B, and C.
- **Key numbers:** 12/12 NLP unit tests passed (including all 3 original notebook sentences: breathlessness detected via pronoun expansion, exertional false positive eliminated, contrastive negation verified).
- **Any concern:** None; unmapped scispaCy biomedical entities preserved in result dictionary.

## Task 6: NLP Gold Set & Baseline Comparison
- **What was done:** Curated 50-sentence gold test corpus in evaluation/nlp_cases.json; benchmarked full pipeline against Milestone 2 rule-only baseline.
- **Key numbers:** Pipeline F1: 0.8022 vs Baseline F1: 0.6358 (+16.64% F1 gain); pipeline precision 0.8068 / recall 0.7978.
- **Any concern:** Corpus marked 'needs_human_review: true' (status: 'draft gold set, not yet reviewed') for team verification.






