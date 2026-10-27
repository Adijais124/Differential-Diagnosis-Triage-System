# PROJECT_BRIEF.md — Differential Diagnosis Triage System

Read this whole file before writing any code. It is the source of truth for the project.
Owner: a team of 4 undergraduate beginners. Course: "Intelligent Model Design Using AI".
This is an ACADEMIC PROTOTYPE for decision support. It is NOT a medical device and must never be presented as clinically validated.

---

## 1. What we are building

Chest-pain triage prototype. A patient description in plain text goes in; the system:

1. extracts structured symptoms with NLP (pretrained biomedical model + our own rules),
2. feeds them to a Bayesian Network (pgmpy) that gives a probability for each of 4 conditions,
3. picks the most informative unanswered question using Expected Information Gain (EIG),
4. updates the probabilities after each answer and repeats,
5. stops by an explicit rule and outputs a triage level with an explanation.

The NLP only extracts. The Bayesian Network does all probabilistic reasoning.
Active questioning (EIG) is the main intelligent component. A Streamlit app is the demo.

The 4 conditions: ACS, GERD, Panic, Pericarditis.
ACS = DDXPlus "Unstable angina" + "Stable angina" + "Possible NSTEMI / STEMI" merged.
(Musculoskeletal was the original idea; DDXPlus has no suitable match, so Pericarditis replaced it. Do not change this silently.)

## 2. Rules for the agent (non-negotiable)

1. **Small steps.** Do one task at a time (Section 8). Stop after each task and show: files changed, how to run it, test output.
2. **No invented numbers.** Every number used in the report or in the UI must come from code that was run and saved to `results/`. If something was not run, say "not run".
3. **No LLM / external API** anywhere in the pipeline. Everything runs locally.
4. **Do not change the stack** (Section 3) or add frameworks (no FastAPI, React, database, cloud) unless asked.
5. **Beginner-friendly code.** Short functions, clear names, comments explaining WHY. After each task, write a plain-English explanation in `docs/` of what the code does (the team must explain it in a viva).
6. **Tests.** Every module gets small pytest tests. Run them and show the output. Do not say "done" if tests were not run.
7. **Be honest about limits.** DDXPlus is synthetic. CPT values learned from it are not clinical probabilities. Add that caveat to any user-facing output and to docs.
8. **Ask before big decisions** (which symptoms to include, thresholds, changing conditions). Propose options with a recommendation, then wait.
9. Never print or commit secrets. Do not commit the raw DDXPlus files (large).

## 3. Stack (final)

- Python **3.11** (use a venv; spaCy 3.7.5 may fail to build on newer Python)
- pandas, numpy 1.26.4, scikit-learn
- spaCy 3.7.5, scispaCy 0.5.4, model `en_core_sci_md` 0.5.4
  (install: `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz`)
- pgmpy 0.1.26 (class is `BayesianNetwork` in this version; newer versions renamed it `DiscreteBayesianNetwork`)
- Streamlit + Plotly for the app
- pytest for tests; Git + GitHub
- `negspacy` is installed in the notebook but the current `extract()` does NOT use it (negation is regex-based). Do not depend on it. Either remove it or use it deliberately.

## 4. Current state (from the Colab notebook `Milestone2_.ipynb`)

Done and run:
- Loaded DDXPlus train: 1,025,602 rows, 49 pathologies, 223 evidence codes.
- Filtered to the 4 conditions: 133,282 rows — ACS 59,499 (44.6%), GERD 25,979, Panic 25,019, Pericarditis 22,785.
- Parsed `EVIDENCES` into a binary matrix (54 evidence columns after filtering).
- Computed P(evidence | condition) and the spread between conditions.
- NLP pipeline: scispaCy `en_core_sci_md` + `PhraseMatcher` synonym dictionary + regex negation + regex extras (age, sex, smoking, duration).
- Rule-only baseline (gave identical output to the pipeline on 3 test sentences).
- Toy pgmpy network (3 conditions, 3 made-up symptom CPTs) with inference.
- EIG on the toy network: Sweating=yes → Exertional EIG 0.407 bits vs Nausea 0.062 bits.

NOT done: real node list, real CPTs, 4-condition network, EIG on the real network, NLP↔BN integration, Streamlit app, evaluation, final report numbers.

## 5. Known bugs and weaknesses (fix these first)

A. Sentence "She can't catch her breath, has a racing heart and feels anxious. No chest pain on exertion." — `breathlessness` is NOT detected (the synonym list has "can't catch my breath" only; the sentence says "her").
B. Same sentence gives `exertional = True`; it should be False ("No chest pain on exertion"). Cause: `exertional` is set by a plain keyword check with no negation. Negation must apply to every node, per mention, within its clause (stop the negation scope at "but", "however", ";", ".", and at affirming verbs like "reports/has/with").
C. One synonym is copied from the demo sentence ("heavy pressure in the centre of his chest"). Replace with generated variants (e.g. `{pressure|tightness|heaviness} {in|in the centre of|...} {the|his|her|my} chest`).
D. The pretrained model adds little: scispaCy `en_core_sci_md` is a small CNN model (pipeline has `tok2vec`, not a transformer) and labels every entity just `ENTITY` (even "Patient", "reports"). Our dictionary does the real work. Be honest about this. See Task 9 for an optional, measured upgrade.
E. Preprocessing keeps only the base evidence code: `i.split('_@_')[0]`. Multi-valued evidences (e.g. `E_54_@_V_161`) lose their VALUE. The pain evidences (character, location, radiation, intensity, onset) are all 1.00 for every condition once collapsed, but their VALUES differ between conditions and carry much of the chest-pain signal. For the pain-related nodes keep the values (use `release_evidences.json` → `possible-values` / `value_meaning`).
F. Many P(evidence | condition) are exactly 0.00 or 1.00 (a synthetic-data artefact, e.g. "diabetes" 0.00 outside ACS). Use Laplace smoothing; never leave zero probabilities in a CPT. Also state in docs that accuracy on DDXPlus will look unrealistically high.
G. The notebook's install cells are duplicated/conflicting (red pip warnings). A clean `requirements.txt` replaces them in the repo.

## 6. DDXPlus facts

Files (from figshare): `release_conditions.json`, `release_evidences.json`, `release_train_patients.zip` (inner file is named `release_train_patients`, no extension), plus validate and test patient files.
Patient columns: `AGE, DIFFERENTIAL_DIAGNOSIS, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE`.
`EVIDENCES` is a string of a Python list, e.g. `"['E_91', 'E_53_@_V_12']"`. `E_xx_@_V_yy` = multi-valued evidence with value `V_yy`.
`release_evidences.json` maps each code to `question_en`, value meanings, and whether it is an antecedent or symptom.

Missing evidence: in the DDXPlus simulator a code NOT in a patient's list means the patient does not have it (that is how interactive systems built on DDXPlus answer questions). It is a statement about the SIMULATOR, not about real patients. Verify against the DDXPlus paper (Fansi Tchango et al., 2022, arXiv:2205.09148) and cite it.
In the live app: a question that has not been answered is UNOBSERVED (the Bayesian Network marginalises it). It is never treated as "no".

Splits: use the official train / validate / test files. Learn CPTs from train only; tune thresholds on validate; report final numbers on test. Do not learn and evaluate on the same rows.

Evidence codes seen so far in the spread analysis (verify each by its `question_en`; some such as head trauma or numbness are poor choices for a chest-pain demo):
E_66 shortness of breath · E_79 smoker · E_70 overweight · E_105 prior heart attack/angina · E_69 diabetes · E_71 high cholesterol · E_225 family cardiac history · E_81 chronic anxiety · E_155 racing heart · E_215 worse after eating · E_217 worse lying down · E_111 feel like dying · E_82 lightheaded/dizzy · E_171 detached from self · E_173 burning sensation · E_3 history of pericarditis (nearly gives away the label — avoid or flag) · E_0 recent viral infection.
Sweating, nausea, high blood pressure and the pain-character/location/radiation codes were not in the printed top-20; find them by searching `question_en`.

## 7. Target repo layout

```
project/
  README.md
  requirements.txt
  PROJECT_BRIEF.md
  config/nodes.json          # the shared node table (Section 7.1)
  data/raw/                  # DDXPlus files (gitignored)
  data/processed/            # filtered tables (gitignored if large)
  src/
    data/        load_ddxplus.py  build_evidence_table.py
    bayesian/    build_model.py  inference.py
    active_query/ eig.py  policy.py   # next question + stop rule + escalation
    nlp/         extractor.py  synonyms.py  negation.py
    pipeline/    session.py     # glue: text -> NLP -> BN -> EIG loop
  app/streamlit_app.py
  evaluation/    nlp_eval.py  bn_eval.py  query_eval.py
  tests/
  results/                      # every reported number is saved here
  docs/                         # plain-English explanation of each module
  notebooks/Milestone2_.ipynb   # the original, kept for reference
```

### 7.1 Interfaces (contracts — build against these so people can work in parallel)

**Node table `config/nodes.json`** — one entry per Bayesian node:
```json
{"name": "sweating", "ddxplus_code": "E_xx", "states": ["no","yes"],
 "question_text": "Have you had significantly increased sweating?",
 "nlp_phrases_key": "sweating"}
```
Multi-valued nodes (e.g. pain character, radiation) list their states and the mapping from DDXPlus value codes to states. Age is binned (e.g. "<40","40-60",">60"); sex is "M"/"F".

**NLP module**
```python
extract_findings(text: str) -> dict
# {"findings":   {"sweating": "yes", "nausea": "no", "pain_character": "pressure"},
#  "attributes": {"age": 55, "sex": "M", "duration_min": 20},
#  "unmapped":   ["burning sensation"]}      # entities the pretrained model found but we could not map
```
A node absent from `findings` means "not mentioned" (unobserved), NOT "no".

**Bayesian module**
```python
build_model(train_df, nodes_config, alpha=1.0) -> model     # Laplace smoothing, alpha=1
posterior(model, evidence: dict[str,str]) -> dict[str,float] # {"ACS":0.6,"GERD":0.2,"Panic":0.1,"Pericarditis":0.1}
```

**Active-query module**
```python
next_step(model, evidence, answered: set, cfg) -> dict
# {"action": "ask"|"stop", "question": node_name|None, "eig": float,
#  "posterior": {...}, "triage": "URGENT"|"SEMI-URGENT"|"ROUTINE"|None, "reason": str}
```

**Mock modules.** The UI developer may use fake versions of the three modules (same signatures, hard-coded return values) in `src/mocks/` until the real ones exist.

## 8. Tasks (do in order; one at a time)

**Task 0 — Repo setup.** Create the layout, venv (Python 3.11), `requirements.txt`, `.gitignore` (raw data, venv), README stub, pytest config. Copy the notebook to `notebooks/`. *Done when:* `pytest` runs (even with zero tests) and the scispaCy model loads in a 3-line script.

**Task 1 — Data module.** Load DDXPlus (train/validate/test), filter to the 4 conditions with the ACS merge, parse evidences (keep values for multi-valued evidences), save processed tables. *Done when:* the printed counts match Section 4 for train, and a test checks that no row has an unknown pathology.

**Task 2 — Candidate nodes (needs a human decision).** Script that prints a table of candidate evidences: question text, P(evidence | condition) for each condition, spread, and flags (`constant`, `leaky`, `rare`). Recommend 12–15 nodes with reasons (askable by a nurse, extractable from text, discriminative, not label-leaking; keep a pain-character node and a radiation node). **Stop and wait for the team to confirm**, then write `config/nodes.json`.

**Task 3 — Bayesian model.** Naive-Bayes structure: `Condition → each node`. Learn the prior and CPTs from the train split with Laplace smoothing (alpha=1). Save the model. Provide `posterior()`. *Tests:* every CPT column sums to 1; no CPT entry is exactly 0 or 1; a classic ACS-looking evidence set ranks ACS first; a GERD-looking set ranks GERD first; contradictory evidence does not crash. Report test-split accuracy and per-class recall (ACS recall matters most) to `results/bn_eval.json`. Discuss the prior choice (empirical prior has ACS = 44.6%) and report both empirical and uniform prior.

**Task 4 — EIG engine and stopping policy.** Implement entropy, expected information gain for every unanswered node using `VariableElimination`, and `next_step()`. Stop when ANY of: max 8 questions; best remaining EIG < 0.02 bits (tunable); top posterior ≥ 0.90. Add a SAFETY rule: triage is based on risk, not only on the top class — if P(ACS) ≥ a threshold (start 0.25, tune on validate, label it an assumption) the output is URGENT even when another condition leads. Never stop on "confidence" alone for a possible-ACS presentation. *Tests:* on the toy network from the notebook it reproduces Exertional 0.407 / Nausea 0.062; answered questions are never asked again; max-question cap works.

**Task 5 — NLP module.** Port the notebook code to `src/nlp/`. Fix bugs A, B, C. Negation per mention within its clause; handle "denies X but reports Y", "no X, Y or Z" (list scope), "does not have X", "non-smoker". Generated synonym variants (pronouns, word order). Align phrases to the nodes in `config/nodes.json`. Keep the scispaCy entities that we could not map in `unmapped`. *Tests:* the three notebook sentences give correct output (including `exertional = no` for "No chest pain on exertion" and `breathlessness = yes` for "can't catch her breath"), plus at least 10 new cases.

**Task 6 — Labelled NLP test set + baseline comparison.** Create `evaluation/nlp_cases.json` with 40–60 sentences and gold labels covering: plain positives, negation, paraphrases, multiple symptoms, list negation, contradictions, symptoms not in our dictionary. Written by at least two team members (not by the agent alone). Compare (a) rule-only baseline (keyword + naive 25-character negation window) vs (b) the full pipeline. Report per-node precision/recall/F1 in `results/nlp_eval.json`. Report honestly if the pipeline does not beat the baseline.

**Task 7 — Session glue.** `pipeline/session.py`: text → `extract_findings` → convert to BN evidence → loop `next_step` → user answer → update → until stop. Works from a command-line demo before any UI exists.

**Task 8 — Streamlit app.** Text box; table of extracted findings that the user can edit/correct; bar chart of the 4 probabilities; the next question with a short reason (its EIG and which conditions it separates); answer buttons (Yes / No / Don't know); probabilities update after every answer; final triage level with the top evidence that moved the result; visible disclaimer: "Educational prototype. Not medical advice." Can be built against mocks first.

**Task 9 — Evaluation of the active questioning.** Simulate patients from the DDXPlus TEST split (answers come from the case's evidence list; absent = "no"). Compare EIG vs random order vs fixed order on: questions needed, final accuracy, ACS recall, entropy reduction. Fixed seeds, results saved to `results/query_eval.json`, plus a plot. *Optional extension (only after Tasks 0–9 work):* add a pretrained biomedical sentence-embedding model to map unseen paraphrases to nodes, and measure dictionary-only vs dictionary+embedding on held-out paraphrases. Verify the model's release date and card before calling it "latest".

**Task 10 — Report and viva notes.** Regenerate the report numbers from `results/`. List limitations (synthetic data, condition mapping, naive-Bayes independence assumption, not clinically validated).

## 9. Team split (parallel work, one Git branch each)

| Member | Tasks | Can start now because |
|---|---|---|
| 1 — NLP | 5, 6 | needs only the node names; use a draft list until Task 2 is confirmed |
| 2 — Data + Bayesian | 1, 2, 3 | works from DDXPlus files only |
| 3 — Active query + evaluation | 4, 9 | develops on the toy network, then switches to the real one |
| 4 — App + integration | 0, 7, 8, 10 | builds the UI and glue against `src/mocks/` |

Everyone must be able to explain every module (viva). After each task, each member reads the `docs/` explanation of the other modules.

## 10. Wording to use about limitations

"DDXPlus is a synthetic differential-diagnosis dataset. Probabilities learned from it are for demonstrating the method and are not clinical estimates. The system is an educational prototype for decision support and must not be used for real patient care."
