# Task 6: NLP Benchmarking & Gold Set Evaluation

## Status: `draft gold set, not yet reviewed` (needs_human_review: true)

### What was done in this task?
1. **Curated Gold Evaluation Corpus (`evaluation/nlp_cases.json`)**:
   - Constructed a benchmark set of **50 clinical presentation sentences** reflecting real emergency intake language.
   - Covers:
     - Plain positives (e.g. "severe chest pressure")
     - Clause-level and contrastive negation (e.g. "denies nausea but reports...")
     - Linguistic paraphrases & pronouns (e.g. "can't catch his breath", "winded")
     - Multiple concurrent symptoms in single presentations
     - List negation (e.g. "no chest pain, palpitations or diaphoresis")
     - Contradictions and symptoms outside our dictionary (distractors)
   - Labeled with `"needs_human_review": true` per user instructions so the team can manually verify annotations before viva.

2. **Comparative Benchmark**:
   - **(a) Rule-Only Baseline**: Exact implementation from `Milestone2_.ipynb` (keyword search + naive 25-character backward negation window).
   - **(b) Full Clinical NLP Pipeline**: scispaCy `en_core_sci_md` + `PhraseMatcher` + pronoun variant generation + clause-aware syntactic negation.

3. **Results (`results/nlp_eval.json`)**:
   - **Rule-Only Baseline**:
     - Precision: `0.6358` | Recall: `0.6358` | F1-Score: **`0.6358`**
     - Primary failure mode: Naive 25-character negation scope bled across conjunctions ("denies nausea but reports severe chest pressure" falsely negated chest pressure), and missed pronoun variations ("can't catch her breath").
   - **Full Clinical NLP Pipeline**:
     - Precision: `0.8068` | Recall: `0.7978` | F1-Score: **`0.8022`**
     - **Performance Gain**: **+16.64% F1 improvement** over the baseline.

### Viva Q&A Cheat Sheet
- **Q: Why does the full pipeline outperform the baseline?**
  *A: The baseline uses a crude character distance window for negation which fails on list structures and contrastive conjunctions. The full pipeline uses syntactic clause splitting (stopping at 'but', 'however', and verbs like 'reports') and generates combinatorial pronoun variants for clinical phrasings.*
