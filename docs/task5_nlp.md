# Task 5: Clinical NLP Pipeline & Clause-Level Negation

## Plain-English Viva Explanation

### What was done in this task?
1. **Hybrid Architecture (Biomedical Model + Domain Phrase Matching)**:
   - Utilizes AllenAI's pretrained biomedical model (`en_core_sci_md`), which was trained on Medline and PubMed abstracts.
   - However, scispaCy `en_core_sci_md` is a small CNN (`tok2vec`) and tags every medical concept as a generic `ENTITY` without distinguishing symptom types.
   - We built an exact-matching `PhraseMatcher` on top of scispaCy that maps linguistic variants directly to the canonical nodes in `config/nodes.json`.
   - Pretrained entities that do not match our chest-pain node schema are preserved in `unmapped` so no patient information is lost.

2. **Resolution of Known Bugs (A, B, C)**:
   - **Bug A Fix (Pronoun Blindness)**:
     - The notebook missed "can't catch her breath" because its dictionary only contained the first-person phrase "can't catch my breath".
     - In `src/nlp/synonyms.py`, we implemented combinatorial template generation across all pronouns (`{my|his|her|their}`) and determiners (`{the|a}`).
     - Also explicitly handled positive negation idioms: "can't catch breath" contains the word "can't", but semantically affirms positive dyspnea.
   - **Bug B Fix (Clause-Aware Negation & Exertional False Positives)**:
     - The notebook's `exertional` check used an un-negated keyword lookup that wrongly triggered on "No chest pain on exertion".
     - In `src/nlp/negation.py`, negation scope is strictly bounded to its syntactic clause:
       - Stopped at sentence terminators (`.`, `;`, `!`, `?`).
       - Stopped at contrastive conjunctions (`but`, `however`, `although`).
       - Stopped at affirmative transitional verbs (`reports`, `presents with`, `has`).
       - Example: "Patient denies nausea but reports severe chest pressure" correctly negates nausea while keeping chest pressure positive.
   - **Bug C Fix (Overfitted Demo Phrasing)**:
     - The notebook hardcoded a literal sentence from the prompt: "heavy pressure in the centre of his chest".
     - We replaced this with generic variants: `{heavy pressure|pressure|tightness|heaviness|squeezing} {in|across|in the centre of} {the|his|her|my|their} chest`.

3. **Demographic & Temporal Attributes**:
   - Extracted patient age, biological sex, and episode duration in minutes using robust regular expressions.

### Viva Q&A Cheat Sheet
- **Q: Why not use a large LLM like GPT-4 or Gemini for extraction?**
  *A: Per project rules and clinical edge-computing constraints, this decision-support system operates 100% locally with zero cloud dependencies, zero latency, zero API costs, and full patient privacy compliance.*
- **Q: What is the role of scispaCy if the dictionary does the mapping?**
  *A: scispaCy provides medical tokenization, part-of-speech tagging, and identifies clinical entities outside our 14 nodes. These are returned in `unmapped` so clinicians can see peripheral findings (e.g. "headache", "rash") that the Bayesian Network does not model.*
