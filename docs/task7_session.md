# Task 7: Session Pipeline & Integration Glue

## Plain-English Viva Explanation

### What was done in this task?
1. **Pipeline Glue (`src/pipeline/session.py`)**:
   - Implemented `TriageSession`, the central orchestrator connecting:
     - NLP extraction (`src/nlp/extractor.py`)
     - Bayesian belief updating (`src/bayesian/inference.py`)
     - Active questioning and clinical escalation (`src/active_query/policy.py`)
   - Manages state across conversational turns: maintains `initial_findings`, `evidence`, `answered` questions set, and posterior probability trajectory.

2. **Observation Semantics in Live Sessions**:
   - When a patient describes their complaint (e.g. "Chest tightness and sweating"), only the symptoms extracted from the text are entered into `evidence`.
   - All other 12 Bayesian nodes remain **UNOBSERVED** (marginalised out).
   - When the user answers an active question:
     - Providing "yes" or "no" (or a specific multi-valued state) adds that symptom to `evidence`.
     - Providing "don't know" or "skip" marks the question as `answered` (so it will never be asked again) but **leaves it unobserved in `evidence`**.

3. **Command-Line Interface (CLI)**:
   - Provides a standalone terminal demonstration (`python -m src.pipeline.session`) enabling thorough pair testing of the questioning loop before touching UI code.

### Viva Q&A Cheat Sheet
- **Q: Why is skipping a question handled differently from answering "no"?**
  *A: In Bayesian Networks, an unobserved variable is marginalised: $\sum_x P(C, X=x)$. Answering "no" conditions on $X=\text{no}$. If a patient does not know whether they have high blood pressure, forcing "no" biases the model; leaving it unobserved preserves truthful uncertainty.*
