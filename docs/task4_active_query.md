# Task 4: Expected Information Gain (EIG) & Stopping Policy

## Plain-English Viva Explanation

### What was done in this task?
1. **Expected Information Gain (EIG) Formulation**:
   - Triage intake should not ask random questions. It should ask the single question that will reduce diagnostic uncertainty the most.
   - Uncertainty is quantified as **Shannon Entropy**:
     $$H(C \mid \mathbf{e}) = -\sum_{c \in \text{Conditions}} P(C=c \mid \mathbf{e}) \log_2 P(C=c \mid \mathbf{e})$$
   - Before asking question $Q$, the system computes the expected entropy after receiving its answer:
     $$\text{EIG}(Q \mid \mathbf{e}) = H(C \mid \mathbf{e}) - \sum_{a \in \text{Answers}} P(Q=a \mid \mathbf{e}) H(C \mid \mathbf{e}, Q=a)$$
   - The question with the highest EIG (in bits) is chosen next.

2. **Reproduction Test**:
   - On the 3-condition toy network from `Milestone2_.ipynb` with known evidence `Sweating=yes`:
     - Current entropy: $1.338$ bits
     - $\text{EIG}(\text{Exertional}) = \mathbf{0.407}$ bits
     - $\text{EIG}(\text{Nausea}) = \mathbf{0.062}$ bits
   - Our implementation reproduces these values to three decimal places with both `VariableElimination` and vectorized fast inference.

3. **Explicit Stopping Rules**:
   - Stop if **ANY** of:
     1. **Question Cap**: 8 questions answered (prevents patient fatigue in urgent care).
     2. **Negligible Information**: Best remaining $\text{EIG} < 0.02$ bits (further questions provide negligible entropy reduction).
     3. **Definitive Confidence**: Top posterior $\ge 0.90$ (unless ACS safety escalation applies).

4. **Clinical Safety Rule (Risk vs Argmax)**:
   - In emergency medicine, triage is dictated by **threat to life**, not purely by the most probable diagnosis.
   - If $P(\text{ACS}) \ge 0.25$ (25% threshold), the case is escalated to **`URGENT`** even if GERD or Panic has a higher probability (e.g. 50%).
   - Missing an evolving myocardial infarction has catastrophic consequences.
   - Tested on validation split: threshold 0.25 achieves 100% ACS recall with zero missed cardiac events (documented in `results/threshold_tuning.json`).

### Viva Q&A Cheat Sheet
- **Q: Why use bits (log base 2) for entropy?**
  *A: Base 2 measures uncertainty in bits. An entropy of 2.0 bits over 4 conditions corresponds to complete uniform uncertainty ($\log_2 4 = 2$). An EIG of 0.407 bits means asking this question cuts our uncertainty by nearly half a bit.*
- **Q: Why not just stop when one condition has the highest probability?**
  *A: A patient might have a 40% probability of GERD and a 30% probability of ACS. GERD leads, but the 30% risk of ACS is an immediate medical emergency requiring emergency department referral.*
