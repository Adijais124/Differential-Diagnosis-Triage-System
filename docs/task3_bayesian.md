# Task 3: Bayesian Network Modeling & Posterior Inference

## Plain-English Viva Explanation

### What was done in this task?
1. **Model Architecture**:
   - Built a Naive Bayes network using `pgmpy 0.1.26` where the target condition node (`Condition`) is the parent pointing to each of the 14 candidate clinical finding nodes:
     $$\text{Condition} \rightarrow X_i \quad \text{for each symptom } X_i$$
   - Under this conditional independence topology, symptoms are conditionally independent given the true underlying pathology.

2. **Laplace Smoothing ($\alpha = 1.0$)**:
   - Synthetic medical datasets like DDXPlus contain extreme distributions where certain evidence counts are 0 outside a condition (e.g. prior heart attack `E_105` is 0 outside ACS).
   - Without smoothing, a single non-matching symptom yields a zero probability ($P = 0.00$), causing total mathematical failure when evaluating real patient variations.
   - Laplace smoothing:
     $$P(X=s_k \mid C=c_j) = \frac{N(X=s_k, C=c_j) + 1}{N(C=c_j) + K}$$
     guarantees that:
     1. Every column of every CPT sums exactly to $1.00$.
     2. No probability is ever $0.00$ or $1.00$.
     3. Contradictory symptoms produce a valid posterior instead of crashing.

3. **Prior Selection Comparison**:
   - **Empirical Prior**: Derived directly from train data proportions:
     - ACS: 44.64% (59,499 / 133,282)
     - GERD: 19.49% (25,979 / 133,282)
     - Panic: 18.77% (25,019 / 133,282)
     - Pericarditis: 17.10% (22,785 / 133,282)
   - **Uniform Prior**: Equal initial prior of $25\%$ for all 4 conditions.
   - Test evaluation results: Both achieve 99.98% overall accuracy and 100% ACS recall on the test set of 18,202 cases (saved in `results/bn_eval.json`).

4. **Inference & Observation Semantics**:
   - In live testing and application use, symptoms that have not been asked or mentioned are **omitted from evidence**, allowing the Bayesian Network to marginalise over them rather than falsely assuming they are negative.

### Viva Q&A Cheat Sheet
- **Q: Why does the Naive Bayes assumption work well here despite symptom correlations?**
  *A: While physiological symptoms can correlate, Naive Bayes acts as a strong, robust linear separator in log-odds space. Even when exact probability calibration is distorted by correlation, the relative ranking of conditions remains exceptionally stable.*
- **Q: Why does accuracy look unrealistically high (99.98%)?**
  *A: DDXPlus is a rule-based synthetic generator created by an automated clinical simulator. Its symptom profiles are much more cleanly separated than real-world emergency room presentations.*
