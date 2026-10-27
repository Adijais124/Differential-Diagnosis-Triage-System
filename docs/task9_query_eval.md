# Task 9: Evaluation of Active Questioning Policies

## Plain-English Viva Explanation

### What was done in this task?
1. **Simulation Framework on DDXPlus Test Split**:
   - Evaluated active triage questioning by simulating synthetic patient encounters from the official held-out test split (`test_filtered.csv.gz`).
   - Benchmarked across **400 test cases** with a fixed seed (`seed = 42`) for reproducibility on Ryzen 5 CPU.
   - When a policy asks an active question, the synthetic patient simulator answers based on their true underlying evidence profile; unasked questions remain strictly unobserved.

2. **Policies Compared**:
   - **(1) EIG Active Querying**: Dynamically selects the unanswered question maximizing Expected Information Gain at each turn.
   - **(2) Random Order**: Selects an unanswered candidate question uniformly at random.
   - **(3) Static Fixed Order**: Asks candidate questions in a predetermined fixed sequence.

3. **Key Experimental Results (`results/query_eval.json`)**:

| Metric | EIG Active Querying | Random Order | Static Fixed Order |
|---|---|---|---|
| **Avg Questions Needed** | **1.36** | 6.00 | 4.73 |
| **Final Accuracy** | **97.50%** | 95.50% | 97.50% |
| **ACS Recall (Safety)** | **100.00%** | 94.59% *(misses emergencies)* | 100.00% |
| **ACS Safety Escalation Recall** | **100.00%** | 95.68% | 100.00% |
| **Avg Entropy Reduction** | **1.798 bits** | 1.758 bits | 1.797 bits |

4. **Key Viva Takeaways**:
   - **Massive Efficiency Gain**: EIG active questioning resolves patient uncertainty in an average of **1.36 questions**, compared to **6.00 questions** for random ordering (a **77.3% reduction** in required patient questioning).
   - **Patient Safety**: Random questioning frequently exhausts the 8-question cap before uncovering critical risk factors, dropping ACS recall to 94.6% (leaving 5.4% of cardiac emergencies under-triaged). EIG achieves **100.0% ACS recall** because it prioritizes high-discriminative chest pain findings immediately.
   - Interactive plot generated at [`results/query_eval_plot.html`](file:///c:/Users/Aditya/Desktop/2201%20imd%20lab/results/query_eval_plot.html).

### Viva Q&A Cheat Sheet
- **Q: Why does EIG need only 1.36 questions on average?**
  *A: The top discriminative nodes (pain character `E_54` and radiation `E_57`) carry high mutual information with the disease condition. EIG identifies this immediately, asking the highest-leverage question on Turn 1. If that question provides decisive evidence (e.g. heavy crushing pressure radiating to arm), the posterior crosses the confidence threshold and stops without asking irrelevant questions.*
