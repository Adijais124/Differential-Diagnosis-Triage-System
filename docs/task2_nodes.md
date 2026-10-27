# Task 2: Candidate Bayesian Nodes Selection

## Selection Status: `PROVISIONAL` (Ready for team review)

All subsequent modules (Bayesian modeling, active querying, NLP extraction) are configuration-driven by [`config/nodes.json`](file:///c:/Users/Aditya/Desktop/2201%20imd%20lab/config/nodes.json). The team can review and edit this list at any time and re-run downstream scripts cleanly.

---

### 1. Selection Criteria
Per the brief instructions, candidates were evaluated on:
1. **Clinical Askability**: Questions a triage nurse or interactive intake system can easily ask a patient.
2. **Text Extractability**: Symptoms with clear NLP lexicons and phrase synonyms.
3. **Condition Discrimination (Spread)**: Large separation across the 4 conditions ($P(E|C)$).
4. **Non-Leakage**: Avoiding codes that give away the diagnosis immediately (e.g. `E_3` "Have you ever had pericarditis?").
5. **Pain Specificity**: Preserving multi-valued pain character (`E_54`) and pain radiation (`E_57`) values.

---

### 2. The 14 Recommended Nodes

| Node Name | DDXPlus Code | Type & States | Clinical Rationale & Separation |
|---|---|---|---|
| `pain_character` | `E_54` | Multi-valued: `heavy_pressure`, `burning`, `sharp_knife`, `cramp_other`, `none` | **ACS vs GERD vs Pericarditis vs Panic**: In DDXPlus, `heavy_pressure` is #1 for ACS (42,756 cases), `burning` is #1 for GERD (19,381 cases), `sharp/knife-stroke` is #1 for Pericarditis (18,442 cases), and `cramp` is #1 for Panic (19,718 cases). |
| `pain_radiation` | `E_57` | Multi-valued: `nowhere`, `arm_shoulder`, `back_chest_wall`, `neck_jaw`, `other` | **ACS vs Panic**: Panic radiates `nowhere` in 100% of cases (`V_123`). ACS strongly radiates to `biceps/arm/shoulder` (42,000+ cases). Pericarditis radiates to `posterior chest wall/back` (17,000+ cases). |
| `sweating` | `E_50` | Binary (`no`, `yes`) | **Diaphoresis**: Present in Panic (60.7%) and ACS (41.7%), completely absent (0.0%) in GERD and Pericarditis. |
| `shortness_of_breath` | `E_66` | Binary (`no`, `yes`) | **Dyspnea**: High in Panic (70.6%), ACS (63.8%), Pericarditis (63.0%), but 0.0% in uncomplicated GERD. |
| `racing_heart` | `E_155` | Binary (`no`, `yes`) | **Palpitations/Tachycardia**: Panic (72.4%) and Pericarditis (56.0%), vs 0.0% in GERD and ACS. |
| `worse_after_eating` | `E_215` | Binary (`no`, `yes`) | **Postprandial**: 74.5% in GERD, 0.0% in ACS, Panic, and Pericarditis. |
| `burning_sensation` | `E_173` | Binary (`no`, `yes`) | **Pyrosis (Heartburn)**: 74.9% in GERD, 0.0% in others. |
| `worse_lying_down` | `E_217` | Binary (`no`, `yes`) | **Positional Discomfort**: Pericarditis (71.1%) and GERD (64.5%), 0.0% in ACS and Panic. |
| `pleuritic_pain` | `E_220` | Binary (`no`, `yes`) | **Pleuritic Pain**: Increases with deep breath/cough; 69.5% in Pericarditis, 0.0% in others. |
| `recent_viral_infection` | `E_0` | Binary (`no`, `yes`) | **Pericarditis Etiology**: Viral prodrome; 75.0% in Pericarditis, 0.0% in others. |
| `prior_heart_attack_angina` | `E_105` | Binary (`no`, `yes`) | **Known CAD / Prior MI**: 78.3% in ACS, 0.0% in others. |
| `smoker` | `E_79` | Binary (`no`, `yes`) | **Cardiovascular Risk Factor**: 72.4% in ACS, 66.8% in GERD, 0.0% in Panic/Pericarditis. |
| `high_blood_pressure` | `E_104` | Binary (`no`, `yes`) | **Hypertension**: 69.4% in ACS, 0.0% in others. |
| `feeling_of_impending_doom` | `E_111` | Binary (`no`, `yes`) | **Sense of Dying / Panic**: 76.8% in Panic, 0.0% in others. |

---

### 3. Excluded Candidate Codes
- **`E_3` ("Have you ever had pericarditis?")**: Spread = 0.8374. Excluded because it almost completely leaks the diagnosis label.
- **`E_185` ("Have you ever had a head trauma?")**: Spread = 0.7136. Clinically unrelated to chest-pain emergency triage; would confuse clinicians in an active questioning demo.
- **`E_157` ("Have you recently had numbness / loss of sensation?")**: Spread = 0.8045. Peripheral neurological symptom, less relevant for acute chest pain triage presentation.
