# Task 8: Interactive Streamlit Web Application

## Plain-English Viva Explanation

### What was done in this task?
1. **Interactive Clinical Dashboard (`app/streamlit_app.py`)**:
   - Developed a complete web interface using **Streamlit** and **Plotly** to showcase the full end-to-end differential diagnosis pipeline.
   - Built to run locally without external cloud or API dependencies.

2. **Core User Interface Components**:
   - **Mandatory Disclaimer Banner**: Prominently displayed at the top:
     *"Educational prototype. Not medical advice. DDXPlus is a synthetic differential-diagnosis dataset. Probabilities learned from it are for demonstrating the method and are not clinical estimates."*
   - **Clinical Narrative Input & Presets**: Free text narrative box with pre-loaded clinical scenarios representing typical ACS, GERD, Panic, and Pericarditis presentations.
   - **Findings Review & Correction Table**: Displays entities extracted by scispaCy NLP with buttons allowing the user to remove or adjust extracted findings.
   - **Real-Time Probability Bar Chart (Plotly)**: Visualizes the 4 disease probabilities ($P(\text{ACS})$, $P(\text{GERD})$, $P(\text{Panic})$, $P(\text{Pericarditis})$), re-rendering dynamically after each answer.
   - **Active Questioning Card (EIG)**: Displays the highest-information-gain question, its quantitative EIG value (in bits), and the diagnostic rationale for why it was selected.
   - **Answer Buttons**: Dynamic buttons matching the allowed node states, plus a "Don't Know / Skip" button that correctly leaves unasked symptoms unobserved.
   - **Final Triage Output**: Prominent color-coded severity badges (`URGENT` in red, `SEMI-URGENT` in amber, `ROUTINE` in green) with full clinical justification.

3. **How to Launch the App Locally**:
   ```powershell
   .venv\Scripts\streamlit.exe run app/streamlit_app.py
   ```

### Viva Q&A Cheat Sheet
- **Q: How does the UI handle a patient answering "Don't know"?**
  *A: The UI passes the skip signal to `TriageSession`. The symptom is marked as answered so the system will not loop on it, but it is omitted from the Bayesian Network's evidence dictionary, allowing the model to marginalise over it.*
- **Q: Why does the chart use horizontal bars rather than a pie chart?**
  *A: Horizontal bar charts provide an intuitive direct comparison of probabilities on a linear 0–100% scale and allow clear color coding (e.g. emergency red for ACS) to emphasize clinical risk.*
