"""Streamlit Web Application: Differential Diagnosis Triage System.

Educational prototype demonstrating active clinical questioning via EIG,
Bayesian belief updating, and clinical safety escalation.
"""

import os
import sys
from pathlib import Path
import pickle

# Ensure project root is in sys.path so 'src' can be imported regardless of execution CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go

from src.active_query.policy import next_step, DEFAULT_POLICY_CONFIG
from src.bayesian.build_model import load_nodes_config
from src.bayesian.inference import posterior
from src.nlp.extractor import extract_findings

st.set_page_config(
    page_title="Differential Diagnosis Triage System",
    page_icon="🩺",
    layout="wide",
)

# Mandatory Clinical Limitation & Educational Disclaimer (Section 10)
st.warning(
    "**EDUCATIONAL PROTOTYPE — NOT MEDICAL ADVICE**  \n"
    "DDXPlus is a synthetic differential-diagnosis dataset. Probabilities learned from it are for "
    "demonstrating probabilistic AI methods and are not clinical estimates. This system is an educational "
    "prototype for decision support and must NOT be used for real patient care."
)


@st.cache_resource
def load_resources():
    """Load Bayesian model and node configuration."""
    model_path = os.path.join(PROJECT_ROOT, "results", "bn_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at {model_path}. Please train the model first.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    cfg = load_nodes_config(str(PROJECT_ROOT / "config" / "nodes.json"))
    return model, cfg


model, nodes_cfg = load_resources()
nodes_list = nodes_cfg.get("nodes", [])
node_dict = {n["name"]: n for n in nodes_list}

# Initialize session state
if "session_active" not in st.session_state:
    st.session_state.session_active = False
    st.session_state.evidence = {}
    st.session_state.answered = set()
    st.session_state.history = []
    st.session_state.current_text = ""
    st.session_state.attributes = {}
    st.session_state.unmapped = []
    st.session_state.final_result = None

# Sidebar: Preset clinical scenarios
st.sidebar.title("Sample Clinical Presets")
presets = {
    "Select a preset...": "",
    "Scenario 1: Classic ACS (Urgent)": (
        "A 55-year-old male smoker reports heavy pressure in his chest for 20 minutes, "
        "radiating to left arm with sweating. Denies nausea."
    ),
    "Scenario 2: GERD (Postprandial Heartburn)": (
        "42-year-old woman complains of burning chest pain and acid taste in throat, "
        "significantly worse after eating dinner. Denies shortness of breath."
    ),
    "Scenario 3: Panic Attack (Young Adult)": (
        "She can't catch her breath, has a racing heart and feels anxious. "
        "Feels like she is about to die. No chest pain on exertion."
    ),
    "Scenario 4: Pericarditis (Pleuritic & Positional)": (
        "30-year-old male with sharp knife-like chest pain that is worse when taking a deep breath, "
        "relieved by leaning forward. Had a viral illness 1 week ago."
    ),
}

selected_preset = st.sidebar.selectbox("Load sample case:", list(presets.keys()))

st.title("🩺 Chest-Pain Differential Diagnosis & Active Triage")
st.markdown(
    "*NLP Extraction (scispaCy) $\\rightarrow$ Bayesian Network (`pgmpy`) $\\rightarrow$ "
    "Active Questioning (Expected Information Gain)*"
)

# Patient narrative input
default_text = presets.get(selected_preset, "") if selected_preset != "Select a preset..." else st.session_state.current_text

user_narrative = st.text_area(
    "**Enter patient chief complaint / clinical triage note:**",
    value=default_text,
    height=100,
    placeholder="e.g. 58-year-old male with crushing chest pressure radiating to arm and cold sweat...",
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🚀 Analyze & Start Triage", type="primary"):
        if not user_narrative.strip():
            st.error("Please enter a clinical narrative first.")
        else:
            with st.spinner("Extracting clinical findings with scispaCy biomedical NLP..."):
                nlp_out = extract_findings(user_narrative)
                st.session_state.session_active = True
                st.session_state.current_text = user_narrative
                st.session_state.attributes = nlp_out["attributes"]
                st.session_state.unmapped = nlp_out["unmapped"]
                st.session_state.evidence = dict(nlp_out["findings"])
                st.session_state.answered = set(st.session_state.evidence.keys())
                st.session_state.history = []
                st.session_state.final_result = None

with col_btn2:
    if st.session_state.session_active and st.button("🔄 Reset Session"):
        st.session_state.session_active = False
        st.session_state.evidence = {}
        st.session_state.answered = set()
        st.session_state.history = []
        st.session_state.final_result = None
        st.rerun()

# If session is active
if st.session_state.session_active:
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    # Left Column: Extracted findings and demographics
    with col_left:
        st.subheader("1. Extracted Patient Findings")
        if st.session_state.attributes:
            demo_str = " | ".join(f"**{k.capitalize()}**: {v}" for k, v in st.session_state.attributes.items())
            st.info(f"📋 **Patient Profile**: {demo_str}")

        # Findings table editor
        st.markdown("**Findings derived from text (unmentioned symptoms are unobserved):**")
        if st.session_state.evidence:
            for node_name, val in list(st.session_state.evidence.items()):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"• `{node_name}`: **{val}**")
                if c2.button("Remove", key=f"del_{node_name}"):
                    del st.session_state.evidence[node_name]
                    st.rerun()
        else:
            st.write("*(No specific candidate symptoms identified in text yet)*")

        if st.session_state.unmapped:
            st.caption(f"Unmapped scispaCy medical entities: {', '.join(st.session_state.unmapped)}")

    # Right Column: Bayesian Belief State (Bar Chart)
    current_post = posterior(model, st.session_state.evidence, use_fast_bayes=True)
    with col_right:
        st.subheader("2. Diagnostic Belief Distribution")
        colors = {
            "ACS": "#D32F2F",       # Emergency red
            "GERD": "#00897B",      # Teal
            "Panic": "#7B1FA2",     # Purple
            "Pericarditis": "#F57C00" # Orange
        }
        fig = go.Figure(go.Bar(
            x=[f"{p * 100:.1f}%" for p in current_post.values()],
            y=list(current_post.keys()),
            orientation="h",
            marker=dict(color=[colors.get(c, "#1976D2") for c in current_post.keys()]),
            text=[f"{p * 100:.1f}%" for p in current_post.values()],
            textposition="auto",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 100], title="Probability (%)"),
            yaxis=dict(autorange="reversed"),
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. Active Questioning / Stopping Policy Section
    st.markdown("---")
    st.subheader("3. Active Questioning & Triage Policy")

    # Evaluate next policy step
    step = next_step(
        model,
        evidence=st.session_state.evidence,
        answered=st.session_state.answered,
        cfg=DEFAULT_POLICY_CONFIG,
    )

    if step["action"] == "stop":
        st.session_state.final_result = step
        triage_level = step["triage"]
        triage_color = {
            "URGENT": "error",
            "SEMI-URGENT": "warning",
            "ROUTINE": "success",
        }.get(triage_level, "info")

        if triage_level == "URGENT":
            st.error(f"### 🚨 TRIAGE LEVEL: {triage_level}")
        elif triage_level == "SEMI-URGENT":
            st.warning(f"### ⚠️ TRIAGE LEVEL: {triage_level}")
        else:
            st.success(f"### 🟢 TRIAGE LEVEL: {triage_level}")

        st.markdown(f"**Clinical Disposition Rationale:**  \n{step['reason']}")

        # Show final probabilities table
        st.markdown("**Final Calibrated Likelihoods:**")
        cols = st.columns(4)
        for i, (cond, prob) in enumerate(step["posterior"].items()):
            cols[i].metric(cond, f"{prob * 100:.2f}%")

    else:
        # Active Question Prompt Card
        q_name = step["question"]
        q_info = node_dict.get(q_name, {})
        q_text = q_info.get("question_text", f"Do you experience {q_name}?")
        q_states = q_info.get("states", ["no", "yes"])
        eig_val = step["eig"]

        st.info(
            f"**Recommended Question (EIG = {eig_val:.3f} bits):**  \n"
            f"### {q_text}"
        )
        st.caption(f"Reason: Node `{q_name}` provides maximum expected entropy reduction between remaining candidate pathologies.")

        # Interactive Answer Buttons
        st.write("**Select patient answer:**")
        ans_cols = st.columns(len(q_states) + 1)
        for i, st_val in enumerate(q_states):
            label = st_val.replace("_", " ").capitalize()
            if ans_cols[i].button(label, key=f"btn_{q_name}_{st_val}"):
                st.session_state.answered.add(q_name)
                st.session_state.evidence[q_name] = st_val
                st.session_state.history.append((q_name, st_val))
                st.rerun()

        # Skip / Don't Know button
        if ans_cols[-1].button("Don't Know / Skip", key=f"btn_{q_name}_skip"):
            st.session_state.answered.add(q_name)
            # Stays unobserved in evidence
            st.session_state.history.append((q_name, "skipped"))
            st.rerun()

        st.write(f"*(Turn {len(st.session_state.history) + 1} of max {DEFAULT_POLICY_CONFIG['max_questions']} questions)*")
