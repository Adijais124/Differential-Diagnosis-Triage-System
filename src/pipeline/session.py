"""End-to-end interactive session glue tying NLP, Bayesian Network, and Active Questioning.

Flow:
1. Patient narrative in plain text -> extract_findings()
2. Initial findings converted to Bayesian evidence (unmentioned symptoms remain unobserved)
3. next_step() selects highest-EIG question
4. User/Simulator provides answer -> evidence updated
5. Repeats until stopping policy triggers -> final triage output with explanation
"""

import os
import pickle
from pgmpy.models import BayesianNetwork
from src.active_query.policy import next_step, DEFAULT_POLICY_CONFIG
from src.bayesian.build_model import load_nodes_config
from src.bayesian.inference import posterior
from src.nlp.extractor import extract_findings


class TriageSession:
    """Manages the interactive diagnostic questioning and triage session."""

    def __init__(
        self,
        model: BayesianNetwork | None = None,
        nodes_config: dict | None = None,
        policy_config: dict | None = None,
    ):
        if model is None:
            model_path = os.path.join("results", "bn_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found. Run Task 3 first.")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = model

        self.nodes_config = nodes_config or load_nodes_config()
        self.policy_config = policy_config or dict(DEFAULT_POLICY_CONFIG)

        # Mapping node_name -> node_dict
        self.node_dict = {n["name"]: n for n in self.nodes_config.get("nodes", [])}

        # Session state
        self.initial_text = ""
        self.attributes = {}
        self.unmapped = []
        self.evidence = {}
        self.answered = set()
        self.history = []
        self.is_finished = False
        self.final_result = None

    def start_from_text(self, text: str) -> dict:
        """Initialize session from clinical free text narrative.

        Extracts structured findings and initializes evidence.
        """
        self.initial_text = text
        nlp_out = extract_findings(text)
        self.attributes = nlp_out["attributes"]
        self.unmapped = nlp_out["unmapped"]

        # Only nodes found by NLP are added to evidence; all others remain unobserved!
        self.evidence = dict(nlp_out["findings"])
        self.answered = set(self.evidence.keys())
        self.history = []
        self.is_finished = False
        self.final_result = None

        current_post = posterior(self.model, self.evidence, use_fast_bayes=True)
        return {
            "findings": self.evidence,
            "attributes": self.attributes,
            "unmapped": self.unmapped,
            "posterior": current_post,
        }

    def get_current_step(self) -> dict:
        """Query policy for next action (ask question or stop)."""
        if self.is_finished and self.final_result:
            return self.final_result

        step = next_step(
            self.model,
            evidence=self.evidence,
            answered=self.answered,
            cfg=self.policy_config,
        )

        if step["action"] == "stop":
            self.is_finished = True
            self.final_result = step
        else:
            # Attach question text and options for display
            q_node = self.node_dict.get(step["question"])
            if q_node:
                step["question_text"] = q_node.get("question_text", f"Do you have {step['question']}?")
                step["states"] = q_node.get("states", ["no", "yes"])

        return step

    def submit_answer(self, question: str, answer: str) -> dict:
        """Incorporate answer to active question and update state.

        If answer is 'unknown' or 'skip', symptom is treated as unobserved.
        """
        self.answered.add(question)
        ans_clean = answer.strip().lower()

        if ans_clean not in ["unknown", "skip", "don't know", "none"]:
            self.evidence[question] = ans_clean

        updated_post = posterior(self.model, self.evidence, use_fast_bayes=True)
        self.history.append({
            "question": question,
            "answer": answer,
            "posterior_after": updated_post,
        })

        return self.get_current_step()


def run_cli_demo():
    """Interactive command-line demonstration of the complete triage pipeline."""
    print("=" * 70)
    print("Differential Diagnosis Triage System — CLI Interactive Demo")
    print("=" * 70)

    demo_text = (
        "A 55-year-old male smoker reports heavy pressure in his chest for 20 minutes, "
        "radiating to left arm with sweating. Denies nausea."
    )
    print("\nDefault demo sentence:")
    print(f'"{demo_text}"')
    choice = input("\nPress [Enter] to use default demo, or type your own text: ").strip()
    input_text = choice if choice else demo_text

    session = TriageSession()
    init_res = session.start_from_text(input_text)

    print("\n--- 1. Extracted Findings ---")
    print("Findings:", init_res["findings"])
    print("Attributes:", init_res["attributes"])
    if init_res["unmapped"]:
        print("Unmapped scispaCy entities:", init_res["unmapped"])
    print("\nInitial Posterior Probabilities:")
    for c, p in init_res["posterior"].items():
        print(f"  {c:<14}: {p * 100:.1f}%")

    print("\n--- 2. Active Questioning Loop (EIG) ---")
    turn = 1
    while not session.is_finished:
        step = session.get_current_step()
        if step["action"] == "stop":
            break

        q_name = step["question"]
        q_text = step.get("question_text", f"Question about {q_name}")
        states = step.get("states", ["no", "yes"])
        eig = step["eig"]

        print(f"\n[Turn {turn}] Question: {q_text}")
        print(f"  (EIG: {eig:.3f} bits | Options: {', '.join(states)})")

        ans = input(f"Answer for {q_name} [default: {states[0]}]: ").strip().lower()
        if not ans:
            ans = states[0]

        step_res = session.submit_answer(q_name, ans)
        print("Updated Posterior:")
        for c, p in step_res["posterior"].items():
            print(f"  {c:<14}: {p * 100:.1f}%")
        turn += 1

    final = session.final_result
    print("\n" + "=" * 70)
    print("FINAL TRIAGE ASSESSMENT")
    print("=" * 70)
    print(f"Triage Level : {final['triage']}")
    print(f"Reason       : {final['reason']}")
    print("Final Class Probabilities:")
    for c, p in final["posterior"].items():
        print(f"  {c:<14}: {p * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_cli_demo()
