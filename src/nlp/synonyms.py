"""Synonym dictionary and combinatorial variant generator for clinical findings.

Generates comprehensive linguistic variants for pronouns (my/his/her/their),
prepositions, and word order to fix Bug A, Bug B, and Bug C.
Aligned directly with config/nodes.json.
"""

from itertools import product


def generate_phrase_variants(templates: list[str]) -> list[str]:
    """Expand templated string options like '{pressure|tightness} in {the|his|her|my} chest'."""
    expanded = set()
    for tmpl in templates:
        # Simple expansion of {a|b|c} tokens
        parts = []
        last_idx = 0
        import re
        tokens = list(re.finditer(r"\{([^}]+)\}", tmpl))
        if not tokens:
            expanded.add(tmpl)
            continue

        options_list = []
        static_segments = []
        pos = 0
        for m in tokens:
            static_segments.append(tmpl[pos:m.start()])
            options_list.append([opt.strip() for opt in m.group(1).split("|")])
            pos = m.end()
        static_segments.append(tmpl[pos:])

        for combo in product(*options_list):
            res = ""
            for i, opt in enumerate(combo):
                res += static_segments[i] + opt
            res += static_segments[-1]
            expanded.add(" ".join(res.split()))

    return sorted(expanded)


def build_synonym_dictionary() -> dict[str, list[str]]:
    """Build canonical node -> phrasings dictionary aligned with config/nodes.json."""
    syn_templates = {
        # 1. Multi-valued: Pain character sub-states
        "pain_character_heavy_pressure": [
            "{heavy pressure|pressure|tightness|heaviness|crushing pain|squeezing} {in|across|in the centre of} {the|his|her|my|their} chest",
            "chest {pressure|tightness|heaviness|squeezing}",
            "{squeezing|crushing|heavy|tight|pressing} {chest pain|pain}",
            "tight chest",
            "heavy chest",
            "crushing chest pain",
            "band around {the|his|her|my} chest",
            "weight on {the|his|her|my} chest",
        ],
        "pain_character_burning": [
            "{burning|acid burning|fire-like} {pain|sensation} {in|behind} {the|his|her|my} {chest|breastbone|sternum}",
            "heartburn",
            "burning in {the|his|her|my} chest",
            "acid reflux pain",
        ],
        "pain_character_sharp_knife": [
            "{sharp|stabbing|knife-like|knife stroke|piercing} {pain|chest pain}",
            "sharp pain in {the|his|her|my} chest",
            "catching sharp pain",
            "needle-like chest pain",
        ],
        "pain_character_cramp_other": [
            "cramping chest pain",
            "chest muscle cramp",
            "spasm in the chest",
        ],

        # 2. Multi-valued: Radiation sub-states
        "pain_radiation_arm_shoulder": [
            "{radiating|radiates|shooting|spreads|spreading} to {the|his|her|my|their} {left arm|right arm|arm|shoulder|left shoulder}",
            "pain down {the|his|her|my} {left arm|arm}",
            "pain in {the|his|her|my} left arm",
            "radiation to {the|his|her|my} arm",
        ],
        "pain_radiation_neck_jaw": [
            "{radiating|radiates|shooting|spreads} to {the|his|her|my} {jaw|neck|throat}",
            "pain radiating to {the|his|her|my} jaw",
            "radiation to the neck",
        ],
        "pain_radiation_back_chest_wall": [
            "{radiating|radiates|shooting} to {the|his|her|my} {back|shoulder blades|scapula}",
            "posterior chest wall pain",
            "pain through to {the|his|her|my} back",
        ],

        # 3. Binary symptoms
        "sweating": [
            "sweating",
            "sweaty",
            "diaphoresis",
            "cold sweat",
            "drenched in sweat",
            "profuse sweating",
            "perspiring heavily",
        ],
        "shortness_of_breath": [
            "shortness of breath",
            "short of breath",
            "breathlessness",
            "breathless",
            "dyspnea",
            "difficulty breathing",
            "breathing difficulty",
            "can't catch {my|his|her|their} breath",
            "cannot catch {my|his|her|their} breath",
            "struggling to breathe",
            "winded",
            "out of breath",
        ],
        "racing_heart": [
            "racing heart",
            "heart racing",
            "palpitations",
            "heart palpitations",
            "pounding heart",
            "heart pounding",
            "fluttering in {the|his|her|my} chest",
            "rapid heartbeat",
            "fast heartbeat",
        ],
        "worse_after_eating": [
            "{worse|aggravated|increases|increased} {after|following} {eating|food|meals|dinner|lunch}",
            "worse with food",
            "postprandial pain",
        ],
        "burning_sensation": [
            "burning sensation",
            "burning in {the|his|her|my} throat",
            "acid in {the|his|her|my} throat",
            "sour taste",
            "regurgitation",
            "heartburn sensation",
        ],
        "worse_lying_down": [
            "{worse|aggravated} {when|while} {lying down|flat|supine}",
            "{relieved|better|improves} {when|by} {sitting up|leaning forward}",
            "worse lying flat",
        ],
        "pleuritic_pain": [
            "{pain|chest pain} {worse|increases|sharp} on {deep breath|breathing in|inspiration|coughing}",
            "{worse|increases|increased|sharp} on {deep breath|breathing in|inspiration|taking a breath|coughing}",
            "{worse|increases|increased|sharp} with {deep breath|deep breathing|inspiration|coughing}",
            "hurts to breathe deeply",
            "hurts to breathe",
            "pleuritic chest pain",
            "pleuritic pain",
            "pain with deep inspiration",
        ],
        "recent_viral_infection": [
            "recent {viral infection|cold|flu|respiratory illness|virus|cough and cold}",
            "had a virus recently",
            "recent upper respiratory infection",
        ],
        "prior_heart_attack_angina": [
            "{previous|prior|history of} {heart attack|myocardial infarction|angina|stent|bypass}",
            "known coronary artery disease",
            "had a heart attack",
        ],
        "smoker": [
            "smoker",
            "smokes cigarettes",
            "cigarette smoker",
            "history of smoking",
            "smokes tobacco",
            "pack a day",
        ],
        "high_blood_pressure": [
            "high blood pressure",
            "hypertension",
            "elevated blood pressure",
            "blood pressure medication",
        ],
        "feeling_of_impending_doom": [
            "feeling of impending doom",
            "feels like {he is|she is|i am} dying",
            "feel like {he is|she is|i am} dying",
            "felt like dying",
            "fear of dying",
            "afraid {he|she|i} was about to die",
            "sense of impending doom",
        ],
        "anxiety": [
            "anxiety",
            "anxious",
            "panic",
            "panic attack",
            "feeling panicked",
            "very nervous",
        ],
    }

    dictionary = {}
    for key, tmpls in syn_templates.items():
        dictionary[key] = generate_phrase_variants(tmpls)

    return dictionary
