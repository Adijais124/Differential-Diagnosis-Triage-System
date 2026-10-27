# Task 0: Repository Setup & Architecture

## Plain-English Viva Explanation

### What was done in this task?
1. **Established the Project Layout**:
   - Structured folders according to clean separation of concerns:
     - `src/data/`: Data ingestion and preprocessing.
     - `src/bayesian/`: Probabilistic modeling with Bayesian Networks.
     - `src/active_query/`: Expected Information Gain (EIG) calculation and sequential questioning.
     - `src/nlp/`: Natural language processing and clause-level negation handling.
     - `src/pipeline/`: Integration glue tying NLP, BN, and questioning together.
     - `app/`: Interactive Streamlit user interface.
     - `tests/`: Automated unit tests using `pytest`.
     - `docs/`: Module-by-module plain-English explanations for viva preparation.
     - `results/`: Storage for experiment metrics, ensuring zero invented numbers.

2. **Isolated Environment & Dependency Management**:
   - Pinned Python version to 3.11. This is critical because `spaCy 3.7.5` and biomedical models (`scispaCy 0.5.4` and `en_core_sci_md`) have pre-compiled binary compatibility with Python 3.11, avoiding C++ build issues that occur on Python 3.12+.
   - Pinned `pgmpy==0.1.26` where the primary class is `BayesianNetwork` (renamed in later releases).
   - Created a `.gitignore` to prevent committing massive raw dataset files (DDXPlus) and virtual environments.
   - Configured `pytest.ini` for immediate and consistent test execution.

### Viva Q&A Cheat Sheet
- **Q: Why are we using a virtual environment and Python 3.11 specifically?**
  *A: Specialized scientific and NLP packages like `scispaCy 0.5.4` and `pgmpy 0.1.26` depend on specific compiled C-extensions (like specific Cython/Thinc builds in spaCy). Python 3.11 provides stable, prebuilt binary wheels without requiring a local C++ compilation toolchain.*
- **Q: Why are raw datasets excluded from Git?**
  *A: The DDXPlus dataset contains over 1 million synthetic patient encounters across gigabytes of data. Large files bloat the Git history and violate repository limits; instead, reproducibility scripts download and process raw data locally into `data/`.*
