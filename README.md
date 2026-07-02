# Offline Candidate Ranker — Team Kirmada.exe

Welcome to the **Offline Candidate Ranker**, a highly scalable, explainable AI pipeline designed to identify the top 100 best-fit candidates from a massive talent pool. This repository is our official submission for the Redrob Hackathon v4.

---

## 🎥 Demonstration Video
> 
https://github.com/user-attachments/assets/1c40084a-e4e5-46e0-801a-832bae1aad04





## 🌐 Live Sandbox (Streamlit)
> **[Experience the Offline Candidate Ranker Live](https://offline-candidate-ranker-f8fn73ffz2kdeu9fnwz8kj.streamlit.app/)**

---

## 🏗️ System Architecture & Diagram
> <img width="1501" height="844" alt="System Architecture" src="https://github.com/user-attachments/assets/06ccc961-1805-4de2-81f8-fc7b57d94e07" />


Our ranking system uses a **seven-stage explainable AI pipeline** to ensure accurate, fair, and scalable candidate ranking. By heavily relying on offline pre-computation, we bypass the need for external LLM APIs during runtime, ensuring our pipeline easily processes 100,000+ candidates locally on CPU within the strict 5-minute time limit.

### Stage 1: JD Parsing & Normalization
Raw job descriptions (`.docx` / `.pdf`) are rarely clean or structured. The pipeline first extracts and cleans the text, detects its underlying structure, and normalizes it into a standard format — turning inconsistent, free-form JDs into a machine-readable schema the rest of the pipeline can trust.

### Stage 2: Requirement Extraction
From the normalized JD, we extract the discrete signals that actually drive matching: required skills, role & experience level, location, education, and other constraints (salary band, notice period, etc.). This converts a wall of text into structured requirement fields used downstream by the scoring engine.

### Stage 3: Offline Precomputation Pipeline
This is where the heavy lifting happens **before** the 5-minute clock starts, using static resources (`seniority_map.json`, `skill_aliases.json`, `company_founding_years.json`, `consulting_firms.json`) alongside the candidate pool:
1. **Honeypot Detector** (`01_honeypot_detector.py`) — flags fraudulent or logically inconsistent profiles early.
2. **Feature Extractor** (`02_feature_extractor.py`) — parses candidates into structured, scorable features.
3. **SentenceTransformer Embeddings** (`03_embedder.py`) — generates semantic embeddings locally, with no external API calls.
4. **BM25 Index Builder** (`04_build_bm25.py`) — builds a high-speed keyword-matching index.

All outputs are cached to disk as reusable artifacts (`honeypot_flags.parquet`, `features.parquet`, `embeddings.npy`, `bm25_index.pkl`), so the live ranking step never re-computes them.

### Stage 4: Matching & Ranking Engine (`rank.py`)
With requirements extracted and artifacts precomputed, the live ranking step retrieves and scores candidates using a hybrid, multi-factor approach:
- **Semantic Retrieval** (embedding similarity) + **BM25 Retrieval** (keyword matching) — a hybrid search that catches both conceptual and exact-term matches.
- **Feature Engineering** — derives signals like experience, stability, and education from the precomputed features.
- **Heuristic Scoring** — combines everything into a weighted, multi-factor `composite_score`.
- **Ranking & Tie-Breaking** — applies deterministic business rules to resolve close calls.
- **Reasoning Generator** — produces the explainable, human-readable justification for each ranked candidate.
- **Validator** — enforces the top-100 cutoff, correct output format, and final sanity checks before anything is written out.

### Stage 5: Output Generation
The validated results are written to `submission.csv` — the top 100 ranked candidates, each with their composite score and generated reasoning.

### Stage 6: Streamlit Dashboard (`sandbox/app.py`)
Recruiters can interactively explore results without touching the underlying data: ranking overview, per-candidate details, reasoning viewer, honeypot filter, feature comparison, and exportable reports.

### Stage 7: Recruiter Outcomes
The end result: faster screening, explainable rankings, and data-driven shortlisting — leading to reduced hiring bias, higher-quality hires, and audit-ready reports..

---

## 📁 Repository Structure
```text
Offline-candidate-ranker/
├── artifacts/             # Pre-computed outputs (FAISS indexes, BM25 pickles, embeddings)
├── data/                  # Input data folder (candidates.jsonl, job_description.docx)
├── models/                # Locally cached model weights (e.g., sentence-transformers)
├── precompute/            # Offline phase scripts
│   ├── 01_honeypot_detector.py
│   ├── 02_feature_extractor.py
│   ├── 03_embed_and_index.py
│   ├── 04_build_bm25.py
│   └── run_all.py         # Orchestrates the pre-computation pipeline
├── scripts/               # Utilities (normalization, sample data generation)
├── src/                   # Core Python modules
│   ├── jd_parser.py       # Job description extraction
│   ├── reasoning.py       # Explainable output generator
│   ├── scorer.py          # Multi-factor weighting logic
│   └── validator.py       # Strict output validation enforcing Hackathon rules
├── static/                # Static lookups (company_founding_years.json)
├── tests/                 # Comprehensive PyTest suite
├── rank.py                # Main ranking script (The 5-minute bounded step)
├── run_all.ps1            # 1-click execution script for the entire workflow
├── validate_output.py     # Final verification script for submission.csv
└── README.md              # You are here!
```

---

## 🚀 How to Run the Pipeline

Our code is heavily optimized to run locally on a 16GB CPU machine, completing the final ranking phase comfortably within the 5-minute limit.

### Setup & Requirements
Before running the pipeline, ensure you have Python 3.12+ installed. 

Create and activate your virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:
```powershell
pip install -r requirements.txt
```

### Option 1: The Automated Way (Recommended)
We have included a convenient PowerShell script that manages the entire pipeline end-to-end. It will automatically handle the data normalization, pre-computation, ranking, validation, and testing.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -CandidatesFile ".\data\candidates.jsonl" -JdFile ".\data\job_description.docx"
```
*(If the `CandidatesFile` is not found, the script will instantly fail safely rather than overwriting your data.)*

### Option 2: Running Individual Commands Step-by-Step
If you prefer to see exactly what is happening under the hood, you can run the pipeline sequentially:

**1. Data Normalization**
Flattens and normalizes the deeply nested raw candidate JSON file so our scripts can process it cleanly.
```bash
python scripts/normalize_candidates_for_local.py --input ./data/candidates.jsonl --output ./data/candidates.normalized.jsonl
```

**2. Pre-computing Artifacts (Offline Phase)**
This step extracts features, generates embeddings, and builds the FAISS/BM25 indexes. 
*(Note: This offline step may take ~27 minutes on a full 100k dataset).*
```bash
python precompute/run_all.py --candidates ./data/candidates.normalized.jsonl --jd ./data/job_description.docx
```

**3. Running the Ranker (5-Minute bounded step)**
This is the main ranking step that executes the multi-factor scoring engine on the pre-computed artifacts, outputting the top 100 candidates into `submission.csv`.
```bash
python rank.py --candidates ./data/candidates.normalized.jsonl --jd ./data/job_description.docx --out ./submission.csv
```

**4. Validating the Output**
Ensures the generated CSV perfectly matches the strict hackathon formatting rules (exactly 100 rows, monotonically decreasing scores, valid IDs).
```bash
python validate_output.py --submission ./submission.csv --candidates ./data/candidates.normalized.jsonl
```

**5. Final Hackathon Submission Validation**
Runs the final standalone script to double-check that your output matches the final server-side submission criteria.
```bash
python validate_submission.py ./submission.csv
```

---

### AI Usage Declaration
We utilized AI coding assistants strictly for accelerating local development, standardizing data schemas, and automating repetitive coding tasks. 
**No candidate data is sent to external LLM APIs.** All embeddings, scoring algorithms, and ranking inference take place 100% locally on CPU to strictly adhere to the compute limits and data privacy requirements of the competition. 
