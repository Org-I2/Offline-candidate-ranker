# Offline Candidate Ranker — Team Kirmada.exe

Welcome to the **Offline Candidate Ranker**, a highly scalable, explainable AI pipeline designed to identify the top 100 best-fit candidates from a massive talent pool. This repository is our official submission for the Redrob Hackathon v4.

---

## 🎥 Demonstration Video
> **[Insert Video Demonstration Link Here]**

## 🌐 Live Sandbox (Streamlit)
> **[Experience the Offline Candidate Ranker Live](https://offline-candidate-ranker-f8fn73ffz2kdeu9fnwz8kj.streamlit.app/)**

---

## 🏗️ System Architecture & Diagram
> **[Insert Architecture Diagram Image Here]**

Our ranking system uses a **three-stage explainable AI pipeline** to ensure accurate, fair, and scalable candidate ranking. By heavily relying on offline pre-computation, we bypass the need for external LLM APIs during runtime, ensuring our pipeline easily processes 100,000+ candidates locally on CPU within the strict 5-minute time limit.

### Stage 1: Data Normalization & Offline Pre-computation
When dealing with massive candidate pools, raw data can be deeply nested and inconsistent. 
1. **Normalization**: The pipeline first flattens the unstructured `candidates.jsonl` into a standard, aliased format using `scripts/normalize_candidates_for_local.py`.
2. **Artifact Generation**: We run the pre-computation suite (`precompute/run_all.py`), which generates semantic embeddings using a locally-hosted MiniLM model, constructs a high-speed FAISS vector index, and builds a BM25 keyword-matching index. This step is run entirely offline.

### Stage 2: Multi-Factor Scoring Engine (Ranking)
During the live 5-minute ranking step (`rank.py`), the shortlisted candidates retrieved by our Hybrid Engine (BM25 + Semantic Vector Search) are evaluated using a dynamic scoring engine. 

The engine calculates a `composite_score` by evaluating multiple, weighted real-world signals:
- **Skills Alignment (25%)**: Semantic overlap between candidate skills and JD requirements.
- **Experience Relevance (20%)**: Length and applicability of past work history.
- **Project Quality (15%)**: Signal values parsed from candidate achievements.
- **Education Match (10%)**: Degree mapping and relevance.
- **Domain Expertise (10%)**: Years spent in product vs. consulting roles.
- **AI Reasoning Confidence (10%)**: Quality of the inferred match.
- **Certification Score (5%)**: Value of validated credentials.
- **Fraud Risk Adjustment (5%)**: Dynamic penalties for logically contradictory profiles (e.g., claiming 10 years of experience at a company founded 3 years ago).

### Stage 3: Explainable Reasoning
We do not just output raw numbers. As the final phase of the ranker, the system generates an explainable, 1-2 sentence reasoning report for each of the top 100 candidates. This reasoning explicitly surfaces their strengths and honest gaps based purely on the facts in their profile (e.g., *“Strong NLP background; some concern on notice period (120 days) but otherwise strong fit”*).

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
