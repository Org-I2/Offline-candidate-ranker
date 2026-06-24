"""
Sandbox Streamlit Demo App
Accepts a small candidate JSON upload (up to 100 candidates) plus JD text input.
Runs the runtime ranking pipeline on the sample.
Outputs a downloadable CSV and shows top-10 results table.

Usage:
    streamlit run sandbox/app.py
"""

import importlib.util
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Offline Candidate Ranker — Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1 { color: #e2e8f0; font-weight: 700; }
    h2, h3 { color: #cbd5e0; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stage-row { display: flex; justify-content: space-between; padding: 4px 0; }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 8px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1>🎯 Offline AI Candidate Ranker</h1>
        <p style="color:#718096; font-size:1.1rem;">
            Semantic + BM25 hybrid ranking · 100% offline · CPU only
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — instructions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ How it works")
    st.info(
        """
        1. Upload candidates (JSONL/JSON, max 100)
        2. Paste or edit the JD text
        3. Click **Run Ranking**
        4. Download the ranked CSV
        """
    )
    st.markdown("### 📦 Pipeline stages")
    stages = [
        ("Parse JD", "~1s"),
        ("Feature extract", "~2s"),
        ("BM25 retrieve", "~1s"),
        ("Semantic embed", "~3s"),
        ("Deep score", "~1s"),
        ("Build reasoning", "<1s"),
    ]
    for name, est in stages:
        st.markdown(f"**{name}** &nbsp; `{est}`")

    st.divider()
    st.caption("Model: all-MiniLM-L6-v2 · No API calls · No GPU")

# ---------------------------------------------------------------------------
# Main columns
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("### 📂 Upload Candidates")
    uploaded_file = st.file_uploader(
        "Drag & drop candidates.jsonl or .json (max 100 candidates)",
        type=["jsonl", "json"],
        help="Each line should be a JSON object with candidate fields",
    )

    st.markdown("### 📝 Job Description")
    default_jd = (
        "We are looking for a Senior AI Engineer with 5-8 years of experience. "
        "Required skills: Python, embeddings, vector database, ranking evaluation, BM25, "
        "semantic search, learning to rank. "
        "You will build and ship production ML systems. "
        "Experience with FAISS, Pinecone, or similar vector stores required. "
        "NDCG, MRR, MAP evaluation experience needed. "
        "Startup experience valued. Must be willing to work scrappy and ship fast."
    )
    jd_text = st.text_area(
        "Paste your Job Description here",
        value=default_jd,
        height=200,
        help="Full JD text — the parser extracts required skills, seniority, and signals",
    )

    run_btn = st.button(
        "🚀 Run Ranking",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or not jd_text.strip()),
    )

# ---------------------------------------------------------------------------
# Run pipeline on button click
# ---------------------------------------------------------------------------

def _parse_uploaded_candidates(uploaded) -> list[dict]:
    """Parse uploaded file (jsonl or json) into list of dicts."""
    content = uploaded.read().decode("utf-8")
    candidates = []
    # Try JSONL first
    for i, line in enumerate(content.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not candidates:
        # Try as a JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                candidates = data
        except json.JSONDecodeError:
            pass
    return candidates[:100]  # cap at 100


def _run_sandbox_pipeline(candidates: list[dict], jd_text: str) -> tuple[pd.DataFrame, dict]:
    """
    Lightweight sandbox pipeline that runs end-to-end on a small candidate set.
    Returns (output_df, stage_timings).
    """
    from src.jd_parser import JDParser
    from src.scorer import (
        compute_required_skill_coverage,
        compute_trajectory_score,
        compute_behavioral_availability,
        compute_consistency_score,
        compute_skill_depth_score,
        compute_experience_range_fit,
        compute_composite_score,
    )
    from src.reasoning import build_reasoning

    timings = {}

    # ---- Stage 1: Parse JD from text ----
    t0 = time.perf_counter()
    parser = JDParser(static_dir=REPO_ROOT / "static")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as jdf:
        jdf.write(jd_text)
        jd_path = jdf.name
    jd = parser.parse(jd_path)
    Path(jd_path).unlink(missing_ok=True)
    timings["Parse JD"] = time.perf_counter() - t0

    required_skills = jd.get("required_skills", [])
    functional_equivalents = jd.get("skill_functional_equivalents", {})
    seniority_range = jd.get("seniority_range", [5, 9])
    seniority_level = jd.get("seniority_level", "senior")
    _level_map = {
        "intern": 0, "junior": 1, "associate": 2, "mid": 3,
        "senior": 4, "lead": 5, "staff": 5, "principal": 6,
        "director": 7, "vp": 8, "chief": 9,
    }
    seniority_target = _level_map.get(seniority_level.lower(), 4)

    # ---- Stage 2: Feature extraction (inline lightweight version) ----
    t0 = time.perf_counter()
    _feat_spec = importlib.util.spec_from_file_location(
        "feature_extractor",
        REPO_ROOT / "precompute" / "02_feature_extractor.py",
    )
    feat_mod = importlib.util.module_from_spec(_feat_spec)
    _feat_spec.loader.exec_module(feat_mod)

    records = []
    for i, c in enumerate(candidates):
        cid = str(c.get("candidate_id") or c.get("id") or f"cand_{i:05d}")
        c.setdefault("candidate_id", cid)

    # Write temp candidates.jsonl + dummy honeypot flags
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cands_file = tmpdir / "candidates.jsonl"
        with open(cands_file, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

        # Dummy honeypot flags (no candidate flagged for sandbox)
        import pyarrow as pa
        import pyarrow.parquet as pq
        flags_df = pd.DataFrame({
            "candidate_id": [str(c.get("candidate_id", f"cand_{i}")) for i, c in enumerate(candidates)],
            "honeypot_score": [0] * len(candidates),
            "honeypot_reasons": [[]] * len(candidates),
            "bm25_seed_score": [0.0] * len(candidates),
        })
        flags_path = tmpdir / "flags.parquet"
        features_path = tmpdir / "features.parquet"
        flags_df.to_parquet(flags_path, index=False)

        feat_mod.extract_features(
            str(cands_file),
            str(flags_path),
            str(REPO_ROOT / "static"),
            str(features_path),
        )
        features_df = pd.read_parquet(features_path)

    timings["Feature extract"] = time.perf_counter() - t0

    # ---- Stage 3: Embed JD + candidates with MiniLM ----
    t0 = time.perf_counter()
    model_dir = REPO_ROOT / "models" / "minilm"
    if model_dir.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_dir), device="cpu")
        jd_query = " ".join(required_skills) + " " + jd_text[:500]
        jd_emb = model.encode([jd_query], normalize_embeddings=True, convert_to_numpy=True)[0]

        texts = features_df["raw_profile_text"].fillna("").tolist()
        cand_embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        sem_scores = (cand_embs @ jd_emb.reshape(-1, 1)).flatten()
        features_df["semantic_similarity"] = sem_scores.tolist()
    else:
        # Fallback: no embedding model available
        features_df["semantic_similarity"] = 0.5
    timings["Semantic embed"] = time.perf_counter() - t0

    # ---- Stage 4: Deep score ----
    t0 = time.perf_counter()
    scored_rows = []
    for _, row in features_df.iterrows():
        cid = str(row["candidate_id"])
        try:
            def _gl(val):
                return val if isinstance(val, list) else []
            def _gd(val):
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return {}
                return {}

            cov = compute_required_skill_coverage(
                _gl(row.get("skills_list")), required_skills, functional_equivalents
            )
            traj = compute_trajectory_score(
                float(row.get("trajectory_slope") or 0.0),
                int(row.get("current_seniority") or 3),
                seniority_target,
            )
            behav = compute_behavioral_availability(
                row.get("platform_last_active_days"),
                row.get("recruiter_response_rate"),
            )
            consist = compute_consistency_score(
                float(row.get("title_skill_alignment") or 0.5),
                float(row.get("profile_completeness") or 0.5),
            )
            depth = compute_skill_depth_score(
                _gd(row.get("skill_recency_map")), required_skills
            )
            exp_fit = compute_experience_range_fit(
                float(row.get("total_years_exp_computed") or 0.0), seniority_range
            )
            signals = {
                "semantic_similarity": float(row.get("semantic_similarity") or 0.0),
                "required_skill_coverage": cov,
                "trajectory_score": traj,
                "behavioral_availability": behav,
                "consistency_score": consist,
                "skill_depth_score": depth,
                "experience_range_fit": exp_fit,
            }
            weights = {
                "semantic_similarity": 0.25,
                "required_skill_coverage": 0.25,
                "trajectory_score": 0.20,
                "behavioral_availability": 0.10,
                "consistency_score": 0.10,
                "skill_depth_score": 0.05,
                "experience_range_fit": 0.05,
            }
            composite = compute_composite_score(signals, weights)
            r = row.to_dict()
            r.update(signals)
            r["required_skill_coverage"] = cov
            r["composite_score"] = composite
            scored_rows.append(r)
        except Exception:
            pass

    timings["Deep score"] = time.perf_counter() - t0

    scored_df = pd.DataFrame(scored_rows).sort_values("composite_score", ascending=False).reset_index(drop=True)

    # ---- Stage 5: Build reasoning ----
    t0 = time.perf_counter()
    reasonings = []
    for i, row in scored_df.iterrows():
        rank = i + 1
        try:
            r = build_reasoning(row.to_dict(), jd, rank)
            r = r.replace("\n", " ").strip()
        except Exception:
            r = f"Ranked {rank} based on composite score."
        reasonings.append(r)
    scored_df["reasoning"] = reasonings
    scored_df["rank"] = scored_df.index + 1
    scored_df["score"] = scored_df["composite_score"].round(4)
    timings["Build reasoning"] = time.perf_counter() - t0

    return scored_df, timings


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
with col_right:
    if run_btn and uploaded_file is not None:
        with st.spinner("Running ranking pipeline..."):
            try:
                candidates = _parse_uploaded_candidates(uploaded_file)
                if not candidates:
                    st.error("No valid candidates found in the uploaded file.")
                elif len(candidates) > 100:
                    st.warning("Capped at 100 candidates for the sandbox.")
                    candidates = candidates[:100]

                scored_df, timings = _run_sandbox_pipeline(candidates, jd_text)

                # ---- Summary metrics ----
                st.markdown("### ✅ Pipeline Complete")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Candidates", len(candidates))
                m2.metric("Ranked", min(len(scored_df), 100))
                m3.metric("Top Score", f"{scored_df['score'].iloc[0]:.3f}" if not scored_df.empty else "N/A")
                m4.metric("Total Time", f"{sum(timings.values()):.1f}s")

                # ---- Stage timings ----
                st.markdown("#### ⏱ Stage timings")
                timing_cols = st.columns(len(timings))
                for col, (stage, t) in zip(timing_cols, timings.items()):
                    col.metric(stage, f"{t:.2f}s")

                st.divider()

                # ---- Top 10 table ----
                st.markdown("### 🏆 Top 10 Candidates")
                display_cols = ["rank", "full_name", "current_title", "score", "reasoning"]
                available = [c for c in display_cols if c in scored_df.columns]
                top10 = scored_df.head(10)[available].copy()

                if "rank" in top10.columns:
                    top10["rank"] = top10["rank"].apply(lambda r: f"#{r}")
                if "score" in top10.columns:
                    top10["score"] = top10["score"].apply(lambda s: f"{float(s):.4f}")

                st.dataframe(
                    top10,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "rank": st.column_config.TextColumn("Rank", width="small"),
                        "full_name": st.column_config.TextColumn("Name", width="medium"),
                        "current_title": st.column_config.TextColumn("Title", width="medium"),
                        "score": st.column_config.TextColumn("Score", width="small"),
                        "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                    },
                )

                # ---- Download button ----
                st.markdown("### 📥 Download Full Results")
                output_cols = ["candidate_id", "rank", "score", "reasoning"]
                if "full_name" in scored_df.columns:
                    output_cols = ["candidate_id", "full_name"] + ["rank", "score", "reasoning"]
                download_df = scored_df.head(100)[
                    [c for c in output_cols if c in scored_df.columns]
                ].copy()
                csv_bytes = download_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download submission.csv",
                    data=csv_bytes,
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

    else:
        # Placeholder when not yet run
        st.markdown("### 📊 Results will appear here")
        st.info(
            "Upload a candidates file and paste a job description, then click **Run Ranking** to see results."
        )
        st.markdown(
            """
            **Expected candidate fields:**
            ```json
            {
              "candidate_id": "unique_id",
              "full_name": "Alice Smith",
              "current_title": "ML Engineer",
              "current_company": "TechCorp",
              "skills": [{"name": "python", "proficiency": "expert", "years_used": 5}],
              "work_history": [...],
              "education": [...]
            }
            ```
            """
        )
