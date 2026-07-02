"""
Streamlit sandbox app.

Runs a true 100-candidate mini version of the local ranking pipeline:
uploaded candidates -> real precompute artifacts -> rank.py scoring functions -> CSV.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATIC_DIR = REPO_ROOT / "static"
MODEL_DIR = REPO_ROOT / "models" / "minilm"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


st.set_page_config(
    page_title="Offline Candidate Ranker - Sandbox",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Inter, Segoe UI, sans-serif; }
    .main, .stApp { background-color: #0f1117; }
    h1 { color: #e2e8f0; font-weight: 700; }
    h2, h3 { color: #cbd5e0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_module(filename: str, module_name: str):
    path = REPO_ROOT / "precompute" / filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def _load_precompute_modules() -> dict[str, Any]:
    return {
        "honeypot": _load_module("01_honeypot_detector.py", "sandbox_honeypot_detector"),
        "features": _load_module("02_feature_extractor.py", "sandbox_feature_extractor"),
        "embedder": _load_module("03_embedder.py", "sandbox_embedder"),
        "bm25": _load_module("04_build_bm25.py", "sandbox_build_bm25"),
    }


@st.cache_resource(show_spinner=False)
def _load_rank_module():
    import rank as rank_mod

    return rank_mod


@st.cache_resource(show_spinner="Loading MiniLM model...")
def _load_minilm_model(model_source: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_source, device="cpu")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _first_present(*values: Any) -> Any:
    for value in values:
        if _present(value):
            return value
    return None


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _days_since(value: Any) -> float | None:
    if not _present(value):
        return None
    try:
        ts = pd.Timestamp(pd.to_datetime(value, errors="raise"))
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return float((pd.Timestamp.today() - ts).days)
    except Exception:
        return None


def _normalize_skills(skills: Any) -> Any:
    if not isinstance(skills, list):
        return skills

    normalized: list[Any] = []
    for item in skills:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        skill = dict(item)
        if not _present(skill.get("years_used")) and not _present(skill.get("years")):
            months = skill.get("duration_months")
            try:
                if months is not None:
                    skill["years_used"] = round(max(0.0, float(months)) / 12.0, 2)
            except (TypeError, ValueError):
                pass
        normalized.append(skill)
    return normalized


def _normalize_education(education: Any) -> Any:
    if not isinstance(education, list):
        return education

    normalized: list[Any] = []
    for item in education:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        edu = dict(item)
        end_year = edu.get("end_year")
        if _present(end_year):
            edu.setdefault("graduation_year", end_year)
            edu.setdefault("end", str(end_year))
        normalized.append(edu)
    return normalized


def _set_if_present(candidate: dict, key: str, value: Any) -> None:
    if key not in candidate and _present(value):
        candidate[key] = value


def _normalize_candidate_schema(candidate: dict, idx: int) -> dict:
    """Add flat aliases expected by legacy pipeline code while preserving raw fields."""
    c = copy.deepcopy(candidate)
    profile = _as_dict(c.get("profile"))
    signals = _as_dict(c.get("redrob_signals"))

    candidate_id = _first_present(c.get("candidate_id"), c.get("id"), f"cand_{idx:05d}")
    c["candidate_id"] = str(candidate_id)

    full_name = _first_present(
        c.get("full_name"),
        c.get("name"),
        profile.get("anonymized_name"),
        profile.get("full_name"),
        profile.get("name"),
    )
    current_title = _first_present(c.get("current_title"), c.get("title"), profile.get("current_title"))
    current_company = _first_present(
        c.get("current_company"),
        c.get("current_employer"),
        profile.get("current_company"),
    )
    current_location = _first_present(c.get("current_location"), c.get("location"), profile.get("location"))
    summary = _first_present(c.get("summary"), c.get("bio"), profile.get("summary"))
    years = _first_present(
        c.get("total_years_experience"),
        c.get("years_of_experience"),
        profile.get("years_of_experience"),
    )
    work_history = _first_present(c.get("work_history"), c.get("career_history"), c.get("experience"))

    _set_if_present(c, "full_name", full_name)
    _set_if_present(c, "name", full_name)
    _set_if_present(c, "current_title", current_title)
    _set_if_present(c, "current_company", current_company)
    _set_if_present(c, "current_employer", current_company)
    _set_if_present(c, "current_location", current_location)
    _set_if_present(c, "location", current_location)
    _set_if_present(c, "summary", summary)
    _set_if_present(c, "total_years_experience", years)
    _set_if_present(c, "years_of_experience", years)
    _set_if_present(c, "work_history", work_history)

    if "skills" in c:
        c["skills"] = _normalize_skills(c["skills"])
    if "education" in c:
        c["education"] = _normalize_education(c["education"])

    response_rate = signals.get("recruiter_response_rate")
    last_active = signals.get("last_active_date")
    github_score = signals.get("github_activity_score")
    notice_period = signals.get("notice_period_days")
    completeness = signals.get("profile_completeness_score")

    _set_if_present(c, "recruiter_response_rate", response_rate)
    _set_if_present(c, "platform_last_active_days", _days_since(last_active))
    _set_if_present(c, "github_activity_score", github_score)
    _set_if_present(c, "notice_period_days", notice_period)
    _set_if_present(c, "profile_completeness_score", completeness)

    try:
        has_open_source = github_score is not None and float(github_score) > 0
    except (TypeError, ValueError):
        has_open_source = False
    if has_open_source:
        c.setdefault("has_open_source", True)
        c.setdefault("open_source", True)
        c.setdefault("github", {"activity_score": github_score})

    if not profile:
        c["profile"] = {
            "anonymized_name": full_name,
            "current_title": current_title,
            "current_company": current_company,
            "location": current_location,
            "years_of_experience": years,
            "summary": summary,
        }

    return c


def _parse_uploaded_candidates(uploaded) -> list[dict]:
    content = uploaded.read().decode("utf-8-sig")
    stripped = content.strip()
    if not stripped:
        return []

    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            candidates = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            nested = data.get("candidates")
            if isinstance(nested, list):
                candidates = [item for item in nested if isinstance(item, dict)]
            else:
                candidates = [data]
        else:
            candidates = []
    except json.JSONDecodeError:
        candidates = []
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    candidates.append(item)
            except json.JSONDecodeError:
                continue

    return candidates[:100]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jd_input(tmpdir: Path, jd_text: str, jd_upload: Any | None) -> tuple[Path, str]:
    if jd_upload is not None:
        suffix = Path(jd_upload.name or "").suffix.lower()
        if suffix not in {".docx", ".pdf", ".txt"}:
            suffix = ".txt"
        jd_path = tmpdir / f"uploaded_jd{suffix}"
        jd_path.write_bytes(jd_upload.getvalue())
        return jd_path, f"Uploaded file: {jd_upload.name}"

    jd_path = tmpdir / "jd.txt"
    jd_path.write_text(jd_text, encoding="utf-8")
    return jd_path, "Text area"


def _timed(timings: dict[str, float], name: str, fn: Callable, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    timings[name] = time.perf_counter() - t0
    return result


def _model_source_and_path(tmpdir: Path) -> tuple[str, Path, str]:
    if MODEL_DIR.exists():
        return str(MODEL_DIR), MODEL_DIR, "Local models/minilm"

    placeholder = tmpdir / "hf_minilm_placeholder"
    placeholder.mkdir(parents=True, exist_ok=True)
    return MODEL_NAME, placeholder, "Hugging Face all-MiniLM-L6-v2"


def _build_embeddings_with_cached_model(
    embedder_mod,
    features_path: Path,
    model_source: str,
    model_path_for_module: Path,
    faiss_index_path: Path,
    faiss_ids_path: Path,
) -> None:
    model = _load_minilm_model(model_source)
    original = embedder_mod.SentenceTransformer
    embedder_mod.SentenceTransformer = lambda *args, **kwargs: model
    try:
        embedder_mod.build_embeddings(
            str(features_path),
            str(model_path_for_module),
            str(faiss_index_path),
            str(faiss_ids_path),
            batch_size=32,
        )
    finally:
        embedder_mod.SentenceTransformer = original


def _parse_and_embed_jd_with_cached_model(
    rank_mod,
    jd_path: Path,
    model_source: str,
    model_path_for_rank: Path,
):
    model = _load_minilm_model(model_source)

    import sentence_transformers

    original_sentence_transformer = sentence_transformers.SentenceTransformer
    original_model_dir = rank_mod.MODEL_DIR
    sentence_transformers.SentenceTransformer = lambda *args, **kwargs: model
    rank_mod.MODEL_DIR = model_path_for_rank
    try:
        return rank_mod.parse_and_embed_jd(str(jd_path))
    finally:
        sentence_transformers.SentenceTransformer = original_sentence_transformer
        rank_mod.MODEL_DIR = original_model_dir


def _load_temp_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    import faiss

    with open(paths["bm25"], "rb") as f:
        bm25_payload = pickle.load(f)

    return {
        "honeypot_df": pd.read_parquet(paths["honeypot_flags"]),
        "features_df": pd.read_parquet(paths["features"]),
        "faiss_index": faiss.read_index(str(paths["faiss_index"])),
        "faiss_ids": np.load(str(paths["faiss_ids"]), allow_pickle=True),
        "bm25": bm25_payload["bm25"],
        "bm25_ids": np.load(str(paths["bm25_ids"]), allow_pickle=True),
    }


def _format_submission(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = scored_df.head(100).copy().reset_index(drop=True)
    df["rank"] = df.index + 1
    df["score"] = df["composite_score"].round(6)
    final_df = df[OUTPUT_COLUMNS].copy()
    final_df["candidate_id"] = final_df["candidate_id"].astype(str)
    final_df["rank"] = final_df["rank"].astype(int)
    final_df["score"] = final_df["score"].astype(float)
    final_df["reasoning"] = final_df["reasoning"].astype(str)
    return final_df


def _run_mini_pipeline(
    candidates: list[dict],
    jd_text: str,
    jd_upload: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any]]:
    def emit(message: str) -> None:
        if progress:
            progress(message)

    modules = _load_precompute_modules()
    rank_mod = _load_rank_module()
    timings: dict[str, float] = {}
    meta: dict[str, Any] = {"input_candidates": len(candidates)}

    normalized = [_normalize_candidate_schema(c, i) for i, c in enumerate(candidates)]

    with tempfile.TemporaryDirectory(prefix="candidate_ranker_sandbox_") as tmp:
        tmpdir = Path(tmp)
        paths = {
            "candidates": tmpdir / "candidates.jsonl",
            "honeypot_flags": tmpdir / "honeypot_flags.parquet",
            "features": tmpdir / "features.parquet",
            "faiss_index": tmpdir / "faiss.index",
            "faiss_ids": tmpdir / "candidate_ids_faiss.npy",
            "bm25": tmpdir / "bm25.pkl",
            "bm25_ids": tmpdir / "candidate_ids_bm25.npy",
        }

        emit("Writing uploaded data to temporary files...")
        _timed(timings, "Write temp inputs", _write_jsonl, paths["candidates"], normalized)
        jd_path, jd_source = _timed(timings, "Write JD input", _write_jd_input, tmpdir, jd_text, jd_upload)
        meta["jd_source"] = jd_source

        emit("Parsing JD for precompute signals...")
        from src.jd_parser import JDParser

        jd_for_precompute = _timed(
            timings,
            "Parse JD for precompute",
            JDParser(static_dir=STATIC_DIR).parse,
            str(jd_path),
        )
        jd_required_skills = jd_for_precompute.get("required_skills", [])

        emit("Running real honeypot detection...")
        _timed(
            timings,
            "Honeypot detection",
            modules["honeypot"].detect_honeypots,
            str(paths["candidates"]),
            str(paths["honeypot_flags"]),
            str(STATIC_DIR / "company_founding_years.json"),
            jd_required_skills,
        )
        honeypot_df = pd.read_parquet(paths["honeypot_flags"])
        meta["flagged_candidates"] = int(honeypot_df["honeypot_score"].sum()) if not honeypot_df.empty else 0

        emit("Running real feature extraction...")
        _timed(
            timings,
            "Feature extraction",
            modules["features"].extract_features,
            str(paths["candidates"]),
            str(paths["honeypot_flags"]),
            str(STATIC_DIR),
            str(paths["features"]),
        )
        features_df = pd.read_parquet(paths["features"])
        meta["feature_rows"] = len(features_df)

        if features_df.empty:
            empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
            meta["model_source"] = "Not loaded - no non-honeypot candidates"
            return empty, pd.DataFrame(), timings, meta

        model_source, model_path_for_module, model_label = _model_source_and_path(tmpdir)
        meta["model_source"] = model_label

        emit("Building real MiniLM/FAISS artifact...")
        _timed(
            timings,
            "Embedding + FAISS",
            _build_embeddings_with_cached_model,
            modules["embedder"],
            paths["features"],
            model_source,
            model_path_for_module,
            paths["faiss_index"],
            paths["faiss_ids"],
        )

        emit("Building real BM25 artifact...")
        _timed(
            timings,
            "BM25 index",
            modules["bm25"].build_bm25,
            str(paths["features"]),
            str(paths["bm25"]),
            str(paths["bm25_ids"]),
        )

        emit("Loading temporary artifacts...")
        artifacts = _timed(timings, "Load mini artifacts", _load_temp_artifacts, paths)

        emit("Parsing and embedding JD through rank.py logic...")
        jd, jd_embedding = _timed(
            timings,
            "Parse + embed JD",
            _parse_and_embed_jd_with_cached_model,
            rank_mod,
            jd_path,
            model_source,
            model_path_for_module,
        )

        emit("Applying rank.py honeypot filter...")
        clean_df = _timed(
            timings,
            "Apply honeypot filter",
            rank_mod.apply_honeypot_filter,
            artifacts["features_df"],
            artifacts["honeypot_df"],
        )
        meta["rankable_candidates"] = len(clean_df)

        if clean_df.empty:
            empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
            return empty, pd.DataFrame(), timings, meta

        emit("Retrieving candidates with rank.py BM25...")
        top_bm25_ids = _timed(
            timings,
            "BM25 retrieve",
            rank_mod.bm25_retrieve,
            jd,
            artifacts["bm25"],
            artifacts["bm25_ids"],
            clean_df,
            min(5000, len(clean_df)),
        )

        if not top_bm25_ids:
            empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
            return empty, pd.DataFrame(), timings, meta

        emit("Reranking candidates with rank.py semantic rerank...")
        semantic_top = _timed(
            timings,
            "Semantic rerank",
            rank_mod.semantic_rerank,
            top_bm25_ids,
            jd_embedding,
            artifacts["faiss_index"],
            artifacts["faiss_ids"],
            min(500, len(top_bm25_ids)),
        )

        if not semantic_top:
            empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
            return empty, pd.DataFrame(), timings, meta

        emit("Scoring candidates with rank.py deep_score...")
        scored_df = _timed(
            timings,
            "Deep score",
            rank_mod.deep_score,
            semantic_top,
            clean_df,
            jd,
            min(150, len(semantic_top)),
        )

        if not scored_df.empty:
            max_score = scored_df["composite_score"].max()
            if max_score > 0:
                scale_factor = 0.95 / max_score
                scored_df["composite_score"] = (scored_df["composite_score"] * scale_factor).clip(upper=0.99)
                scored_df["composite_score"] = scored_df["composite_score"].round(6)

        scored_df = scored_df.head(100).reset_index(drop=True)

        emit("Building reasoning with rank.py...")
        scored_df = _timed(
            timings,
            "Build reasoning",
            rank_mod.build_reasoning_column,
            scored_df,
            jd,
        )

        submission_df = _timed(timings, "Format CSV", _format_submission, scored_df)
        meta["ranked_candidates"] = len(submission_df)
        return submission_df, scored_df, timings, meta


st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1>Offline AI Candidate Ranker</h1>
        <p style="color:#718096; font-size:1.05rem;">
            True mini pipeline: real honeypot detection, features, FAISS, BM25, and rank.py scoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

with st.sidebar:
    st.markdown("### How it works")
    st.info(
        """
        1. Upload JSONL or JSON candidates, up to 100
        2. Paste the JD text
        3. Run the mini pipeline
        4. Download the submission-shaped CSV
        """
    )
    st.markdown("### Runtime notes")
    st.caption("Uses temp artifacts only. It does not depend on data/candidates.jsonl, artifacts/, or models/ in Streamlit Cloud.")
    st.caption("If models/minilm is missing, the app loads all-MiniLM-L6-v2 from Hugging Face and caches it for the session.")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("### Upload Candidates")
    uploaded_file = st.file_uploader(
        "Candidates JSONL or JSON array",
        type=["jsonl", "json"],
        help="The sandbox caps ranking to the first 100 valid candidate objects.",
    )

    st.markdown("### Job Description")
    jd_upload = st.file_uploader(
        "Optional JD document (.docx, .pdf, .txt)",
        type=["docx", "pdf", "txt"],
        help="If provided, the app parses this file with src/jd_parser.py. Otherwise it uses the text below.",
    )

    default_jd = (
        "We are looking for a Senior AI Engineer with 5-8 years of experience. "
        "Required skills: Python, embeddings, vector database, ranking evaluation, BM25, "
        "semantic search, learning to rank. You will build and ship production ML systems. "
        "Experience with FAISS, Pinecone, or similar vector stores required. "
        "NDCG, MRR, MAP evaluation experience needed. Startup experience valued. "
        "Must be willing to work scrappy and ship fast."
    )
    jd_text = st.text_area(
        "Paste JD text",
        value=default_jd,
        height=230,
    )

    run_btn = st.button(
        "Run Mini Pipeline",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or (jd_upload is None and not jd_text.strip())),
    )

with col_right:
    if run_btn and uploaded_file is not None:
        progress_box = st.empty()

        try:
            candidates = _parse_uploaded_candidates(uploaded_file)
            if not candidates:
                st.error("No valid candidate objects found in the uploaded file.")
                st.stop()

            if len(candidates) == 100:
                st.info("Using the first 100 candidates from the upload.")

            with st.spinner("Running the true mini ranking pipeline..."):
                submission_df, scored_df, timings, meta = _run_mini_pipeline(
                    candidates,
                    jd_text,
                    jd_upload=jd_upload,
                    progress=lambda msg: progress_box.info(msg),
                )
            progress_box.empty()

            st.markdown("### Pipeline Complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Uploaded", meta.get("input_candidates", len(candidates)))
            m2.metric("Flagged", meta.get("flagged_candidates", 0))
            m3.metric("Ranked", len(submission_df))
            top_score = f"{submission_df['score'].iloc[0]:.3f}" if not submission_df.empty else "N/A"
            m4.metric("Top Score", top_score)

            st.caption(f"Model source: {meta.get('model_source', 'unknown')}")
            st.caption(f"JD source: {meta.get('jd_source', 'unknown')}")

            timing_df = pd.DataFrame(
                [{"stage": stage, "seconds": round(seconds, 3)} for stage, seconds in timings.items()]
            )
            st.markdown("#### Stage Timings")
            st.dataframe(timing_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Top Candidates")
            if submission_df.empty:
                st.warning("No candidates were rankable after honeypot filtering and retrieval.")
            else:
                display_df = submission_df.copy()
                if not scored_df.empty:
                    extra_cols = [c for c in ["full_name", "current_title", "current_company"] if c in scored_df.columns]
                    if extra_cols:
                        extras = scored_df[["candidate_id"] + extra_cols].copy()
                        display_df = display_df.merge(extras, on="candidate_id", how="left")
                        ordered_cols = ["rank", "candidate_id"] + extra_cols + ["score", "reasoning"]
                        display_df = display_df[[c for c in ordered_cols if c in display_df.columns]]

                st.dataframe(
                    display_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Download")
            csv_bytes = submission_df[OUTPUT_COLUMNS].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as exc:
            progress_box.empty()
            st.error(f"Pipeline error: {exc}")
            import traceback

            st.code(traceback.format_exc())
    else:
        st.markdown("### Results will appear here")
        st.info("Upload candidates and paste a JD, then run the mini pipeline.")
        st.markdown(
            """
            Expected output CSV columns:
            ```text
            candidate_id,rank,score,reasoning
            ```
            """
        )
