"""
Streamlit sandbox for the Offline Candidate Ranker.

Runs a true small-candidate version of the full pipeline:
upload candidates + JD text, build temporary precompute artifacts, rank with
rank.py's runtime functions, and download the submission-shaped CSV.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
PRECOMPUTE_DIR = REPO_ROOT / "precompute"
STATIC_DIR = REPO_ROOT / "static"
LOCAL_MODEL_DIR = REPO_ROOT / "models" / "minilm"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rank
from src.jd_parser import JDParser


st.set_page_config(
    page_title="Offline Candidate Ranker Sandbox",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
    .stApp { background-color: #101216; }
    h1, h2, h3 { color: #eef2f7; }
    .subtle { color: #98a2b3; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MiniLM model...")
def _load_minilm_model():
    from sentence_transformers import SentenceTransformer

    model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_ID
    return SentenceTransformer(model_source, device="cpu")


@st.cache_resource(show_spinner=False)
def _load_precompute_module(filename: str, module_name: str):
    path = PRECOMPUTE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Precompute script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timed(timings: dict[str, float], name: str, fn: Callable, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    timings[name] = time.perf_counter() - start
    return result


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _days_since(date_value: Any) -> float | None:
    if date_value is None:
        return None
    parsed = pd.to_datetime(date_value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return float(max(0, (pd.Timestamp.now(tz="UTC") - parsed).days))


def _parse_uploaded_candidates(uploaded_file) -> list[dict[str, Any]]:
    content = uploaded_file.getvalue().decode("utf-8-sig")
    if not content.strip():
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            return [row for row in data["candidates"] if isinstance(row, dict)]
    except json.JSONDecodeError:
        pass

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_no} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} must be a JSON object.")
        rows.append(row)
    return rows


def _normalize_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(candidate)
    profile = dict(candidate.get("profile") or {})
    signals = dict(candidate.get("redrob_signals") or {})

    candidate_id = _first_present(
        candidate.get("candidate_id"),
        candidate.get("id"),
        profile.get("candidate_id"),
        f"candidate_{index + 1:03d}",
    )
    normalized["candidate_id"] = str(candidate_id)

    full_name = _first_present(
        profile.get("anonymized_name"),
        profile.get("full_name"),
        candidate.get("full_name"),
        candidate.get("name"),
    )
    current_title = _first_present(profile.get("current_title"), candidate.get("current_title"))
    current_company = _first_present(profile.get("current_company"), candidate.get("current_company"), candidate.get("current_employer"))
    current_location = _first_present(profile.get("location"), candidate.get("current_location"), candidate.get("location"))
    years_experience = _first_present(
        profile.get("years_of_experience"),
        candidate.get("total_years_experience"),
        candidate.get("years_of_experience"),
    )
    summary = _first_present(profile.get("summary"), candidate.get("summary"), candidate.get("bio"))

    if full_name is not None:
        normalized["full_name"] = full_name
        profile["anonymized_name"] = full_name
    if current_title is not None:
        normalized["current_title"] = current_title
        profile["current_title"] = current_title
    if current_company is not None:
        normalized["current_company"] = current_company
        normalized["current_employer"] = current_company
        profile["current_company"] = current_company
    if current_location is not None:
        normalized["current_location"] = current_location
        profile["location"] = current_location
    if years_experience is not None:
        normalized["total_years_experience"] = years_experience
        profile["years_of_experience"] = years_experience
    if summary is not None:
        normalized["summary"] = summary
        profile["summary"] = summary

    career_history = candidate.get("career_history")
    if not isinstance(career_history, list) or not career_history:
        career_history = candidate.get("work_history")
    if not isinstance(career_history, list) or not career_history:
        career_history = candidate.get("experience")
    if isinstance(career_history, list):
        normalized["career_history"] = career_history
        normalized["work_history"] = career_history

    response_rate = _first_present(signals.get("recruiter_response_rate"), candidate.get("recruiter_response_rate"))
    if response_rate is not None:
        normalized["recruiter_response_rate"] = response_rate
        signals["recruiter_response_rate"] = response_rate

    last_active = _first_present(signals.get("last_active_date"), candidate.get("last_active_date"))
    active_days = _first_present(candidate.get("platform_last_active_days"), _days_since(last_active))
    if active_days is not None:
        normalized["platform_last_active_days"] = active_days
    if last_active is not None:
        signals["last_active_date"] = last_active

    github_score = _first_present(signals.get("github_activity_score"), candidate.get("github_activity_score"))
    has_github_activity = False
    try:
        has_github_activity = github_score is not None and float(github_score) > 0
    except (TypeError, ValueError):
        has_github_activity = False
    if has_github_activity:
        normalized["open_source"] = True
        normalized.setdefault("github", "activity-present")
        signals["github_activity_score"] = github_score

    normalized["profile"] = profile
    if signals:
        normalized["redrob_signals"] = signals

    for key in ("skills", "certifications", "education"):
        if key in candidate:
            normalized[key] = candidate[key]

    return normalized


def _normalize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_candidate(candidate, idx) for idx, candidate in enumerate(candidates)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_and_embed_jd(jd_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    parser = JDParser(static_dir=STATIC_DIR)
    jd = parser.parse(str(jd_path))

    required_skills_text = " ".join(jd.get("required_skills", []))
    raw_jd_text = jd.get("raw_jd_text", "")[:1000]
    jd_query_text = f"{required_skills_text} {raw_jd_text}".strip()

    model = _load_minilm_model()
    jd_embedding = model.encode(
        [jd_query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(np.float32)
    norm = np.linalg.norm(jd_embedding)
    if norm > 0:
        jd_embedding = jd_embedding / norm
    jd["text_embedding"] = jd_embedding.tolist()
    return jd, jd_embedding


def _build_embeddings_with_fallback(
    embedder_module,
    features_path: Path,
    faiss_index_path: Path,
    faiss_ids_path: Path,
    temp_model_marker: Path,
) -> None:
    if LOCAL_MODEL_DIR.exists():
        embedder_module.build_embeddings(
            str(features_path),
            str(LOCAL_MODEL_DIR),
            str(faiss_index_path),
            str(faiss_ids_path),
            batch_size=32,
        )
        return

    temp_model_marker.mkdir(parents=True, exist_ok=True)
    original_sentence_transformer = embedder_module.SentenceTransformer

    def cached_sentence_transformer(_model_dir: str, device: str = "cpu"):
        return _load_minilm_model()

    embedder_module.SentenceTransformer = cached_sentence_transformer
    try:
        embedder_module.build_embeddings(
            str(features_path),
            str(temp_model_marker),
            str(faiss_index_path),
            str(faiss_ids_path),
            batch_size=32,
        )
    finally:
        embedder_module.SentenceTransformer = original_sentence_transformer


def _load_mini_artifacts(tmpdir: Path) -> dict[str, Any]:
    with (tmpdir / "bm25.pkl").open("rb") as handle:
        bm25_payload = pickle.load(handle)
    return {
        "honeypot_df": pd.read_parquet(tmpdir / "honeypot_flags.parquet"),
        "features_df": pd.read_parquet(tmpdir / "features.parquet"),
        "faiss_index": faiss.read_index(str(tmpdir / "faiss.index")),
        "faiss_ids": np.load(str(tmpdir / "candidate_ids_faiss.npy"), allow_pickle=True),
        "bm25": bm25_payload["bm25"],
        "bm25_ids": np.load(str(tmpdir / "candidate_ids_bm25.npy"), allow_pickle=True),
    }


def _empty_submission() -> pd.DataFrame:
    return pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"])


def _rank_uploaded_candidates(
    raw_candidates: list[dict[str, Any]],
    jd_text: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any]]:
    timings: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}

    candidates = _timed(timings, "Normalize candidates", _normalize_candidates, raw_candidates)
    diagnostics["uploaded_candidates"] = len(raw_candidates)
    diagnostics["ranked_candidates_cap"] = len(candidates)

    with tempfile.TemporaryDirectory(prefix="ranker_sandbox_") as tmp:
        tmpdir = Path(tmp)
        candidates_path = tmpdir / "candidates.jsonl"
        jd_path = tmpdir / "jd.txt"
        honeypot_path = tmpdir / "honeypot_flags.parquet"
        features_path = tmpdir / "features.parquet"
        faiss_index_path = tmpdir / "faiss.index"
        faiss_ids_path = tmpdir / "candidate_ids_faiss.npy"
        bm25_path = tmpdir / "bm25.pkl"
        bm25_ids_path = tmpdir / "candidate_ids_bm25.npy"

        def write_inputs() -> None:
            _write_jsonl(candidates_path, candidates)
            jd_path.write_text(jd_text, encoding="utf-8")

        _timed(timings, "Write temp inputs", write_inputs)

        jd, jd_embedding = _timed(timings, "Parse and embed JD", _parse_and_embed_jd, jd_path)
        diagnostics["required_skills"] = jd.get("required_skills", [])

        honeypot_module = _load_precompute_module("01_honeypot_detector.py", "sandbox_honeypot_detector")
        feature_module = _load_precompute_module("02_feature_extractor.py", "sandbox_feature_extractor")
        embedder_module = _load_precompute_module("03_embedder.py", "sandbox_embedder")
        bm25_module = _load_precompute_module("04_build_bm25.py", "sandbox_bm25_builder")

        _timed(
            timings,
            "Honeypot detection",
            honeypot_module.detect_honeypots,
            str(candidates_path),
            str(honeypot_path),
            str(STATIC_DIR / "company_founding_years.json"),
            jd.get("required_skills", []),
        )
        _timed(
            timings,
            "Feature extraction",
            feature_module.extract_features,
            str(candidates_path),
            str(honeypot_path),
            str(STATIC_DIR),
            str(features_path),
        )

        features_snapshot = pd.read_parquet(features_path)
        if features_snapshot.empty:
            honeypot_snapshot = pd.read_parquet(honeypot_path)
            diagnostics["honeypot_flagged"] = (
                int(honeypot_snapshot["honeypot_score"].sum()) if not honeypot_snapshot.empty else 0
            )
            diagnostics["after_honeypot_filter"] = 0
            return _empty_submission(), features_snapshot, timings, diagnostics

        _timed(
            timings,
            "Embedding and FAISS",
            _build_embeddings_with_fallback,
            embedder_module,
            features_path,
            faiss_index_path,
            faiss_ids_path,
            tmpdir / "hf_minilm_marker",
        )
        _timed(
            timings,
            "BM25 build",
            bm25_module.build_bm25,
            str(features_path),
            str(bm25_path),
            str(bm25_ids_path),
        )

        artifacts = _timed(timings, "Load mini artifacts", _load_mini_artifacts, tmpdir)
        honeypot_df = artifacts["honeypot_df"]
        diagnostics["honeypot_flagged"] = int(honeypot_df["honeypot_score"].sum()) if not honeypot_df.empty else 0

        clean_df = _timed(
            timings,
            "Apply honeypot filter",
            rank.apply_honeypot_filter,
            artifacts["features_df"],
            honeypot_df,
        )
        diagnostics["after_honeypot_filter"] = len(clean_df)
        if clean_df.empty:
            return _empty_submission(), clean_df, timings, diagnostics

        top_bm25_ids = _timed(
            timings,
            "BM25 retrieve",
            rank.bm25_retrieve,
            jd,
            artifacts["bm25"],
            artifacts["bm25_ids"],
            clean_df,
            top_k=min(5000, len(clean_df)),
        )
        diagnostics["bm25_retrieved"] = len(top_bm25_ids)
        if not top_bm25_ids:
            return _empty_submission(), clean_df, timings, diagnostics

        semantic_results = _timed(
            timings,
            "Semantic rerank",
            rank.semantic_rerank,
            top_bm25_ids,
            jd_embedding,
            artifacts["faiss_index"],
            artifacts["faiss_ids"],
            top_k=min(500, len(top_bm25_ids)),
        )
        diagnostics["semantic_reranked"] = len(semantic_results)
        if not semantic_results:
            return _empty_submission(), clean_df, timings, diagnostics

        scored_df = _timed(
            timings,
            "Deep score",
            rank.deep_score,
            semantic_results,
            clean_df,
            jd,
            top_k=min(150, len(semantic_results)),
        )
        if scored_df.empty:
            return _empty_submission(), clean_df, timings, diagnostics

        scored_df = _timed(timings, "Top-10 precision pass", rank.top10_precision_pass, scored_df)
        scored_df = scored_df.head(100).reset_index(drop=True)
        scored_df = _timed(timings, "Build reasoning", rank.build_reasoning_column, scored_df, jd)

        final_df = scored_df.copy().reset_index(drop=True)
        final_df["rank"] = final_df.index + 1
        final_df["score"] = final_df["composite_score"].round(6)
        submission_df = final_df[["candidate_id", "rank", "score", "reasoning"]].copy()
        submission_df["candidate_id"] = submission_df["candidate_id"].astype(str)
        submission_df["rank"] = submission_df["rank"].astype(int)
        submission_df["score"] = submission_df["score"].astype(float)
        submission_df["reasoning"] = submission_df["reasoning"].astype(str)
        diagnostics["ranked_rows"] = len(submission_df)

        return submission_df, final_df, timings, diagnostics


st.title("Offline Candidate Ranker Sandbox")
st.markdown(
    '<p class="subtle">Upload up to 100 candidates and rank them with the same honeypot, feature, BM25, FAISS, scoring, and reasoning path used by the full pipeline.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Pipeline")
    st.caption(
        "MiniLM source: "
        + (str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_ID)
    )

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Upload candidates")
    uploaded_file = st.file_uploader(
        "Candidates JSONL or JSON",
        type=["jsonl", "json"],
    )
    st.caption("The sandbox ranks at most 100 uploaded candidates.")

    st.subheader("Job description")
    default_jd = (
        "We are looking for a Senior AI Engineer with 5-8 years of experience. "
        "Required skills: Python, embeddings, vector database, ranking evaluation, "
        "BM25, semantic search, learning to rank. You will build production ML "
        "systems and work with FAISS or similar vector stores."
    )
    jd_text = st.text_area("Job description", value=default_jd, height=260)
    run_btn = st.button(
        "Run ranking",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or not jd_text.strip(),
    )

with right:
    if not run_btn:
        st.info("Upload candidates, paste a JD, and run ranking.")
    elif uploaded_file is None:
        st.error("Upload a candidates file first.")
    elif not jd_text.strip():
        st.error("Paste a job description first.")
    else:
        try:
            raw_candidates = _parse_uploaded_candidates(uploaded_file)
            if not raw_candidates:
                st.error("No valid candidate objects found in the uploaded file.")
                st.stop()
            if len(raw_candidates) > 100:
                st.warning(f"{len(raw_candidates)} candidates uploaded. Ranking the first 100.")
                raw_candidates = raw_candidates[:100]

            with st.spinner("Running mini pipeline..."):
                submission_df, scored_df, timings, diagnostics = _rank_uploaded_candidates(raw_candidates, jd_text)

            st.success("Pipeline complete")

            metric_cols = st.columns(4)
            metric_cols[0].metric("Uploaded", diagnostics.get("uploaded_candidates", len(raw_candidates)))
            metric_cols[1].metric("Flagged", diagnostics.get("honeypot_flagged", 0))
            metric_cols[2].metric("Ranked", len(submission_df))
            metric_cols[3].metric("Total time", f"{sum(timings.values()):.2f}s")

            if diagnostics.get("required_skills"):
                st.caption("Required skills parsed: " + ", ".join(diagnostics["required_skills"][:12]))

            st.subheader("Stage timings")
            timing_df = pd.DataFrame(
                [{"stage": stage, "seconds": round(seconds, 3)} for stage, seconds in timings.items()]
            )
            st.dataframe(timing_df, hide_index=True, use_container_width=True)

            if submission_df.empty:
                st.warning("No candidates remained after filtering and retrieval.")
            else:
                st.subheader("Top candidates")
                preview_cols = [
                    col
                    for col in ["rank", "candidate_id", "full_name", "current_title", "score", "reasoning"]
                    if col in scored_df.columns
                ]
                st.dataframe(
                    scored_df.head(10)[preview_cols],
                    hide_index=True,
                    use_container_width=True,
                )

                csv_bytes = submission_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download submission.csv",
                    data=csv_bytes,
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                with st.expander("Submission CSV preview"):
                    st.dataframe(submission_df.head(100), hide_index=True, use_container_width=True)

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            import traceback

            st.code(traceback.format_exc())
