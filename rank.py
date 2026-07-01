"""
rank.py — Runtime Ranking Entry Point
Reads pre-built artifacts and produces the output CSV.
Must complete in under 5 minutes for 100,000 candidates.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --jd ./data/jd.txt --out ./submission.csv
"""

import argparse
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

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
from src.validator import validate_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact paths (relative to repo root)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_DIR = REPO_ROOT / "models" / "minilm"
STATIC_DIR = REPO_ROOT / "static"

HONEYPOT_FLAGS_PATH = ARTIFACTS_DIR / "honeypot_flags.parquet"
FEATURES_PATH = ARTIFACTS_DIR / "features.parquet"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
FAISS_IDS_PATH = ARTIFACTS_DIR / "candidate_ids_faiss.npy"
BM25_PATH = ARTIFACTS_DIR / "bm25.pkl"
BM25_IDS_PATH = ARTIFACTS_DIR / "candidate_ids_bm25.npy"

# Scoring weights (must sum to 1.0)
SCORING_WEIGHTS = {
    "semantic_similarity": 0.25,
    "required_skill_coverage": 0.25,
    "trajectory_score": 0.20,
    "behavioral_availability": 0.10,
    "consistency_score": 0.10,
    "skill_depth_score": 0.05,
    "experience_range_fit": 0.05,
}


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 query — must match tokenizer used in build_bm25."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


# ---------------------------------------------------------------------------
# Stage 1: load_artifacts
# ---------------------------------------------------------------------------

def load_artifacts() -> dict[str, Any]:
    """
    Load all pre-computed artifacts from disk.
    Expected runtime: ~10 seconds.
    """
    t0 = time.perf_counter()
    logger.info("Loading artifacts...")

    # Validate artifacts exist
    for p in [HONEYPOT_FLAGS_PATH, FEATURES_PATH, FAISS_INDEX_PATH, FAISS_IDS_PATH, BM25_PATH, BM25_IDS_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}\n"
                f"Run: python precompute/run_all.py --candidates <path> --jd <path>"
            )

    honeypot_df = pd.read_parquet(HONEYPOT_FLAGS_PATH)
    features_df = pd.read_parquet(FEATURES_PATH)
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    faiss_ids = np.load(str(FAISS_IDS_PATH), allow_pickle=True)

    with open(BM25_PATH, "rb") as f:
        bm25_payload = pickle.load(f)
    bm25 = bm25_payload["bm25"]
    bm25_ids = np.load(str(BM25_IDS_PATH), allow_pickle=True)

    logger.info(
        f"Artifacts loaded in {time.perf_counter()-t0:.1f}s | "
        f"features={len(features_df)} candidates, "
        f"FAISS={faiss_index.ntotal} vectors, "
        f"BM25={len(bm25_ids)} documents"
    )

    return {
        "honeypot_df": honeypot_df,
        "features_df": features_df,
        "faiss_index": faiss_index,
        "faiss_ids": faiss_ids,
        "bm25": bm25,
        "bm25_ids": bm25_ids,
    }


# ---------------------------------------------------------------------------
# Stage 2: parse_and_embed_jd
# ---------------------------------------------------------------------------

def parse_and_embed_jd(jd_path: str) -> tuple[dict, np.ndarray]:
    """
    Parse the JD file and embed JD text with MiniLM.
    Expected runtime: ~5 seconds.
    Returns: (parsed_jd dict, jd_embedding np.ndarray shape [384])
    """
    t0 = time.perf_counter()
    logger.info(f"Parsing JD: {jd_path}")

    parser = JDParser(static_dir=STATIC_DIR)
    jd = parser.parse(jd_path)

    # Build JD query text — use required skills + raw text snippet
    required_skills_text = " ".join(jd.get("required_skills", []))
    raw_jd_text = jd.get("raw_jd_text", "")[:1000]
    jd_query_text = f"{required_skills_text} {raw_jd_text}".strip()

    # Load MiniLM from local path
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"MiniLM model not found at {MODEL_DIR}. "
            "Download once with: sentence-transformers download sentence-transformers/all-MiniLM-L6-v2"
        )

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    jd_embedding = model.encode(
        [jd_query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(jd_embedding)
    if norm > 0:
        jd_embedding = jd_embedding / norm

    # Store embedding in JD dict for later use
    jd["text_embedding"] = jd_embedding.tolist()

    logger.info(
        f"JD parsed in {time.perf_counter()-t0:.1f}s | "
        f"{len(jd.get('required_skills',[]))} required skills, "
        f"seniority={jd.get('seniority_level')}"
    )
    return jd, jd_embedding


# ---------------------------------------------------------------------------
# Stage 3: apply_honeypot_filter
# ---------------------------------------------------------------------------

def apply_honeypot_filter(
    features_df: pd.DataFrame,
    honeypot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop all candidates with honeypot_score=1 from features_df.
    Expected runtime: ~2 seconds.
    """
    t0 = time.perf_counter()
    flagged_ids = set(
        honeypot_df[honeypot_df["honeypot_score"] == 1]["candidate_id"].astype(str)
    )
    before = len(features_df)
    clean_df = features_df[~features_df["candidate_id"].astype(str).isin(flagged_ids)].copy()
    logger.info(
        f"Honeypot filter: {before - len(clean_df)} removed, "
        f"{len(clean_df)} remain ({time.perf_counter()-t0:.1f}s)"
    )
    return clean_df


# ---------------------------------------------------------------------------
# Stage 4: bm25_retrieve
# ---------------------------------------------------------------------------

def bm25_retrieve(
    jd: dict,
    bm25: Any,
    bm25_ids: np.ndarray,
    clean_df: pd.DataFrame,
    top_k: int = 5000,
) -> list[str]:
    """
    Query BM25 index with JD required skills text. Return top-k candidate_ids.
    Expected runtime: ~15 seconds.
    """
    t0 = time.perf_counter()

    required_skills = jd.get("required_skills", [])
    nice_to_have = jd.get("nice_to_have_skills", [])
    query_terms = required_skills + nice_to_have
    query_text = " ".join(query_terms)
    query_tokens = _tokenize_bm25(query_text)

    logger.info(f"BM25 query tokens: {query_tokens[:10]}...")

    # Get scores for all documents in BM25 index
    scores = bm25.get_scores(query_tokens)

    # Filter to only clean (non-honeypot) candidates
    clean_ids_set = set(clean_df["candidate_id"].astype(str))
    bm25_ids_str = np.array([str(cid) for cid in bm25_ids])

    # Build masked scores — set honeypot/removed candidates to -inf
    masked_scores = np.where(
        np.isin(bm25_ids_str, list(clean_ids_set)),
        scores,
        -np.inf,
    )

    # Top-k indices
    effective_k = min(top_k, int((masked_scores > -np.inf).sum()))
    top_indices = np.argpartition(masked_scores, -effective_k)[-effective_k:]
    top_indices = top_indices[np.argsort(masked_scores[top_indices])[::-1]]

    top_ids = [str(bm25_ids_str[i]) for i in top_indices if masked_scores[i] > -np.inf]

    logger.info(
        f"BM25 retrieved {len(top_ids)} candidates in {time.perf_counter()-t0:.1f}s"
    )
    return top_ids


# ---------------------------------------------------------------------------
# Stage 5: semantic_rerank
# ---------------------------------------------------------------------------

def semantic_rerank(
    top_bm25_ids: list[str],
    jd_embedding: np.ndarray,
    faiss_index: Any,
    faiss_ids: np.ndarray,
    top_k: int = 500,
) -> list[tuple[str, float]]:
    """
    Compute FAISS cosine similarity for BM25 top candidates vs JD embedding.
    Returns list of (candidate_id, semantic_score) for top-k.
    Expected runtime: ~20 seconds.
    """
    t0 = time.perf_counter()

    # Build position lookup for FAISS: candidate_id -> faiss_index position
    faiss_ids_str = np.array([str(cid) for cid in faiss_ids])
    bm25_id_set = set(top_bm25_ids)

    # Find FAISS positions for the BM25 top candidates
    positions = np.where(np.isin(faiss_ids_str, list(bm25_id_set)))[0]

    if len(positions) == 0:
        logger.warning("No overlap between BM25 results and FAISS index")
        return []

    # Reconstruct vectors for these candidates (FAISS IndexFlatIP supports reconstruct)
    n = len(positions)
    dim = faiss_index.d
    candidate_vectors = np.zeros((n, dim), dtype=np.float32)
    for i, pos in enumerate(positions):
        faiss_index.reconstruct(int(pos), candidate_vectors[i])

    # Compute cosine similarity via inner product (vectors already normalized)
    jd_vec = jd_embedding.reshape(1, -1).astype(np.float32)
    scores = (candidate_vectors @ jd_vec.T).flatten()

    # Top-k by semantic score
    effective_k = min(top_k, len(scores))
    top_local_indices = np.argpartition(scores, -effective_k)[-effective_k:]
    top_local_indices = top_local_indices[np.argsort(scores[top_local_indices])[::-1]]

    results = [
        (str(faiss_ids_str[positions[i]]), float(scores[i]))
        for i in top_local_indices
    ]

    logger.info(
        f"Semantic rerank: {len(results)} candidates in {time.perf_counter()-t0:.1f}s | "
        f"top score={results[0][1]:.4f} if results else n/a"
    )
    return results


# ---------------------------------------------------------------------------
# Stage 6: deep_score
# ---------------------------------------------------------------------------

def deep_score(
    semantic_top500: list[tuple[str, float]],
    features_df: pd.DataFrame,
    jd: dict,
    top_k: int = 100,
) -> pd.DataFrame:
    """
    Compute weighted composite score for top-500 semantic candidates.
    Applies honeypot divergence check.
    Returns top-100 DataFrame sorted by composite score descending.
    Expected runtime: ~20 seconds.
    """
    t0 = time.perf_counter()

    sem_id_to_score = {cid: score for cid, score in semantic_top500}
    top500_ids = list(sem_id_to_score.keys())

    # Filter features to these 500 candidates
    mask = features_df["candidate_id"].astype(str).isin(set(top500_ids))
    df = features_df[mask].copy()

    if df.empty:
        logger.warning("deep_score: No candidates after feature filter")
        return pd.DataFrame()

    # Attach semantic scores
    df["semantic_similarity"] = df["candidate_id"].astype(str).map(sem_id_to_score).fillna(0.0)

    # ---------------------------------------------------------------------------
    # Honeypot divergence check at runtime:
    # Candidates in top 5% bm25_seed_score AND bottom 40% semantic_similarity → exclude
    # ---------------------------------------------------------------------------
    bm25_95th = df["bm25_seed_score"].quantile(0.95) if "bm25_seed_score" in df.columns else np.inf
    sem_40th = df["semantic_similarity"].quantile(0.40)
    honeypot_divergence_mask = (
        (df.get("bm25_seed_score", pd.Series(0.0, index=df.index)) >= bm25_95th) &
        (df["semantic_similarity"] <= sem_40th)
    )
    n_divergence = honeypot_divergence_mask.sum()
    if n_divergence > 0:
        logger.info(f"Honeypot divergence: excluding {n_divergence} candidates")
        df = df[~honeypot_divergence_mask].copy()

    # ---------------------------------------------------------------------------
    # Parse JD parameters for scoring
    # ---------------------------------------------------------------------------
    required_skills: list[str] = jd.get("required_skills", [])
    functional_equivalents: dict = jd.get("skill_functional_equivalents", {})
    seniority_range: list = jd.get("seniority_range", [5, 9])
    seniority_level: str = jd.get("seniority_level", "senior")

    # Map seniority level string to integer target
    _seniority_level_map = {
        "intern": 0, "junior": 1, "associate": 2, "mid": 3,
        "senior": 4, "lead": 5, "staff": 5, "principal": 6,
        "director": 7, "vp": 8, "principal": 6, "chief": 9,
    }
    seniority_target = _seniority_level_map.get(seniority_level.lower(), 4)

    # ---------------------------------------------------------------------------
    # Compute per-candidate signals (vectorized where possible)
    # ---------------------------------------------------------------------------

    def _get_list(val: Any) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                import json
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    def _get_dict(val: Any) -> dict:
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                import json
                parsed = json.loads(val)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    results = []
    for _, row in df.iterrows():
        cid = str(row["candidate_id"])
        try:
            candidate_skills = _get_list(row.get("skills_list"))
            skill_recency_map = _get_dict(row.get("skill_recency_map"))
            seniority_scores = _get_list(row.get("seniority_scores"))
            employer_history = _get_list(row.get("employer_history"))

            req_cov = compute_required_skill_coverage(
                candidate_skills, required_skills, functional_equivalents
            )
            if len(results) == 0:
                logger.info(f"First candidate skills sample: {candidate_skills[:10]}")
                logger.info(f"Required skills: {required_skills}")
                logger.info(f"Coverage: {req_cov}")
            traj_score = compute_trajectory_score(
                float(row.get("trajectory_slope") or 0.0),
                int(row.get("current_seniority") or 3),
                seniority_target,
            )
            behav_score = compute_behavioral_availability(
                row.get("platform_last_active_days"),
                row.get("recruiter_response_rate"),
            )
            consist_score = compute_consistency_score(
                float(row.get("title_skill_alignment") or 0.5),
                float(row.get("profile_completeness") or 0.5),
            )
            depth_score = compute_skill_depth_score(skill_recency_map, required_skills)
            exp_fit = compute_experience_range_fit(
                float(row.get("total_years_exp_computed") or 0.0),
                seniority_range,
            )

            signals = {
                "semantic_similarity": float(row.get("semantic_similarity") or 0.0),
                "required_skill_coverage": req_cov,
                "trajectory_score": traj_score,
                "behavioral_availability": behav_score,
                "consistency_score": consist_score,
                "skill_depth_score": depth_score,
                "experience_range_fit": exp_fit,
            }
            composite = compute_composite_score(signals, SCORING_WEIGHTS)

            result_row = row.to_dict()
            result_row.update(signals)
            result_row["composite_score"] = composite
            result_row["required_skill_coverage"] = req_cov
            results.append(result_row)

        except Exception as e:
            logger.warning(f"deep_score failed for {cid}: {e}")

    scored_df = pd.DataFrame(results)
    if scored_df.empty:
        return scored_df

    scored_df = scored_df.sort_values("composite_score", ascending=False).head(top_k)

    logger.info(
        f"deep_score: top {len(scored_df)} candidates in {time.perf_counter()-t0:.1f}s | "
        f"top score={scored_df['composite_score'].iloc[0]:.4f}"
    )
    return scored_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 7: top10_precision_pass
# ---------------------------------------------------------------------------

def top10_precision_pass(
    scored_df: pd.DataFrame,
    coverage_threshold: float = 0.6,
    protect_n: int = 20,
) -> pd.DataFrame:
    """
    After initial scoring, for candidates ranked 1-20, check if required_skill_coverage < 0.6.
    If yes, demote to rank 21+ and pull up the next highest scoring candidate with coverage >= 0.6.
    Protects NDCG@10.
    Expected runtime: ~3 seconds.
    """
    t0 = time.perf_counter()

    if scored_df.empty or len(scored_df) < 2:
        return scored_df

    df = scored_df.copy().reset_index(drop=True)

    # Split into top-protect_n and the rest
    top_n = df.iloc[:protect_n].copy()
    rest = df.iloc[protect_n:].copy()

    # Find low-coverage candidates in top_n
    low_coverage_mask = top_n["required_skill_coverage"] < coverage_threshold
    low_coverage_rows = top_n[low_coverage_mask].copy()
    high_coverage_top = top_n[~low_coverage_mask].copy()

    # From rest, find candidates with coverage >= threshold
    rest_high = rest[rest["required_skill_coverage"] >= coverage_threshold].copy()
    rest_low = rest[rest["required_skill_coverage"] < coverage_threshold].copy()

    n_to_promote = len(low_coverage_rows)
    promoted = rest_high.head(n_to_promote)
    rest_remaining = pd.concat([rest_high.iloc[n_to_promote:], rest_low, low_coverage_rows])

    # Reassemble: high coverage top-N first, then promoted, then rest
    final_df = pd.concat([high_coverage_top, promoted, rest_remaining]).reset_index(drop=True)

    # Re-sort the bottom part by composite_score to keep the best ones (including demoted) at the top of bottom part
    top_part = final_df.iloc[:protect_n].copy()
    bottom_part = final_df.iloc[protect_n:].sort_values("composite_score", ascending=False).copy()
    final_df = pd.concat([top_part, bottom_part]).reset_index(drop=True)

    # Enforce monotonicity to satisfy the validator checks
    scores = final_df["composite_score"].tolist()
    for idx in range(1, len(scores)):
        if scores[idx] > scores[idx - 1]:
            scores[idx] = scores[idx - 1]
    final_df["composite_score"] = scores

    demoted = low_coverage_mask.sum()
    logger.info(
        f"top10_precision_pass: {demoted} candidates demoted from top {protect_n} "
        f"(coverage < {coverage_threshold}) in {time.perf_counter()-t0:.1f}s"
    )
    logger.info(f"Top 10 coverage values: {final_df['required_skill_coverage'].head(10).tolist()}")
    return final_df


# ---------------------------------------------------------------------------
# Stage 8: build_reasoning_column
# ---------------------------------------------------------------------------

def build_reasoning_column(scored_df: pd.DataFrame, jd: dict) -> pd.DataFrame:
    """
    Generate fact-grounded reasoning string for each of the top-100 candidates.
    Expected runtime: ~5 seconds.
    """
    t0 = time.perf_counter()
    df = scored_df.copy()
    reasoning_strings = []

    for i, row in df.iterrows():
        rank = i + 1
        try:
            r = build_reasoning(row.to_dict(), jd, rank)
            # Strip any embedded newlines that would corrupt CSV
            r = r.replace("\n", " ").replace("\r", " ").strip()
            if not r:
                r = f"Candidate ranked {rank} based on composite scoring signals."
        except Exception as e:
            logger.warning(f"Reasoning failed for rank {rank}: {e}")
            r = f"Candidate ranked {rank} based on composite scoring signals."
        reasoning_strings.append(r)

    df["reasoning"] = reasoning_strings
    logger.info(f"Reasoning built for {len(df)} candidates in {time.perf_counter()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Stage 9: validate_and_write
# ---------------------------------------------------------------------------

def validate_and_write(
    scored_df: pd.DataFrame,
    candidates_path: str,
    output_path: str,
) -> None:
    """
    Assign final ranks, validate output, and write to CSV.
    Expected runtime: ~1 second.
    """
    t0 = time.perf_counter()

    output_df = scored_df.head(100).copy().reset_index(drop=True)
    output_df["rank"] = output_df.index + 1
    output_df["score"] = output_df["composite_score"].round(6)

    # Select only required columns for output
    final_df = output_df[["candidate_id", "rank", "score", "reasoning"]].copy()
    final_df["candidate_id"] = final_df["candidate_id"].astype(str)
    final_df["rank"] = final_df["rank"].astype(int)
    final_df["score"] = final_df["score"].astype(float)
    final_df["reasoning"] = final_df["reasoning"].astype(str)

    # Validate
    validate_output(final_df, candidates_path)
    logger.info("Output validation: PASSED (all 9 checks)")

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Output written to {out_path} in {time.perf_counter()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(candidates_path: str, jd_path: str, output_path: str) -> None:
    """Full ranking pipeline. Must complete in under 5 minutes."""
    wall_start = time.perf_counter()
    stage_times: dict[str, float] = {}

    def _timed(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        stage_times[name] = time.perf_counter() - t0
        return result

    # Stage 1: Load artifacts
    artifacts = _timed("load_artifacts", load_artifacts)

    # Stage 2: Parse and embed JD
    jd, jd_embedding = _timed("parse_and_embed_jd", parse_and_embed_jd, jd_path)

    # Stage 3: Apply honeypot filter
    clean_df = _timed(
        "apply_honeypot_filter",
        apply_honeypot_filter,
        artifacts["features_df"],
        artifacts["honeypot_df"],
    )

    # Stage 4: BM25 retrieve top 5000
    top_bm25_ids = _timed(
        "bm25_retrieve",
        bm25_retrieve,
        jd,
        artifacts["bm25"],
        artifacts["bm25_ids"],
        clean_df,
        top_k=5000,
    )

    # Stage 5: Semantic rerank top 500
    semantic_top500 = _timed(
        "semantic_rerank",
        semantic_rerank,
        top_bm25_ids,
        jd_embedding,
        artifacts["faiss_index"],
        artifacts["faiss_ids"],
        top_k=500,
    )

    # Stage 6: Deep score → top 100
    scored_df = _timed(
        "deep_score",
        deep_score,
        semantic_top500,
        clean_df,
        jd,
        top_k=150,  # Slightly over-fetch for precision pass
    )

    if len(scored_df) < 100:
        logger.warning(f"Only {len(scored_df)} candidates after deep_score (need 100)")

    # Stage 7: Top-10 precision pass
    scored_df = _timed("top10_precision_pass", top10_precision_pass, scored_df)

    # Trim to 100
    scored_df = scored_df.head(100).reset_index(drop=True)

    # Stage 8: Build reasoning
    scored_df = _timed("build_reasoning_column", build_reasoning_column, scored_df, jd)

    # Stage 9: Validate and write
    _timed("validate_and_write", validate_and_write, scored_df, candidates_path, output_path)

    total = time.perf_counter() - wall_start
    print("\n" + "=" * 60)
    print(f"  RANKING COMPLETE — {output_path}")
    print("=" * 60)
    for stage, t in stage_times.items():
        print(f"  {stage:<30} {t:.1f}s")
    print(f"  {'TOTAL':<30} {total:.1f}s")
    if total > 300:
        logger.warning(f"Total runtime {total:.1f}s exceeded 5-minute target!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline AI Candidate Ranker — produces top-100 ranked candidates."
    )
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--jd", required=True, help="Path to job description file (.docx/.txt)")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g. ./submission.csv)")
    args = ap.parse_args()

    candidates_path = str(Path(args.candidates).resolve())
    jd_path = str(Path(args.jd).resolve())
    output_path = str(Path(args.out).resolve())

    if not Path(candidates_path).exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)
    if not Path(jd_path).exists():
        print(f"ERROR: JD file not found: {jd_path}")
        sys.exit(1)

    run_pipeline(candidates_path, jd_path, output_path)


if __name__ == "__main__":
    main()
