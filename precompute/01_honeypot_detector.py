"""
Honeypot Detector — Pre-computation Step 1
Detects fake/planted candidate profiles before any ranking happens.
Outputs a binary flag per candidate. Flagged candidates are excluded from ranking.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
STATIC_DIR = REPO_ROOT / "static"


def _safe_parse_date(val: Any) -> pd.Timestamp | None:
    """Parse a date string/value gracefully. Returns None on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        s = str(val).strip()
        if not s or s.lower() in ("null", "none", "nan", "present", "current"):
            return None
        return pd.Timestamp(dateutil_parser.parse(s, default=pd.Timestamp("2000-01-01")))
    except Exception:
        return None


def _get(d: dict, *keys, default=None) -> Any:
    """Safe nested dict access."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _load_candidates(path: str) -> list[dict]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {i}: JSON decode error — {e}")
    logger.info(f"Loaded {len(candidates)} candidates from {path}")
    return candidates


def _load_company_founding_years() -> dict:
    path = STATIC_DIR / "company_founding_years.json"
    if not path.exists():
        logger.warning("company_founding_years.json not found")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.lower(): v for k, v in data.get("companies", {}).items()}


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def _rule1_company_tenure_vs_founding(candidate: dict, founding_years: dict) -> bool:
    """Rule 1: Role start year < company founding year."""
    work_history = _get(candidate, "work_history") or _get(candidate, "experience") or []
    if not isinstance(work_history, list):
        return False
    for role in work_history:
        if not isinstance(role, dict):
            continue
        employer = str(_get(role, "company") or _get(role, "employer") or "").lower().strip()
        if not employer:
            continue
        founding_year = founding_years.get(employer)
        if founding_year is None:
            continue  # Unknown company — skip, do not flag
        start_raw = _get(role, "start_date") or _get(role, "start")
        start_ts = _safe_parse_date(start_raw)
        if start_ts is None:
            continue
        if start_ts.year < founding_year:
            return True
    return False


def _rule2_expert_skill_zero_years(candidate: dict) -> bool:
    """Rule 2: Expert skill with 0 or null years_used."""
    skills = _get(candidate, "skills") or []
    if not isinstance(skills, list):
        return False
    expert_terms = {"expert", "advanced", "proficient", "master", "5", "5/5"}
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        proficiency = str(_get(skill, "proficiency") or _get(skill, "level") or "").lower().strip()
        if proficiency not in expert_terms:
            continue
        years_used = _get(skill, "years_used") or _get(skill, "years")
        try:
            if years_used is None or float(years_used) == 0:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _rule3_job_before_graduation(candidate: dict) -> bool:
    """Rule 3: Any job started > 12 months before undergrad completion."""
    education = _get(candidate, "education") or []
    if not isinstance(education, list):
        return False

    undergrad_end: pd.Timestamp | None = None
    for edu in education:
        if not isinstance(edu, dict):
            continue
        degree = str(_get(edu, "degree") or "").lower()
        if any(t in degree for t in ("bachelor", "b.tech", "b.e", "b.sc", "b.com", "be ", "btech", "undergraduate", "ug")):
            end_raw = _get(edu, "end_date") or _get(edu, "graduation_year") or _get(edu, "end")
            end_ts = _safe_parse_date(end_raw)
            if end_ts:
                if undergrad_end is None or end_ts < undergrad_end:
                    undergrad_end = end_ts
    if undergrad_end is None:
        return False

    work_history = _get(candidate, "work_history") or _get(candidate, "experience") or []
    if not isinstance(work_history, list):
        return False
    threshold = undergrad_end - pd.DateOffset(months=12)
    for role in work_history:
        if not isinstance(role, dict):
            continue
        start_raw = _get(role, "start_date") or _get(role, "start")
        start_ts = _safe_parse_date(start_raw)
        if start_ts is None:
            continue
        if start_ts < threshold:
            return True
    return False


def _rule4_experience_vs_career_span(candidate: dict) -> bool:
    """Rule 4: self-reported total_years_experience > actual career span + 2."""
    self_reported = _get(candidate, "total_years_experience") or _get(candidate, "years_of_experience")
    if self_reported is None:
        return False
    try:
        self_reported = float(self_reported)
    except (ValueError, TypeError):
        return False

    work_history = _get(candidate, "work_history") or _get(candidate, "experience") or []
    if not isinstance(work_history, list) or not work_history:
        return False

    earliest: pd.Timestamp | None = None
    for role in work_history:
        if not isinstance(role, dict):
            continue
        start_raw = _get(role, "start_date") or _get(role, "start")
        start_ts = _safe_parse_date(start_raw)
        if start_ts and (earliest is None or start_ts < earliest):
            earliest = start_ts
    if earliest is None:
        return False

    today = pd.Timestamp.today()
    actual_span_years = (today - earliest).days / 365.25
    return self_reported > actual_span_years + 2


def _rule5_implausible_skill_breadth(candidate: dict) -> bool:
    """Rule 5: >15 skills but <3 years computed experience."""
    skills = _get(candidate, "skills") or []
    if not isinstance(skills, list):
        return False
    skill_count = len(skills)
    if skill_count <= 15:
        return False

    work_history = _get(candidate, "work_history") or _get(candidate, "experience") or []
    if not isinstance(work_history, list) or not work_history:
        return False

    earliest: pd.Timestamp | None = None
    for role in work_history:
        if not isinstance(role, dict):
            continue
        start_raw = _get(role, "start_date") or _get(role, "start")
        start_ts = _safe_parse_date(start_raw)
        if start_ts and (earliest is None or start_ts < earliest):
            earliest = start_ts
    if earliest is None:
        return False

    today = pd.Timestamp.today()
    actual_span_years = (today - earliest).days / 365.25
    return actual_span_years < 3


def _rule6_duplicate_profile(candidate: dict, seen_hashes: dict) -> bool:
    """Rule 6: Hash of name+employer+title appears more than once."""
    name = str(_get(candidate, "full_name") or _get(candidate, "name") or "").lower().strip()
    employer = str(_get(candidate, "current_employer") or _get(candidate, "current_company") or "").lower().strip()
    title = str(_get(candidate, "current_title") or _get(candidate, "title") or "").lower().strip()
    if not name and not employer and not title:
        return False
    combo = f"{name}|{employer}|{title}"
    h = hashlib.md5(combo.encode("utf-8")).hexdigest()
    cid = str(_get(candidate, "candidate_id") or _get(candidate, "id") or "")
    if h in seen_hashes:
        return True  # Duplicate — flag this one
    seen_hashes[h] = cid
    return False


def _compute_bm25_seed_scores(candidates: list[dict], jd_required_skills: list[str] | None = None) -> list[float]:
    """
    Rule 7 helper: Compute TF-IDF score for each candidate against the JD required skills.
    Returns a list of scores in the same order as candidates.
    If no JD is available, uses generic skill terms.
    """
    query_terms = jd_required_skills or ["python", "machine learning", "embeddings", "ranking"]
    query_text = " ".join(query_terms)

    def _candidate_text(c: dict) -> str:
        parts = []
        title = _get(c, "current_title") or _get(c, "title") or ""
        parts.append(str(title))
        skills = _get(c, "skills") or []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict):
                    parts.append(str(_get(s, "name") or _get(s, "skill") or ""))
                else:
                    parts.append(str(s))
        summary = _get(c, "summary") or _get(c, "bio") or ""
        parts.append(str(summary))
        return " ".join(parts)

    corpus = [_candidate_text(c) for c in candidates]
    all_texts = [query_text] + corpus

    try:
        vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = tfidf_matrix[0]
        candidate_vecs = tfidf_matrix[1:]
        scores = (candidate_vecs @ query_vec.T).toarray().flatten().tolist()
    except Exception as e:
        logger.warning(f"TF-IDF computation failed: {e}")
        scores = [0.0] * len(candidates)
    return scores


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def detect_honeypots(
    candidates_path: str,
    output_path: str,
    company_founding_years_path: str,
    jd_required_skills: list[str] | None = None,
) -> None:
    """
    Detect honeypot/fake candidate profiles. Writes artifacts/honeypot_flags.parquet.

    Args:
        candidates_path: Path to candidates.jsonl
        output_path: Path to write honeypot_flags.parquet
        company_founding_years_path: Path to company_founding_years.json
        jd_required_skills: Optional list of JD required skills for BM25 seed scoring
    """
    candidates = _load_candidates(candidates_path)

    # Load company founding years
    founding_years: dict = {}
    fyp = Path(company_founding_years_path)
    if fyp.exists():
        with open(fyp, "r", encoding="utf-8") as f:
            data = json.load(f)
        founding_years = {k.lower(): v for k, v in data.get("companies", {}).items()}
    else:
        logger.warning(f"Company founding years not found at {company_founding_years_path}")

    logger.info("Computing BM25 seed scores (TF-IDF)...")
    bm25_scores = _compute_bm25_seed_scores(candidates, jd_required_skills)

    seen_hashes: dict[str, str] = {}
    records = []

    logger.info("Running honeypot detection rules...")
    for i, candidate in enumerate(tqdm(candidates, desc="Honeypot detection")):
        cid = str(_get(candidate, "candidate_id") or _get(candidate, "id") or f"unknown_{i}")
        fired_rules: list[str] = []

        try:
            if _rule1_company_tenure_vs_founding(candidate, founding_years):
                fired_rules.append("rule1_company_tenure_before_founding")
        except Exception as e:
            logger.debug(f"Rule1 error for {cid}: {e}")

        try:
            if _rule2_expert_skill_zero_years(candidate):
                fired_rules.append("rule2_expert_skill_zero_years")
        except Exception as e:
            logger.debug(f"Rule2 error for {cid}: {e}")

        try:
            if _rule3_job_before_graduation(candidate):
                fired_rules.append("rule3_job_before_graduation")
        except Exception as e:
            logger.debug(f"Rule3 error for {cid}: {e}")

        try:
            if _rule4_experience_vs_career_span(candidate):
                fired_rules.append("rule4_experience_overstated")
        except Exception as e:
            logger.debug(f"Rule4 error for {cid}: {e}")

        try:
            if _rule5_implausible_skill_breadth(candidate):
                fired_rules.append("rule5_implausible_skill_breadth")
        except Exception as e:
            logger.debug(f"Rule5 error for {cid}: {e}")

        try:
            if _rule6_duplicate_profile(candidate, seen_hashes):
                fired_rules.append("rule6_duplicate_profile")
        except Exception as e:
            logger.debug(f"Rule6 error for {cid}: {e}")

        honeypot_score = 1 if fired_rules else 0
        records.append({
            "candidate_id": cid,
            "honeypot_score": honeypot_score,
            "honeypot_reasons": fired_rules,
            "bm25_seed_score": float(bm25_scores[i]),
        })

    df = pd.DataFrame(records)
    df["candidate_id"] = df["candidate_id"].astype(str)
    df["honeypot_score"] = df["honeypot_score"].astype(int)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    flagged = df["honeypot_score"].sum()
    logger.info(f"Honeypot detection complete: {flagged}/{len(df)} candidates flagged -> {output_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", default="artifacts/honeypot_flags.parquet")
    ap.add_argument("--founding-years", default="static/company_founding_years.json")
    args = ap.parse_args()
    detect_honeypots(args.candidates, args.output, args.founding_years)
