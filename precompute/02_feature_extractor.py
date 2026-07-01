"""
Feature Extractor — Pre-computation Step 2
Computes a rich feature vector for every non-flagged candidate.
Everything in runtime scoring reads from this output.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
STATIC_DIR = REPO_ROOT / "static"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, *keys, default=None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _safe_parse_date(val: Any) -> pd.Timestamp | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        s = str(val).strip()
        if not s or s.lower() in ("null", "none", "nan", "present", "current"):
            return None
        return pd.Timestamp(dateutil_parser.parse(s, default=pd.Timestamp("2000-01-01")))
    except Exception:
        return None


def _is_present(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str) and val.strip().lower() in ("null", "none", "nan", ""):
        return False
    return True


def _load_json(filename: str) -> dict | list | None:
    path = STATIC_DIR / filename
    if not path.exists():
        logger.warning(f"{filename} not found in static/")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return None


def _load_skill_aliases() -> dict:
    raw = _load_json("skill_aliases.json")
    if raw and "aliases" in raw:
        return {k.lower(): v.lower() for k, v in raw["aliases"].items()}
    return {}


def _load_seniority_map() -> dict:
    # Prefer seniority_map.json, fall back to title_hierarchy.json
    raw = _load_json("seniority_map.json")
    if raw and "seniority_levels" in raw:
        return {k.lower(): int(v) for k, v in raw["seniority_levels"].items()}
    raw2 = _load_json("title_hierarchy.json")
    if raw2 and "seniority_scores" in raw2:
        return {k.lower(): int(round(float(v))) for k, v in raw2["seniority_scores"].items()}
    logger.warning("No seniority map found")
    return {}


def _load_consulting_firms() -> list:
    raw = _load_json("consulting_firms.json")
    if raw and "consulting_companies" in raw:
        return [c.lower() for c in raw["consulting_companies"]]
    return []


# ---------------------------------------------------------------------------
# Skill normalization
# ---------------------------------------------------------------------------

def _normalize_skills(skills_raw: Any, aliases: dict) -> list:
    """
    Normalize raw skills field (list of str or list of dict) to lowercase canonical list.
    """
    normalized = []
    if not isinstance(skills_raw, list):
        return normalized
    for item in skills_raw:
        if isinstance(item, dict):
            name = str(_get(item, "name") or _get(item, "skill") or "").lower().strip()
        else:
            name = str(item).lower().strip()
        name = name.rstrip(".,;:-")
        if not name:
            continue
        # Apply alias
        canonical = aliases.get(name, name)
        normalized.append(canonical)
    return list(set(normalized))  # deduplicate


# ---------------------------------------------------------------------------
# Experience features
# ---------------------------------------------------------------------------

def _parse_work_history(candidate: dict) -> list[dict]:
    wh = _get(candidate, "career_history") or _get(candidate, "work_history") or _get(candidate, "experience") or []
    if not isinstance(wh, list):
        return []
    roles = []
    for role in wh:
        if not isinstance(role, dict):
            continue
        start_raw = _get(role, "start_date") or _get(role, "start")
        end_raw = _get(role, "end_date") or _get(role, "end")
        start_ts = _safe_parse_date(start_raw)
        end_str = str(end_raw or "").lower().strip()
        if end_str in ("", "null", "none", "present", "current", "now"):
            end_ts = pd.Timestamp.today()
        else:
            end_ts = _safe_parse_date(end_raw) or pd.Timestamp.today()
        if start_ts is None:
            continue
        title = str(_get(role, "title") or _get(role, "position") or "").lower()
        company = str(_get(role, "company") or _get(role, "employer") or "").lower()
        description = str(_get(role, "description") or _get(role, "summary") or "")
        roles.append({
            "title": title,
            "company": company,
            "start": start_ts,
            "end": end_ts,
            "description": description,
        })
    # Sort chronologically
    roles.sort(key=lambda r: r["start"])
    return roles


def _compute_experience_features(roles: list[dict]) -> dict:
    today = pd.Timestamp.today()
    if not roles:
        return {
            "total_years_exp_computed": 0.0,
            "career_span_years": 0.0,
            "num_roles": 0,
            "avg_tenure_months": 0.0,
            "most_recent_role_tenure_months": 0.0,
        }
    earliest = roles[0]["start"]
    career_span = (today - earliest).days / 365.25

    # Sum non-overlapping tenure
    total_months = 0.0
    for role in roles:
        duration = (role["end"] - role["start"]).days / 30.44
        total_months += max(0.0, duration)
    total_years = total_months / 12.0

    tenures = [(r["end"] - r["start"]).days / 30.44 for r in roles]
    avg_tenure = float(np.mean(tenures)) if tenures else 0.0
    most_recent = tenures[-1] if tenures else 0.0

    return {
        "total_years_exp_computed": round(total_years, 2),
        "career_span_years": round(career_span, 2),
        "num_roles": len(roles),
        "avg_tenure_months": round(avg_tenure, 2),
        "most_recent_role_tenure_months": round(most_recent, 2),
    }


# ---------------------------------------------------------------------------
# Trajectory features
# ---------------------------------------------------------------------------

def _title_to_seniority(title: str, seniority_map: dict) -> int:
    title = title.lower()
    best_score = seniority_map.get("engineer", 3)  # default
    best_match_len = 0
    for keyword, score in seniority_map.items():
        # Multi-word keywords take priority
        if keyword in title and len(keyword) > best_match_len:
            best_score = score
            best_match_len = len(keyword)
    return best_score


def _compute_trajectory_features(roles: list[dict], seniority_map: dict) -> dict:
    if not roles:
        return {
            "seniority_scores": [],
            "trajectory_slope": 0.0,
            "current_seniority": 3,
            "max_seniority_reached": 3,
            "trajectory_direction": "lateral",
        }
    scores = [_title_to_seniority(r["title"], seniority_map) for r in roles]
    current = scores[-1]
    max_seniority = max(scores)

    if len(scores) >= 2:
        xs = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(xs, scores, 1)[0])
    else:
        slope = 0.0

    if slope > 0.1:
        direction = "ascending"
    elif slope < -0.1:
        direction = "descending"
    else:
        direction = "lateral"

    return {
        "seniority_scores": scores,
        "trajectory_slope": round(slope, 4),
        "current_seniority": current,
        "max_seniority_reached": max_seniority,
        "trajectory_direction": direction,
    }


# ---------------------------------------------------------------------------
# Skill recency map
# ---------------------------------------------------------------------------

def _compute_skill_recency(roles: list[dict], skills_list: list) -> dict:
    """
    For each skill, find the most recent year it appeared in a role's description or title.
    If not found in any description, use the most recent role's end year.
    """
    skill_recency: dict = {}
    for skill in skills_list:
        latest_year = None
        for role in roles:
            combined = f"{role['title']} {role['description']}".lower()
            if skill.lower() in combined:
                year = role["end"].year
                if latest_year is None or year > latest_year:
                    latest_year = year
        if latest_year is None and roles:
            latest_year = roles[-1]["end"].year
        if latest_year is not None:
            skill_recency[skill] = latest_year
    return skill_recency


# ---------------------------------------------------------------------------
# Company features
# ---------------------------------------------------------------------------

def _compute_company_features(
    roles: list[dict],
    consulting_firms: list,
) -> dict:
    employer_history = [r["company"] for r in roles if r["company"]]
    has_product = False
    is_consulting_only = True
    has_startup = False

    startup_patterns = [
        r"startup", r"early.stage", r"series [ab]", r"seed", r"incubat",
        r"venture", r"vc.backed",
    ]

    for emp in employer_history:
        emp_lower = emp.lower()
        is_consulting = any(firm in emp_lower for firm in consulting_firms)
        if not is_consulting:
            has_product = True
            is_consulting_only = False
        for pat in startup_patterns:
            if re.search(pat, emp_lower):
                has_startup = True
                break

    if not employer_history:
        is_consulting_only = False

    return {
        "employer_history": employer_history,
        "has_product_company_experience": has_product,
        "consulting_only": is_consulting_only,
        "has_startup_experience": has_startup,
    }


# ---------------------------------------------------------------------------
# Title-skill alignment
# ---------------------------------------------------------------------------

def _compute_title_skill_alignment(skills_list: list, current_title: str, seniority_map: dict) -> float:
    """
    Float 0-1: does skill depth match title seniority?
    Heuristic: senior titles expect >= 8 skills, junior expects <= 5.
    """
    if not skills_list:
        return 0.5
    seniority = _title_to_seniority(current_title or "", seniority_map)
    skill_count = len(skills_list)
    # Expected skill counts by seniority level
    expected_min = max(1, seniority * 2)
    expected_max = seniority * 4 + 5
    if skill_count < expected_min:
        return max(0.0, skill_count / expected_min)
    elif skill_count > expected_max:
        return max(0.0, 1.0 - (skill_count - expected_max) / (expected_max + 1))
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Profile completeness
# ---------------------------------------------------------------------------

def _compute_profile_completeness(candidate: dict) -> float:
    """Use platform's own completeness score — normalized to 0-1."""
    score = _get(candidate, "redrob_signals", "profile_completeness_score")
    if score is not None:
        try:
            return round(min(1.0, max(0.0, float(score) / 100.0)), 3)
        except (ValueError, TypeError):
            pass
    # Fallback: count present nested fields
    checks = [
        _get(candidate, "profile", "anonymized_name"),
        _get(candidate, "profile", "current_title"),
        _get(candidate, "profile", "current_company"),
        _get(candidate, "profile", "location"),
        _get(candidate, "skills"),
        _get(candidate, "career_history"),
        _get(candidate, "education"),
        _get(candidate, "profile", "years_of_experience"),
    ]
    present = sum(1 for f in checks if _is_present(f))
    return round(present / len(checks), 3)


# ---------------------------------------------------------------------------
# Raw profile text for embedding
# ---------------------------------------------------------------------------

def _build_raw_profile_text(candidate: dict, skills_list: list, roles: list[dict]) -> str:
    profile = _get(candidate, "profile") or {}
    title = str(_get(profile, "current_title") or "")
    headline = str(_get(profile, "headline") or "")[:100]
    summary = str(_get(profile, "summary") or "")[:300]
    skills_str = " ".join(skills_list[:30])
    recent_desc = roles[-1]["description"][:500] if roles else ""
    return f"{title} {headline} {skills_str} {recent_desc} {summary}".strip()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def extract_features(
    candidates_path: str,
    honeypot_flags_path: str,
    static_dir: str,
    output_path: str,
) -> None:
    """
    Extract feature vectors for all non-flagged candidates.
    Writes artifacts/features.parquet.
    """
    global STATIC_DIR
    STATIC_DIR = Path(static_dir)

    # Load honeypot flags
    flags_df = pd.read_parquet(honeypot_flags_path)
    flagged_ids = set(flags_df[flags_df["honeypot_score"] == 1]["candidate_id"].astype(str))
    bm25_seed_map = dict(zip(flags_df["candidate_id"].astype(str), flags_df["bm25_seed_score"]))
    logger.info(f"Excluding {len(flagged_ids)} honeypot candidates")

    # Load static configs
    aliases = _load_skill_aliases()
    seniority_map = _load_seniority_map()
    consulting_firms = _load_consulting_firms()

    # Load candidates
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = str(_get(c, "candidate_id") or _get(c, "id") or f"unknown_{i}")
                if cid not in flagged_ids:
                    candidates.append((cid, c))
            except json.JSONDecodeError:
                pass
    logger.info(f"Processing {len(candidates)} non-flagged candidates")

    records = []
    today = pd.Timestamp.today()

    for cid, candidate in tqdm(candidates, desc="Feature extraction"):
        try:
            roles = _parse_work_history(candidate)
            skills_raw = _get(candidate, "skills") or []
            skills_list = _normalize_skills(skills_raw, aliases)

            exp_feats = _compute_experience_features(roles)
            traj_feats = _compute_trajectory_features(roles, seniority_map)
            skill_recency = _compute_skill_recency(roles, skills_list)

            has_recent = any(
                year >= (today.year - 2)
                for year in skill_recency.values()
            ) if skill_recency else False

            # Identity — nested under profile
            profile = _get(candidate, "profile") or {}
            full_name = str(_get(profile, "anonymized_name") or "")
            current_title = str(_get(profile, "current_title") or "")
            current_company = str(_get(profile, "current_company") or "")
            current_location = str(_get(profile, "location") or "")

            company_feats = _compute_company_features(roles, consulting_firms)

            # Behavioral signals — all under redrob_signals
            signals = _get(candidate, "redrob_signals") or {}

            last_active_raw = _get(signals, "last_active_date")
            last_active_ts = _safe_parse_date(last_active_raw)
            platform_active = (
                float((pd.Timestamp.today() - last_active_ts).days)
                if last_active_ts else None
            )

            response_rate = _get(signals, "recruiter_response_rate")
            try:
                response_rate = float(response_rate) if response_rate is not None else None
            except (ValueError, TypeError):
                response_rate = None

            github_score = _get(signals, "github_activity_score")
            try:
                has_open_source = bool(github_score is not None and float(github_score) > 0)
            except (ValueError, TypeError):
                has_open_source = False

            # Certifications — schema has year as integer directly
            certs = _get(candidate, "certifications") or []
            cert_list = certs if isinstance(certs, list) else []
            cert_count = len(cert_list)
            cert_year = None
            for cert in cert_list:
                if not isinstance(cert, dict):
                    continue
                yr = _get(cert, "year")
                try:
                    yr_int = int(yr)
                    if cert_year is None or yr_int > cert_year:
                        cert_year = yr_int
                except (ValueError, TypeError):
                    pass

            # Open source / publications
            has_publications = bool(
                _get(candidate, "publications") or
                _get(profile, "publications") or
                _get(candidate, "papers")
            )

            title_alignment = _compute_title_skill_alignment(skills_list, current_title, seniority_map)
            completeness = _compute_profile_completeness(candidate)
            raw_text = _build_raw_profile_text(candidate, skills_list, roles)

            record = {
                "candidate_id": cid,
                "full_name": full_name,
                "current_title": current_title,
                "current_company": current_company,
                "current_location": current_location,
                "skills_list": skills_list,
                "raw_profile_text": raw_text,
                # Experience
                **exp_feats,
                # Trajectory
                **traj_feats,
                # Skills
                "skill_count": len(skills_list),
                "skill_recency_map": json.dumps(skill_recency),
                "has_recent_skill_activity": has_recent,
                # Behavioral
                "platform_last_active_days": platform_active,
                "recruiter_response_rate": response_rate,
                "certification_count": cert_count,
                "most_recent_certification_year": cert_year,
                "has_open_source": has_open_source,
                "has_publications": has_publications,
                # Consistency
                "title_skill_alignment": round(title_alignment, 3),
                "profile_completeness": completeness,
                # Company
                **company_feats,
                # Honeypot divergence seed
                "bm25_seed_score": bm25_seed_map.get(cid, 0.0),
            }
            records.append(record)

        except Exception as e:
            logger.warning(f"Feature extraction failed for candidate {cid}: {e}")

    df = pd.DataFrame(records)
    
    # Serialize list and dict columns as JSON strings to ensure correct parquet round-trip
    for col in ["skills_list", "seniority_scores", "employer_history"]:
        if col in df.columns:
            df[col] = df[col].apply(json.dumps)
            
    if "skill_recency_map" in df.columns:
        df["skill_recency_map"] = df["skill_recency_map"].apply(
            lambda x: x if isinstance(x, str) else json.dumps(x)
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info(f"Feature extraction complete: {len(df)} candidates -> {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--honeypot-flags", default="artifacts/honeypot_flags.parquet")
    ap.add_argument("--static-dir", default="static")
    ap.add_argument("--output", default="artifacts/features.parquet")
    args = ap.parse_args()
    extract_features(args.candidates, args.honeypot_flags, args.static_dir, args.output)