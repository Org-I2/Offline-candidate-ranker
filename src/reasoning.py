"""
Reasoning Module
Generates fact-grounded reasoning strings for each of the top 100 candidates.
No hallucination possible — only field values are inserted into sentence structures.
"""

import datetime
from typing import Any


def _get(d: dict, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


# ---------------------------------------------------------------------------
# Sentence clause constructors — return (clause_text, strength_score)
# ---------------------------------------------------------------------------

def _clause_ascending_trajectory(row: dict) -> tuple[str, float] | None:
    if _get(row, "trajectory_direction") == "ascending":
        title = _get(row, "current_title") or "current role"
        return (f"consistent upward progression to {title}", 0.9)
    return None


def _clause_skill_coverage(row: dict, jd: dict) -> tuple[str, float] | None:
    coverage = _get(row, "required_skill_coverage", 0.0)
    required_skills = _get(jd, "required_skills") or []
    skills_val = _get(row, "skills_list")
    skills_list = list(skills_val) if skills_val is not None else []
    if coverage is None:
        return None
    total = len(required_skills)
    if total == 0:
        return None
    matched = [
        s for s in required_skills
        if s.lower() in [x.lower() for x in skills_list]
    ]
    n_matched = len(matched)
    if n_matched == 0:
        return None
    sample = ", ".join(matched[:3])
    return (f"covers {n_matched}/{total} required skills ({sample})", min(1.0, coverage + 0.1))


def _clause_experience_range(row: dict, jd: dict) -> tuple[str, float] | None:
    years = _get(row, "total_years_exp_computed")
    seniority_range = _get(jd, "seniority_range") or []
    if years is None or not seniority_range or len(seniority_range) < 2:
        return None
    min_y, max_y = seniority_range[0], seniority_range[1]
    if min_y <= years <= max_y:
        return (f"{years:.0f} years experience within target range ({min_y}\u2013{max_y} yrs)", 0.75)
    return None


def _clause_product_background(row: dict) -> tuple[str, float] | None:
    if _get(row, "has_product_company_experience"):
        return ("product-company background", 0.70)
    return None


def _clause_open_source(row: dict) -> tuple[str, float] | None:
    if _get(row, "has_open_source"):
        return ("open-source contributions signal", 0.65)
    return None


def _clause_certifications(row: dict) -> tuple[str, float] | None:
    count = _get(row, "certification_count", 0) or 0
    year = _get(row, "most_recent_certification_year")
    import math
    if count >= 2 and year is not None:
        try:
            year_f = float(year)
            if not math.isnan(year_f):
                current_year = datetime.date.today().year
                if current_year - int(year_f) <= 3:
                    return (f"{count} recent certifications (latest {int(year_f)})", 0.60)
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Gap clause constructors
# ---------------------------------------------------------------------------

def _gap_missing_skills(row: dict, jd: dict) -> str | None:
    required_skills = _get(jd, "required_skills") or []
    skills_val = _get(row, "skills_list")
    skills_list = [s.lower() for s in skills_val] if skills_val is not None else []
    missing = [
        s for s in required_skills
        if s.lower() not in skills_list
    ]
    if missing:
        critical = _get(jd, "critical_skills") or []
        critical_missing = [s for s in missing if s in critical]
        if critical_missing:
            sample = ", ".join(critical_missing[:2])
            return f"missing critical skills: {sample}"
        sample = ", ".join(missing[:2])
        return f"gap in: {sample}"
    return None


def _gap_inactive(row: dict) -> str | None:
    days = _get(row, "platform_last_active_days")
    if days is not None:
        try:
            days_f = float(days)
            if days_f > 180:
                months = int(days_f / 30)
                return f"inactive on platform for {months} months \u2014 availability uncertain"
        except (ValueError, TypeError):
            pass
    return None


def _gap_consulting_only(row: dict) -> str | None:
    if _get(row, "consulting_only"):
        return "consulting-only background; no product-company signal"
    return None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_reasoning(candidate_row: dict, jd: dict, rank: int) -> str:
    """
    Build a fact-grounded reasoning string for a ranked candidate.
    Maximum 2 sentences. Every claim comes from candidate_row fields.
    Tone gates by rank bucket:
      - ranks 1-20: lead with strengths
      - ranks 21-60: balanced
      - ranks 61-100: lead with fit framing, surface gaps prominently

    Args:
        candidate_row: Dict of candidate features (from features parquet + scoring)
        jd: Parsed JD dict from JDParser
        rank: Integer rank 1-100

    Returns:
        Reasoning string (max 2 sentences).
    """
    # Collect all possible strength clauses
    strength_candidates = [
        _clause_ascending_trajectory(candidate_row),
        _clause_skill_coverage(candidate_row, jd),
        _clause_experience_range(candidate_row, jd),
        _clause_product_background(candidate_row),
        _clause_open_source(candidate_row),
        _clause_certifications(candidate_row),
    ]
    strengths = [
        (text, score)
        for item in strength_candidates
        if item is not None
        for text, score in [item]
    ]
    strengths.sort(key=lambda x: x[1], reverse=True)

    # Gap detection
    coverage = _get(candidate_row, "required_skill_coverage", 1.0) or 1.0
    gap_text: str | None = None
    if coverage < 0.8:
        gap_text = _gap_missing_skills(candidate_row, jd)
    if gap_text is None:
        gap_text = _gap_inactive(candidate_row)
    if gap_text is None:
        gap_text = _gap_consulting_only(candidate_row)

    name = _get(candidate_row, "full_name") or _get(candidate_row, "name") or f"Candidate {_get(candidate_row, 'candidate_id')}"
    years_exp = _get(candidate_row, "total_years_exp_computed") or _get(candidate_row, "years_of_experience") or 0.0
    skills_count = _get(candidate_row, "skill_count") or len(_get(candidate_row, "skills_list") or [])

    # Determine how many strengths vs gaps to surface based on rank
    if rank <= 20:
        # Lead with strengths, mention gap only if coverage < 0.8
        top_strengths = [t for t, _ in strengths[:2]]
        strength_sentence = f"{name} demonstrates {', and '.join(top_strengths)}." if top_strengths else f"{name} is a strong match with {years_exp:.1f} years of experience and {skills_count} skills."
        gap_sentence = f" Note: {gap_text}." if (gap_text and coverage < 0.8) else ""
        return (strength_sentence + gap_sentence).strip()

    elif rank <= 60:
        # Balanced — one strength, one gap
        top_strength = strengths[0][0] if strengths else None
        strength_part = f"{name} shows {top_strength}" if top_strength else f"{name} is a reasonable fit with {years_exp:.1f} years of experience"
        if gap_text:
            return f"{strength_part}; however, {gap_text}."
        return f"{strength_part}."

    else:
        # ranks 61-100: fit framing, surface gaps prominently
        exp_fit = _get(candidate_row, "experience_range_fit", 0.5) or 0.5
        top_strength = strengths[0][0] if strengths else None
        company = _get(candidate_row, "current_company") or ""
        years = _get(candidate_row, "total_years_exp_computed") or 0
    
        if gap_text:
            if top_strength:
                return f"{name} has {top_strength} but {gap_text}; marginal fit."
            return f"{name} presents a partial match with {skills_count} skills — {gap_text}."
        if top_strength:
            # Add years to differentiate candidates with same top_strength
            return f"{name} ({years:.0f} yrs, {company}) has {top_strength}; ranked lower due to weaker overall signal."
        return f"{name} is a lower-confidence match with {years:.0f} years at {company} based on available profile data."
