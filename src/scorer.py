"""
Scorer Module
Pure scoring functions used by rank.py. No side effects, no I/O.
All functions take primitive Python types and return floats/bools.
"""

import datetime
from typing import Optional


def compute_required_skill_coverage(candidate_skills, required_skills, functional_equivalents):
    if not required_skills:
        return 1.0
    
    def _norm(s):
        return s.lower().strip().replace("-", " ").replace("_", " ") if isinstance(s, str) else ""
    
    candidate_set = {_norm(s) for s in (candidate_skills or []) if s}
    
    covered = 0
    for skill in required_skills:
        skill_norm = _norm(skill)
        if skill_norm in candidate_set:
            covered += 1
            continue
        equiv_list = (
            functional_equivalents.get(skill)
            or functional_equivalents.get(skill.lower().strip())
            or []
        )
        equivalents = [_norm(e) for e in equiv_list if e]
        if any(eq in candidate_set for eq in equivalents):
            covered += 1
    
    return covered / len(required_skills)


def compute_trajectory_score(
    trajectory_slope: float,
    current_seniority: int,
    seniority_target: int,
) -> float:
    """
    Score combining trajectory direction and distance from target seniority.
    - Peak score when current_seniority == seniority_target or one level below.
    - Heavy penalty when > 2 levels below target.
    Returns 0.0–1.0.
    """
    # Slope contribution: normalize [-1, 1] → [0, 1]
    clamped_slope = min(1.0, max(-1.0, float(trajectory_slope or 0.0)))
    slope_score = (clamped_slope + 1.0) / 2.0  # 0 = descending, 0.5 = flat, 1 = ascending

    # Seniority distance contribution
    delta = int(seniority_target) - int(current_seniority or 0)
    if delta <= 0:
        seniority_score = 1.0          # at or above target
    elif delta == 1:
        seniority_score = 0.85         # one level below — ideal growth trajectory
    elif delta == 2:
        seniority_score = 0.50         # two levels below — marginal
    else:
        seniority_score = max(0.0, 0.50 - (delta - 2) * 0.15)  # heavy penalty

    return round(slope_score * 0.4 + seniority_score * 0.6, 4)


def compute_behavioral_availability(
    platform_last_active_days: Optional[float],
    recruiter_response_rate: Optional[float],
) -> float:
    """
    Score reflecting candidate availability and responsiveness.
    Base score is 1.0. Multiplicative penalties applied independently.
    - platform_last_active_days > 180 → ×0.7
    - recruiter_response_rate < 0.30  → ×0.8
    Both penalties can stack.
    Returns 0.0–1.0.
    """
    score = 1.0
    if platform_last_active_days is not None:
        try:
            if float(platform_last_active_days) > 180:
                score *= 0.7
        except (ValueError, TypeError):
            pass
    if recruiter_response_rate is not None:
        try:
            if float(recruiter_response_rate) < 0.30:
                score *= 0.8
        except (ValueError, TypeError):
            pass
    return round(score, 4)


def compute_consistency_score(
    title_skill_alignment: float,
    profile_completeness: float,
) -> float:
    """
    Score how internally consistent and complete the profile is.
    title_skill_alignment (0–1): skill depth vs title seniority match
    profile_completeness (0–1): fraction of key fields populated
    Returns 0.0–1.0.
    """
    # Explicit None checks — do NOT use `or 0.5` here because 0.0 is a valid score
    # and Python treats 0.0 as falsy, which would incorrectly replace it with 0.5.
    ta = max(0.0, min(1.0, float(title_skill_alignment) if title_skill_alignment is not None else 0.5))
    pc = max(0.0, min(1.0, float(profile_completeness) if profile_completeness is not None else 0.5))
    return round(ta * 0.6 + pc * 0.4, 4)


def compute_skill_depth_score(
    skill_recency_map: dict,
    required_skills: list,
) -> float:
    """
    Score how recently required skills were used by the candidate.
    Infers recency from role timelines stored in skill_recency_map (skill → year).
    - Used ≤2 years ago → 1.0 credit
    - Used 3–4 years ago → 0.7 credit
    - Used 5–6 years ago → 0.4 credit
    - Older → 0.1 credit
    Returns 0.0–1.0.
    """
    if not required_skills or not skill_recency_map:
        return 0.0
    current_year = datetime.date.today().year
    total_score = 0.0
    matches = 0
    for skill in required_skills:
        skill_lower = skill.lower().strip()
        year = skill_recency_map.get(skill_lower) or skill_recency_map.get(skill)
        if year is None:
            continue
        matches += 1
        try:
            age = current_year - int(year)
        except (ValueError, TypeError):
            continue
        if age <= 2:
            total_score += 1.0
        elif age <= 4:
            total_score += 0.7
        elif age <= 6:
            total_score += 0.4
        else:
            total_score += 0.1
    if matches == 0:
        return 0.0
    return round(total_score / matches, 4)


def compute_experience_range_fit(
    total_years_exp_computed: float,
    seniority_range: list,
) -> float:
    """
    Triangular score peaking at center of [min_years, max_years].
    - At or above center → 1.0 (with small over-qualification penalty above max)
    - Below min → partial credit; the closer to min, the higher the credit
    Returns 0.0–1.0.
    """
    if not seniority_range or len(seniority_range) < 2:
        return 0.5
    try:
        min_yrs = float(seniority_range[0])
        max_yrs = float(seniority_range[1])
        years = float(total_years_exp_computed or 0.0)
    except (ValueError, TypeError):
        return 0.5

    if max_yrs <= min_yrs:
        return 1.0 if abs(years - min_yrs) <= 1 else 0.5

    if years < min_yrs:
        gap = min_yrs - years
        return max(0.0, round(1.0 - gap * 0.3, 4))

    if years > max_yrs:
        # Over-qualification: slight penalty
        gap = years - max_yrs
        return max(0.5, round(1.0 - gap * 0.05, 4))

    # Within range — triangular: peak at center (1.0), edges score 0.5 minimum.
    # Clamping to 0.5 ensures candidates exactly at the boundary still get meaningful credit.
    mid = (min_yrs + max_yrs) / 2.0
    half_width = (max_yrs - min_yrs) / 2.0
    distance_from_mid = abs(years - mid)
    raw_score = 1.0 - distance_from_mid / (half_width + 1e-9)
    score = max(0.5, min(1.0, raw_score))  # clamp: never below 0.5 when within range
    return round(score, 4)


def compute_composite_score(signals: dict, weights: dict) -> float:
    """
    Weighted sum of signal scores. Weights need not sum to 1 — result is
    normalized by the sum of weights for present signals.
    Returns 0.0–1.0.
    """
    total = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        val = signals.get(key)
        if val is not None:
            try:
                total += float(val) * float(weight)
                total_weight += float(weight)
            except (TypeError, ValueError):
                pass
    if total_weight == 0:
        return 0.0
    return round(min(1.0, max(0.0, total / total_weight)), 6)


def is_title_chaser(
    seniority_scores: list,
    employer_history: list,
    max_avg_tenure_months: int,
    min_company_switches: int,
) -> bool:
    """
    Detect title-chasing: frequent company switches where each switch escalated title.
    Returns True if the pattern is detected.

    Args:
        seniority_scores: Ordered list of numeric seniority levels per role
        employer_history: Ordered list of employer names per role
        max_avg_tenure_months: Threshold for average tenure (unused here — evaluated by caller)
        min_company_switches: Minimum number of switches to be considered a pattern
    """
    if not seniority_scores or not employer_history:
        return False
    switches = len(employer_history) - 1
    if switches < min_company_switches:
        return False
    if len(seniority_scores) < 2:
        return False
    # Count how many transitions involved a title escalation
    escalations = sum(
        1
        for i in range(1, len(seniority_scores))
        if seniority_scores[i] > seniority_scores[i - 1]
    )
    # Flag if at least 50% of switches involved escalation
    return escalations >= max(1, switches * 0.5)


def is_consulting_only(
    employer_history: list,
    disqualifying_companies: list,
) -> bool:
    """
    Returns True if ALL employers in history match the disqualifying (consulting) list.
    Empty history returns False.
    """
    if not employer_history:
        return False
    disq_set = {c.lower().strip() for c in disqualifying_companies}
    return all(
        any(d in emp.lower() for d in disq_set)
        for emp in employer_history
        if emp
    )


def has_product_company_exp(
    employer_history: list,
    disqualifying_companies: list,
) -> bool:
    """
    Returns True if at least one employer is NOT in the consulting/disqualifying list.
    Empty history returns False.
    """
    if not employer_history:
        return False
    disq_set = {c.lower().strip() for c in disqualifying_companies}
    return any(
        not any(d in emp.lower() for d in disq_set)
        for emp in employer_history
        if emp
    )


def compute_notice_period_score(notice_period_days: Optional[float]) -> float:
    """
    Score based on candidate's notice period.
    JD preference: sub-30-day notice strongly preferred; 30+ still in scope.
    Returns 0.0–1.0.
    """
    if notice_period_days is None:
        return 0.65  # Unknown — neutral-ish, not penalised heavily
    try:
        days = float(notice_period_days)
    except (ValueError, TypeError):
        return 0.65
    if days <= 0:
        return 1.0   # Immediately available
    if days <= 15:
        return 1.0
    if days <= 30:
        return 0.85
    if days <= 60:
        return 0.65
    return 0.40      # > 60 days — significant penalty


def compute_location_score(location: Optional[str]) -> float:
    """
    Score based on candidate's current location vs JD preferences.
    JD: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR welcome.
    Returns 0.0–1.0.
    """
    if not location:
        return 0.55  # Unknown — mild penalty
    loc = location.lower()
    # Tier 0 — explicitly preferred
    if any(c in loc for c in ["pune", "noida"]):
        return 1.0
    # Tier 1 — explicitly welcomed
    if any(c in loc for c in ["delhi", "ncr", "gurgaon", "gurugram", "hyderabad", "mumbai"]):
        return 0.85
    # Tier 2 — other major Indian tech hubs (Bengaluru listed as welcome to apply)
    if any(c in loc for c in ["bengaluru", "bangalore", "chennai", "kolkata", "ahmedabad"]):
        return 0.75
    # International / Tier-3 / Unknown region
    return 0.50


def compute_github_score(github_activity_score: Optional[float]) -> float:
    """
    Normalize a raw github_activity_score (0–100 platform scale) to 0–1.
    Score of 0 means no activity; higher is better.
    Returns 0.0–1.0.
    """
    if github_activity_score is None:
        return 0.0
    try:
        raw = float(github_activity_score)
    except (ValueError, TypeError):
        return 0.0
    return round(min(1.0, max(0.0, raw / 100.0)), 4)


def compute_domain_penalty(skills_list: list) -> float:
    """
    Detects candidates whose primary domain is CV/Speech/Robotics with NO NLP/IR exposure.
    JD explicitly says these candidates would need to re-learn fundamentals.
    Returns a multiplier: 1.0 (no penalty) or 0.75 (domain mismatch penalty).
    """
    if not skills_list:
        return 1.0

    cv_speech_robotics = {
        "computer vision", "computer-vision", "object detection", "image classification",
        "image segmentation", "gan", "gans", "generative adversarial", "yolo",
        "opencv", "speech recognition", "speech synthesis", "text-to-speech", "tts",
        "asr", "automatic speech recognition", "robotics", "ros", "slam",
        "autonomous driving", "lidar", "point cloud",
    }
    nlp_ir_skills = {
        "nlp", "natural language processing", "information retrieval", "search",
        "ranking", "embeddings", "embedding", "rag", "retrieval", "semantic search",
        "text classification", "named entity recognition", "ner", "sentiment",
        "transformers", "bert", "gpt", "llm", "language model", "question answering",
        "summarization", "machine translation", "bm25", "faiss", "vector",
        "recommendation", "recommender", "learning to rank", "ltr",
    }

    skills_lower = {s.lower().strip() for s in skills_list if s}

    cv_count = sum(1 for s in skills_lower if any(cv in s for cv in cv_speech_robotics))
    nlp_count = sum(1 for s in skills_lower if any(nl in s for nl in nlp_ir_skills))

    # Only penalise if CV/Robotics is the dominant domain AND no NLP/IR signal at all
    if cv_count >= 3 and nlp_count == 0:
        return 0.75
    return 1.0
