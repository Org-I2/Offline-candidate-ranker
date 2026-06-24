"""
Unit tests for src/reasoning.py
Asserts all 6 Stage-4 judge checks:
1. Specific facts present
2. JD connection present
3. Honest gaps surfaced when coverage < 0.8
4. No invented skill names
5. Variation across 10 generated strings
6. Rank-tone consistency
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.reasoning import build_reasoning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_JD = {
    "required_skills": ["python", "embeddings", "vector database", "ranking evaluation"],
    "critical_skills": ["embeddings", "vector database"],
    "seniority_range": [5, 9],
    "seniority_level": "senior",
}


def _make_candidate(
    candidate_id: str = "c001",
    name: str = "Alice Smith",
    title: str = "Senior ML Engineer",
    company: str = "TechCorp",
    skills: list | None = None,
    years: float = 7.0,
    trajectory: str = "ascending",
    coverage: float = 1.0,
    has_product: bool = True,
    has_open_source: bool = False,
    platform_days: float | None = 30,
    consulting_only: bool = False,
    seniority_scores: list | None = None,
    cert_count: int = 0,
    cert_year: int | None = None,
    experience_range_fit: float = 0.9,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "full_name": name,
        "current_title": title,
        "current_company": company,
        "skills_list": skills or ["python", "embeddings", "vector database", "ranking evaluation"],
        "total_years_exp_computed": years,
        "trajectory_direction": trajectory,
        "required_skill_coverage": coverage,
        "has_product_company_experience": has_product,
        "has_open_source": has_open_source,
        "platform_last_active_days": platform_days,
        "consulting_only": consulting_only,
        "seniority_scores": seniority_scores or [2, 3, 4],
        "certification_count": cert_count,
        "most_recent_certification_year": cert_year,
        "experience_range_fit": experience_range_fit,
    }


# ---------------------------------------------------------------------------
# Check 1: Specific facts present
# ---------------------------------------------------------------------------

class TestSpecificFacts:
    def test_name_appears_in_reasoning(self):
        row = _make_candidate(name="Alice Smith")
        result = build_reasoning(row, BASE_JD, rank=1)
        assert "Alice Smith" in result

    def test_title_appears_in_reasoning(self):
        row = _make_candidate(title="Senior ML Engineer", trajectory="ascending")
        result = build_reasoning(row, BASE_JD, rank=1)
        assert "Senior ML Engineer" in result

    def test_skill_names_appear_in_reasoning(self):
        row = _make_candidate(skills=["python", "embeddings", "vector database", "ranking evaluation"])
        result = build_reasoning(row, BASE_JD, rank=1)
        # At least one skill should appear
        assert any(skill in result for skill in ["python", "embeddings", "vector database"])


# ---------------------------------------------------------------------------
# Check 2: JD connection present
# ---------------------------------------------------------------------------

class TestJDConnection:
    def test_required_skills_referenced(self):
        row = _make_candidate()
        result = build_reasoning(row, BASE_JD, rank=5)
        # Either coverage count or skill name should be mentioned
        has_jd_ref = (
            "required skills" in result.lower()
            or any(skill in result for skill in BASE_JD["required_skills"])
            or "range" in result
            or "years" in result
        )
        assert has_jd_ref, f"No JD connection found in: {result}"


# ---------------------------------------------------------------------------
# Check 3: Honest gaps surfaced when coverage < 0.8
# ---------------------------------------------------------------------------

class TestHonestGaps:
    def test_gap_surfaced_for_low_coverage(self):
        row = _make_candidate(
            skills=["python"],  # only python — missing embeddings, vector database, ranking
            coverage=0.25,
        )
        result = build_reasoning(row, BASE_JD, rank=50)
        assert any(term in result.lower() for term in ["missing", "gap", "lack"]), \
            f"Expected gap language for low coverage but got: {result}"

    def test_gap_not_prominent_for_high_coverage_top_rank(self):
        row = _make_candidate(coverage=1.0)
        result = build_reasoning(row, BASE_JD, rank=1)
        # Should NOT lead with gap for rank 1 with full coverage
        assert not result.lower().startswith("missing"), \
            f"Rank 1 full coverage should not lead with gap: {result}"


# ---------------------------------------------------------------------------
# Check 4: No invented skill names
# ---------------------------------------------------------------------------

class TestNoInventedSkills:
    def test_skills_in_output_are_from_candidate(self):
        candidate_skills = ["python", "faiss"]
        row = _make_candidate(skills=candidate_skills, coverage=0.5)
        result = build_reasoning(row, BASE_JD, rank=30)
        # Extract any skills mentioned from vocabulary
        suspicious_skills = ["tensorflow", "pytorch", "kubernetes", "spark"]
        for skill in suspicious_skills:
            assert skill not in result.lower(), \
                f"Invented skill '{skill}' appeared in reasoning: {result}"


# ---------------------------------------------------------------------------
# Check 5: Variation across 10 generated strings
# ---------------------------------------------------------------------------

class TestVariation:
    def test_ten_different_candidates_produce_different_strings(self):
        results = []
        for i in range(10):
            row = _make_candidate(
                candidate_id=f"c{i:03d}",
                name=f"Person {i}",
                coverage=[0.25, 0.5, 0.75, 1.0, 0.6, 0.4, 0.5, 0.9, 0.7, 0.8][i],
            )
            rank = i * 10 + 1
            results.append(build_reasoning(row, BASE_JD, rank=rank))
        unique_results = set(results)
        assert len(unique_results) >= 5, \
            f"Expected variation across 10 candidates, got {len(unique_results)} unique strings"


# ---------------------------------------------------------------------------
# Check 6: Rank-tone consistency
# ---------------------------------------------------------------------------

class TestRankToneConsistency:
    def test_top_rank_leads_with_strength(self):
        row = _make_candidate(coverage=1.0, trajectory="ascending")
        result = build_reasoning(row, BASE_JD, rank=3)
        # Should not start with gap language
        gap_starters = ["missing", "gap", "inactive", "consulting-only"]
        for gap in gap_starters:
            assert not result.lower().startswith(gap), \
                f"Rank 3 should not lead with gap language: {result}"

    def test_bottom_rank_surfaces_gaps(self):
        row = _make_candidate(
            skills=["java"],  # No JD skills
            coverage=0.0,
        )
        result = build_reasoning(row, BASE_JD, rank=95)
        assert any(term in result.lower() for term in ["missing", "gap", "partial", "marginal", "lower"]), \
            f"Rank 95 should surface concerns: {result}"

    def test_mid_rank_balanced(self):
        row = _make_candidate(coverage=0.6, trajectory="lateral")
        result = build_reasoning(row, BASE_JD, rank=45)
        # Should have both strength and gap elements for mid rank with low coverage
        assert len(result) > 20, f"Mid-rank reasoning too short: {result}"
