"""
Unit tests for src/scorer.py
Tests edge cases: zero skills match, all skills match, ascending vs descending
trajectory, inactive candidate, consulting-only background.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.scorer import (
    compute_required_skill_coverage,
    compute_trajectory_score,
    compute_behavioral_availability,
    compute_consistency_score,
    compute_skill_depth_score,
    compute_experience_range_fit,
    compute_composite_score,
    is_title_chaser,
    is_consulting_only,
    has_product_company_exp,
)


# ---------------------------------------------------------------------------
# compute_required_skill_coverage
# ---------------------------------------------------------------------------

class TestRequiredSkillCoverage:
    EQUIVALENTS = {"vector database": ["faiss", "pinecone", "weaviate"]}

    def test_zero_match(self):
        score = compute_required_skill_coverage(
            ["java", "php"], ["python", "embeddings"], self.EQUIVALENTS
        )
        assert score == 0.0

    def test_all_match(self):
        score = compute_required_skill_coverage(
            ["python", "embeddings", "vector database"],
            ["python", "embeddings", "vector database"],
            self.EQUIVALENTS,
        )
        assert score == 1.0

    def test_functional_equivalent_match(self):
        # Candidate has 'faiss' — should count for 'vector database'
        score = compute_required_skill_coverage(
            ["python", "faiss"],
            ["python", "vector database"],
            self.EQUIVALENTS,
        )
        assert score == 1.0

    def test_partial_match(self):
        score = compute_required_skill_coverage(
            ["python"],
            ["python", "embeddings", "vector database"],
            self.EQUIVALENTS,
        )
        assert abs(score - 1/3) < 0.01

    def test_empty_required_skills(self):
        score = compute_required_skill_coverage(["python"], [], {})
        assert score == 1.0

    def test_empty_candidate_skills(self):
        score = compute_required_skill_coverage([], ["python", "embeddings"], {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# compute_trajectory_score
# ---------------------------------------------------------------------------

class TestTrajectoryScore:
    def test_ascending_at_target(self):
        score = compute_trajectory_score(0.5, 4, 4)
        assert score > 0.8

    def test_ascending_one_below_target(self):
        score = compute_trajectory_score(0.3, 3, 4)
        assert score > 0.7

    def test_descending_far_below(self):
        score = compute_trajectory_score(-0.5, 1, 5)
        assert score < 0.3

    def test_flat_at_target(self):
        score = compute_trajectory_score(0.0, 4, 4)
        assert score >= 0.6

    def test_two_levels_below(self):
        score = compute_trajectory_score(0.0, 2, 4)
        assert 0.3 <= score <= 0.7

    def test_above_target(self):
        # At or above target should not be penalized
        score = compute_trajectory_score(0.0, 5, 4)
        assert score >= 0.6


# ---------------------------------------------------------------------------
# compute_behavioral_availability
# ---------------------------------------------------------------------------

class TestBehavioralAvailability:
    def test_fully_active(self):
        score = compute_behavioral_availability(30, 0.8)
        assert score == 1.0

    def test_inactive_platform(self):
        score = compute_behavioral_availability(200, 0.8)
        assert abs(score - 0.7) < 0.01

    def test_low_response_rate(self):
        score = compute_behavioral_availability(30, 0.1)
        assert abs(score - 0.8) < 0.01

    def test_both_penalties_stack(self):
        score = compute_behavioral_availability(200, 0.1)
        assert abs(score - 0.56) < 0.01

    def test_none_values(self):
        score = compute_behavioral_availability(None, None)
        assert score == 1.0


# ---------------------------------------------------------------------------
# compute_consistency_score
# ---------------------------------------------------------------------------

class TestConsistencyScore:
    def test_perfect_alignment(self):
        score = compute_consistency_score(1.0, 1.0)
        assert score == 1.0

    def test_poor_alignment(self):
        score = compute_consistency_score(0.0, 0.0)
        assert score == 0.0

    def test_mixed(self):
        score = compute_consistency_score(0.8, 0.5)
        assert 0.5 <= score <= 0.9


# ---------------------------------------------------------------------------
# compute_skill_depth_score
# ---------------------------------------------------------------------------

class TestSkillDepthScore:
    def test_recent_skills(self):
        import datetime
        year = datetime.date.today().year
        score = compute_skill_depth_score(
            {"python": year - 1, "embeddings": year},
            ["python", "embeddings"],
        )
        assert score == 1.0

    def test_old_skills(self):
        score = compute_skill_depth_score(
            {"python": 2015, "embeddings": 2016},
            ["python", "embeddings"],
        )
        assert score < 0.5

    def test_missing_skills(self):
        score = compute_skill_depth_score({}, ["python", "embeddings"])
        assert score == 0.0

    def test_partial_match(self):
        import datetime
        year = datetime.date.today().year
        score = compute_skill_depth_score(
            {"python": year},
            ["python", "embeddings"],
        )
        # Only python matched
        assert 0.4 < score <= 1.0


# ---------------------------------------------------------------------------
# compute_experience_range_fit
# ---------------------------------------------------------------------------

class TestExperienceRangeFit:
    def test_in_range_center(self):
        score = compute_experience_range_fit(6.0, [4, 8])
        assert score == 1.0

    def test_below_range(self):
        score = compute_experience_range_fit(2.0, [4, 8])
        assert score < 0.5

    def test_above_range(self):
        score = compute_experience_range_fit(12.0, [4, 8])
        # Slight penalty for over-qualification
        assert score < 1.0

    def test_exact_min(self):
        score = compute_experience_range_fit(4.0, [4, 8])
        assert score >= 0.5

    def test_empty_range(self):
        score = compute_experience_range_fit(6.0, [])
        assert score == 0.5


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    WEIGHTS = {
        "semantic_similarity": 0.25,
        "required_skill_coverage": 0.25,
        "trajectory_score": 0.20,
    }

    def test_perfect_signals(self):
        signals = {"semantic_similarity": 1.0, "required_skill_coverage": 1.0, "trajectory_score": 1.0}
        score = compute_composite_score(signals, self.WEIGHTS)
        assert abs(score - 1.0) < 0.01

    def test_zero_signals(self):
        signals = {"semantic_similarity": 0.0, "required_skill_coverage": 0.0, "trajectory_score": 0.0}
        score = compute_composite_score(signals, self.WEIGHTS)
        assert score == 0.0

    def test_missing_signal_excluded(self):
        signals = {"semantic_similarity": 1.0}  # other signals missing
        score = compute_composite_score(signals, self.WEIGHTS)
        assert 0.0 < score <= 1.0

    def test_score_bounded(self):
        signals = {"semantic_similarity": 2.0, "required_skill_coverage": 3.0, "trajectory_score": 5.0}
        score = compute_composite_score(signals, self.WEIGHTS)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# is_title_chaser
# ---------------------------------------------------------------------------

class TestTitleChaser:
    def test_title_chaser_pattern(self):
        result = is_title_chaser(
            seniority_scores=[1, 2, 3, 4],
            employer_history=["company_a", "company_b", "company_c", "company_d"],
            max_avg_tenure_months=18,
            min_company_switches=3,
        )
        assert result is True

    def test_not_enough_switches(self):
        result = is_title_chaser(
            seniority_scores=[1, 2],
            employer_history=["company_a", "company_b"],
            max_avg_tenure_months=18,
            min_company_switches=3,
        )
        assert result is False

    def test_lateral_moves(self):
        # Same seniority across switches = not a title chaser
        result = is_title_chaser(
            seniority_scores=[3, 3, 3, 3],
            employer_history=["a", "b", "c", "d"],
            max_avg_tenure_months=18,
            min_company_switches=3,
        )
        assert result is False


# ---------------------------------------------------------------------------
# is_consulting_only / has_product_company_exp
# ---------------------------------------------------------------------------

class TestConsultingOnly:
    CONSULTING = ["tcs", "infosys", "wipro"]

    def test_consulting_only(self):
        assert is_consulting_only(["tcs", "infosys"], self.CONSULTING) is True

    def test_has_product(self):
        assert is_consulting_only(["tcs", "google"], self.CONSULTING) is False

    def test_empty_history(self):
        assert is_consulting_only([], self.CONSULTING) is False

    def test_has_product_company_exp(self):
        assert has_product_company_exp(["google", "tcs"], self.CONSULTING) is True

    def test_no_product_company_exp(self):
        assert has_product_company_exp(["tcs", "infosys"], self.CONSULTING) is False
