"""
Unit tests for src/jd_parser.py
Runs parser against actual data/job_description.docx and asserts expected outputs.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.jd_parser import JDParser

JD_PATH = REPO_ROOT / "data" / "job_description.docx"


@pytest.fixture(scope="module")
def parsed_jd():
    """Parse the actual JD file once for all tests."""
    if not JD_PATH.exists():
        pytest.skip(f"JD file not found at {JD_PATH}")
    parser = JDParser(static_dir=REPO_ROOT / "static")
    return parser.parse(str(JD_PATH))


class TestRequiredSkills:
    def test_embeddings_in_required_skills(self, parsed_jd):
        required = [s.lower() for s in parsed_jd["required_skills"]]
        assert "embeddings" in required, \
            f"'embeddings' not in required_skills: {parsed_jd['required_skills']}"

    def test_vector_database_in_required_skills(self, parsed_jd):
        required = [s.lower() for s in parsed_jd["required_skills"]]
        assert "vector database" in required, \
            f"'vector database' not in required_skills: {parsed_jd['required_skills']}"

    def test_ranking_evaluation_in_required_skills(self, parsed_jd):
        required = [s.lower() for s in parsed_jd["required_skills"]]
        assert "ranking evaluation" in required, \
            f"'ranking evaluation' not in required_skills: {parsed_jd['required_skills']}"

    def test_python_in_required_skills(self, parsed_jd):
        required = [s.lower() for s in parsed_jd["required_skills"]]
        assert "python" in required, \
            f"'python' not in required_skills: {parsed_jd['required_skills']}"


class TestCriticalSkills:
    def test_exactly_four_critical_skills(self, parsed_jd):
        assert len(parsed_jd["critical_skills"]) == 4, \
            f"Expected 4 critical skills, got {len(parsed_jd['critical_skills'])}: {parsed_jd['critical_skills']}"


class TestSeniorityLevel:
    def test_seniority_is_senior(self, parsed_jd):
        assert parsed_jd["seniority_level"] == "senior", \
            f"Expected seniority_level='senior', got '{parsed_jd['seniority_level']}'"


class TestRoleArchetype:
    def test_role_archetype_is_builder(self, parsed_jd):
        assert parsed_jd["role_archetype"] == "builder", \
            f"Expected role_archetype='builder', got '{parsed_jd['role_archetype']}'"


class TestExtractionConfidence:
    def test_overall_confidence_above_threshold(self, parsed_jd):
        conf = parsed_jd["extraction_confidence"]["overall_confidence"]
        assert conf > 0.85, \
            f"Expected overall_confidence > 0.85, got {conf}"
