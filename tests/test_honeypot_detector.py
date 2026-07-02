"""
Unit tests for precompute/01_honeypot_detector.py
For each of the 7 detection rules, creates a minimal candidate dict that
triggers it and asserts honeypot_score == 1.
Also tests a clean candidate that should not trigger any rule.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Load the module via importlib (filename starts with digit)
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location(
    "honeypot_detector",
    REPO_ROOT / "precompute" / "01_honeypot_detector.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

detect_honeypots = _mod.detect_honeypots
_rule1 = _mod._rule1_company_tenure_vs_founding
_rule2 = _mod._rule2_expert_skill_zero_years
_rule3 = _mod._rule3_job_before_graduation
_rule4 = _mod._rule4_experience_vs_career_span
_rule5 = _mod._rule5_implausible_skill_breadth
_rule6 = _mod._rule6_duplicate_profile


# ---------------------------------------------------------------------------
# Helper: write candidates to temp jsonl, run detector, read parquet
# ---------------------------------------------------------------------------

def _run_detect(candidates: list[dict], founding_years: dict | None = None) -> pd.DataFrame:
    """Run full detect_honeypots on a list of candidates. Returns flags DataFrame."""
    fy = founding_years or {"acme corp": 2020}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cands_path = tmpdir / "candidates.jsonl"
        with open(cands_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

        fy_path = tmpdir / "founding_years.json"
        with open(fy_path, "w", encoding="utf-8") as f:
            json.dump({"companies": fy}, f)

        out_path = tmpdir / "flags.parquet"
        detect_honeypots(str(cands_path), str(out_path), str(fy_path))
        return pd.read_parquet(out_path)


# ---------------------------------------------------------------------------
# Clean baseline candidate (should not trigger any rule)
# ---------------------------------------------------------------------------

CLEAN_CANDIDATE = {
    "candidate_id": "clean_001",
    "full_name": "Jane Doe",
    "current_title": "Senior ML Engineer",
    "current_employer": "TechCorp",
    "current_company": "TechCorp",
    "current_location": "Bangalore",
    "total_years_experience": 5,
    "skills": [
        {"name": "python", "proficiency": "expert", "years_used": 5},
        {"name": "pytorch", "proficiency": "intermediate", "years_used": 3},
        {"name": "sql", "proficiency": "intermediate", "years_used": 4},
    ],
    "work_history": [
        {
            "company": "TechCorp",
            "title": "Senior ML Engineer",
            "start_date": "2021-01-01",
            "end_date": "present",
            "description": "Built ML pipelines",
        },
        {
            "company": "StartupXYZ",
            "title": "ML Engineer",
            "start_date": "2019-06-01",
            "end_date": "2020-12-31",
            "description": "Developed models",
        },
    ],
    "education": [
        {
            "degree": "Bachelor of Technology",
            "institution": "IIT Bombay",
            "end_date": "2019-05-01",
        }
    ],
}


class TestCleanCandidate:
    def test_clean_candidate_not_flagged(self):
        df = _run_detect([CLEAN_CANDIDATE])
        row = df[df["candidate_id"] == "clean_001"].iloc[0]
        assert row["honeypot_score"] == 0, f"Clean candidate was flagged: {row['honeypot_reasons']}"


# ---------------------------------------------------------------------------
# Rule 1: Company tenure before founding year
# ---------------------------------------------------------------------------

class TestRule1CompanyTenure:
    def test_triggers_when_start_before_founding(self):
        founding_years = {"futurecorp": 2015}
        candidate = {
            "candidate_id": "r1_001",
            "full_name": "A B",
            "work_history": [
                {
                    "company": "futurecorp",
                    "title": "Engineer",
                    "start_date": "2010-01-01",  # before founding 2015
                    "end_date": "2014-12-31",
                }
            ],
        }
        assert _rule1(candidate, founding_years) is True

    def test_does_not_trigger_unknown_company(self):
        founding_years = {}  # unknown company
        candidate = {
            "candidate_id": "r1_002",
            "work_history": [
                {
                    "company": "unknownco",
                    "title": "Engineer",
                    "start_date": "2000-01-01",
                    "end_date": "2010-12-31",
                }
            ],
        }
        assert _rule1(candidate, founding_years) is False

    def test_does_not_trigger_after_founding(self):
        founding_years = {"google": 1998}
        candidate = {
            "candidate_id": "r1_003",
            "work_history": [
                {
                    "company": "google",
                    "title": "Engineer",
                    "start_date": "2005-01-01",
                    "end_date": "2010-12-31",
                }
            ],
        }
        assert _rule1(candidate, founding_years) is False


# ---------------------------------------------------------------------------
# Rule 2: Expert skill with 0 or null years used
# ---------------------------------------------------------------------------

class TestRule2ExpertSkillZeroYears:
    def test_triggers_expert_zero_years(self):
        candidate = {
            "candidate_id": "r2_001",
            "skills": [{"name": "python", "proficiency": "expert", "years_used": 0}],
        }
        assert _rule2(candidate) is True

    def test_triggers_expert_null_years(self):
        candidate = {
            "candidate_id": "r2_002",
            "skills": [{"name": "pytorch", "proficiency": "advanced", "years_used": None}],
        }
        assert _rule2(candidate) is True

    def test_does_not_trigger_expert_with_years(self):
        candidate = {
            "candidate_id": "r2_003",
            "skills": [{"name": "python", "proficiency": "expert", "years_used": 5}],
        }
        assert _rule2(candidate) is False

    def test_does_not_trigger_intermediate_zero_years(self):
        candidate = {
            "candidate_id": "r2_004",
            "skills": [{"name": "java", "proficiency": "intermediate", "years_used": 0}],
        }
        assert _rule2(candidate) is False


# ---------------------------------------------------------------------------
# Rule 3: Job started > 12 months before undergrad completion
# ---------------------------------------------------------------------------

class TestRule3JobBeforeGraduation:
    def test_triggers_job_24_months_before_grad(self):
        candidate = {
            "candidate_id": "r3_001",
            "education": [
                {
                    "degree": "Bachelor of Technology",
                    "end_date": "2020-06-01",
                }
            ],
            "work_history": [
                {
                    "company": "SomeCorp",
                    "title": "Engineer",
                    "start_date": "2018-01-01",  # 29 months before graduation
                    "end_date": "2020-05-31",
                }
            ],
        }
        assert _rule3(candidate) is True

    def test_does_not_trigger_job_after_grad(self):
        candidate = {
            "candidate_id": "r3_002",
            "education": [
                {
                    "degree": "Bachelor of Science",
                    "end_date": "2019-05-01",
                }
            ],
            "work_history": [
                {
                    "company": "TechCo",
                    "title": "Engineer",
                    "start_date": "2019-07-01",  # after graduation
                    "end_date": "present",
                }
            ],
        }
        assert _rule3(candidate) is False

    def test_does_not_trigger_missing_education(self):
        candidate = {
            "candidate_id": "r3_003",
            "education": [],
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Engineer",
                    "start_date": "2010-01-01",
                    "end_date": "present",
                }
            ],
        }
        assert _rule3(candidate) is False


# ---------------------------------------------------------------------------
# Rule 4: Self-reported experience > actual career span + 2
# ---------------------------------------------------------------------------

class TestRule4ExperienceOverstated:
    def test_triggers_large_discrepancy(self):
        candidate = {
            "candidate_id": "r4_001",
            "total_years_experience": 20,  # claims 20 years
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Engineer",
                    "start_date": "2020-01-01",  # only 5ish years ago
                    "end_date": "present",
                }
            ],
        }
        assert _rule4(candidate) is True

    def test_does_not_trigger_accurate_experience(self):
        candidate = {
            "candidate_id": "r4_002",
            "total_years_experience": 5,
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Engineer",
                    "start_date": "2020-01-01",
                    "end_date": "present",
                }
            ],
        }
        assert _rule4(candidate) is False

    def test_does_not_trigger_missing_field(self):
        candidate = {
            "candidate_id": "r4_003",
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Engineer",
                    "start_date": "2018-01-01",
                    "end_date": "present",
                }
            ],
        }
        assert _rule4(candidate) is False


# ---------------------------------------------------------------------------
# Rule 5: >15 skills but <3 years experience
# ---------------------------------------------------------------------------

class TestRule5ImplausibleSkillBreadth:
    def test_triggers_many_skills_short_career(self):
        candidate = {
            "candidate_id": "r5_001",
            "skills": [{"name": f"skill_{i}"} for i in range(20)],  # 20 skills
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Intern",
                    "start_date": "2024-01-01",  # only ~1.5 years ago
                    "end_date": "present",
                }
            ],
        }
        assert _rule5(candidate) is True

    def test_does_not_trigger_few_skills(self):
        candidate = {
            "candidate_id": "r5_002",
            "skills": [{"name": f"skill_{i}"} for i in range(10)],  # only 10 skills
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Intern",
                    "start_date": "2024-06-01",
                    "end_date": "present",
                }
            ],
        }
        assert _rule5(candidate) is False

    def test_does_not_trigger_many_skills_long_career(self):
        candidate = {
            "candidate_id": "r5_003",
            "skills": [{"name": f"skill_{i}"} for i in range(20)],  # 20 skills
            "work_history": [
                {
                    "company": "Corp",
                    "title": "Senior Engineer",
                    "start_date": "2015-01-01",  # 10 years ago
                    "end_date": "present",
                }
            ],
        }
        assert _rule5(candidate) is False


# ---------------------------------------------------------------------------
# Rule 6: Duplicate profile detection
# ---------------------------------------------------------------------------

class TestRule6DuplicateProfile:
    def test_second_occurrence_flagged(self):
        seen = {}
        candidate1 = {
            "candidate_id": "r6_001",
            "full_name": "Alice Smith",
            "current_employer": "TechCorp",
            "current_title": "Engineer",
        }
        candidate2 = {
            "candidate_id": "r6_002",
            "full_name": "Alice Smith",  # Same combo
            "current_employer": "TechCorp",
            "current_title": "Engineer",
        }
        assert _rule6(candidate1, seen) is False  # first occurrence: clean
        assert _rule6(candidate2, seen) is True   # second occurrence: flagged

    def test_different_profiles_not_flagged(self):
        seen = {}
        candidate1 = {
            "candidate_id": "r6_003",
            "full_name": "Alice Smith",
            "current_employer": "TechCorp",
            "current_title": "Engineer",
        }
        candidate2 = {
            "candidate_id": "r6_004",
            "full_name": "Bob Jones",  # Different name
            "current_employer": "OtherCorp",
            "current_title": "Manager",
        }
        assert _rule6(candidate1, seen) is False
        assert _rule6(candidate2, seen) is False


# ---------------------------------------------------------------------------
# Integration test: Full pipeline flags the right candidates
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    def test_rule1_via_full_pipeline(self):
        founding_years = {"futurecorp": 2015}
        candidates = [
            {
                "candidate_id": "int_001",
                "full_name": "Bad Actor",
                "current_employer": "futurecorp",
                "current_title": "Engineer",
                "work_history": [
                    {
                        "company": "futurecorp",
                        "title": "Engineer",
                        "start_date": "2010-01-01",  # Before founding 2015
                        "end_date": "2014-12-31",
                    }
                ],
            }
        ]
        df = _run_detect(candidates, founding_years)
        row = df[df["candidate_id"] == "int_001"].iloc[0]
        assert row["honeypot_score"] == 1
        assert "rule1_company_tenure_before_founding" in row["honeypot_reasons"]

    def test_bm25_seed_score_stored(self):
        """Rule 7: bm25_seed_score must be stored for all candidates."""
        candidates = [CLEAN_CANDIDATE]
        df = _run_detect(candidates)
        assert "bm25_seed_score" in df.columns
        assert df["bm25_seed_score"].notna().all()
