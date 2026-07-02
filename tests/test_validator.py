"""
Unit tests for src/validator.py
For each of the 9 checks, constructs a DataFrame that fails only that check
and asserts the validator raises with the right message.
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.validator import validate_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df(n: int = 100) -> pd.DataFrame:
    """Create a valid 100-row DataFrame."""
    import hashlib
    rows = []
    for i in range(n):
        rows.append({
            "candidate_id": f"cand_{i:05d}",
            "rank": i + 1,
            "score": round(1.0 - i * 0.009, 6),
            "reasoning": f"Candidate {i} shows strong alignment with {i} years of experience.",
        })
    return pd.DataFrame(rows)


def _make_candidates_file(n: int = 100) -> str:
    """Write a temp candidates.jsonl with n candidates and return path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for i in range(n):
        json.dump({"candidate_id": f"cand_{i:05d}", "name": f"Person {i}"}, tmp)
        tmp.write("\n")
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests — one failing check at a time
# ---------------------------------------------------------------------------

class TestCheck1_ExactlyHundredRows:
    def test_too_few_rows(self, tmp_path):
        candidates_path = _make_candidates_file(100)
        df = _make_valid_df(99)
        with pytest.raises(ValueError, match="Check 1"):
            validate_output(df, candidates_path)

    def test_too_many_rows(self):
        candidates_path = _make_candidates_file(200)
        df = _make_valid_df(101)
        with pytest.raises(ValueError, match="Check 1"):
            validate_output(df, candidates_path)


class TestCheck2_RequiredColumns:
    def test_missing_rank_column(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df = df.drop(columns=["rank"])
        with pytest.raises(ValueError, match="Check 2"):
            validate_output(df, candidates_path)

    def test_missing_score_column(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df = df.drop(columns=["score"])
        with pytest.raises(ValueError, match="Check 2"):
            validate_output(df, candidates_path)


class TestCheck3_RanksOneToHundred:
    def test_duplicate_rank(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df.at[0, "rank"] = 2  # duplicate rank 2
        with pytest.raises(ValueError, match="Check 3"):
            validate_output(df, candidates_path)

    def test_rank_out_of_range(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df.at[99, "rank"] = 101  # rank 101 instead of 100
        with pytest.raises(ValueError, match="Check 3"):
            validate_output(df, candidates_path)


class TestCheck4_ScoresMonotonic:
    def test_score_above_one(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df.at[0, "score"] = 1.5  # above 1.0
        with pytest.raises(ValueError, match="Check 4"):
            validate_output(df, candidates_path)

    def test_score_not_monotonic(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        # Make rank 5 score higher than rank 4
        df.at[3, "score"] = 0.5
        df.at[4, "score"] = 0.9  # higher score at lower rank
        with pytest.raises(ValueError, match="Check 4"):
            validate_output(df, candidates_path)


class TestCheck5_CandidateIdExists:
    def test_unknown_candidate_id(self):
        candidates_path = _make_candidates_file(100)  # IDs: cand_00000 to cand_00099
        df = _make_valid_df()
        df.at[0, "candidate_id"] = "nonexistent_id_xyz"
        with pytest.raises(ValueError, match="Check 5"):
            validate_output(df, candidates_path)


class TestCheck6_NoDuplicateIds:
    def test_duplicate_candidate_id(self):
        candidates_path = _make_candidates_file(100)
        df = _make_valid_df()
        df.at[1, "candidate_id"] = df.at[0, "candidate_id"]  # duplicate
        with pytest.raises(ValueError, match="Check 6"):
            validate_output(df, candidates_path)


class TestCheck7_NoEmptyReasoning:
    def test_empty_string_reasoning(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df.at[5, "reasoning"] = ""
        with pytest.raises(ValueError, match="Check 7"):
            validate_output(df, candidates_path)

    def test_null_reasoning(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df.at[5, "reasoning"] = None
        with pytest.raises(ValueError, match="Check 7"):
            validate_output(df, candidates_path)


class TestCheck8_UniqueReasoning:
    def test_identical_reasoning(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df["reasoning"] = "Same reasoning for everyone."  # all identical
        with pytest.raises(ValueError, match="Check 8"):
            validate_output(df, candidates_path)


class TestCheck9_Utf8Safe:
    def test_valid_utf8(self):
        """Normal unicode should pass check 9."""
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        df.at[0, "reasoning"] = "Valid UTF-8: r\u00e9sum\u00e9, \u00fcber, na\u00efve candidate."
        # Should not raise
        validate_output(df, candidates_path)


class TestPassAll:
    def test_valid_dataframe_passes(self):
        candidates_path = _make_candidates_file()
        df = _make_valid_df()
        validate_output(df, candidates_path)  # Should not raise
