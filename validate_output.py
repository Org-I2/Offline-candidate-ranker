"""
Standalone CLI wrapper for src/validator.py.
Allows humans to validate any submission CSV at any time.

Usage:
    python validate_output.py --submission ./submission.csv --candidates ./data/candidates.jsonl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure src is importable from repo root
sys.path.insert(0, str(Path(__file__).parent))
from src.validator import validate_output


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate a submission CSV against the candidates source file."
    )
    ap.add_argument(
        "--submission",
        required=True,
        help="Path to submission CSV (e.g. ./submission.csv)",
    )
    ap.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates JSONL file (e.g. ./data/candidates.jsonl)",
    )
    args = ap.parse_args()

    submission_path = Path(args.submission)
    candidates_path = Path(args.candidates)

    if not submission_path.exists():
        print(f"FAIL - Submission file not found: {submission_path}")
        sys.exit(1)

    if not candidates_path.exists():
        print(f"FAIL - Candidates file not found: {candidates_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(submission_path, dtype={"candidate_id": str})
    except Exception as e:
        print(f"FAIL - Could not read CSV: {e}")
        sys.exit(1)

    try:
        validate_output(df, str(candidates_path))
        print("PASS - All 9 validation checks passed.")
        print(f"  Rows: {len(df)}")
        print(f"  Score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
    except ValueError as e:
        print(f"FAIL - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL - Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
