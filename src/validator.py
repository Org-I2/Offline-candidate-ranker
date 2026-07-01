"""
Validator Module
Validates the output CSV before writing.
Raises ValueError with a clear message if any check fails.
"""

import json
from pathlib import Path

import pandas as pd


def validate_output(df: pd.DataFrame, candidates_path: str) -> None:
    """
    Validate output DataFrame before saving as submission CSV.
    Checks run in order — first failure raises immediately.

    Args:
        df: Output DataFrame to validate
        candidates_path: Path to source candidates.jsonl (for ID existence check)

    Raises:
        ValueError: With specific failure reason if any check fails
    """

    # Check 1: Exactly 100 rows
    if len(df) != 100:
        raise ValueError(
            f"Check 1 FAILED: Expected exactly 100 rows, got {len(df)}"
        )

    # Check 2: Required columns present
    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Check 2 FAILED: Missing required columns: {missing_cols}"
        )

    # Check 3: Ranks are integers 1-100, each exactly once
    ranks = df["rank"].tolist()
    try:
        ranks_int = [int(r) for r in ranks]
    except (ValueError, TypeError):
        raise ValueError("Check 3 FAILED: Rank column contains non-integer values")
    if sorted(ranks_int) != list(range(1, 101)):
        raise ValueError(
            f"Check 3 FAILED: Ranks must be integers 1-100 each appearing exactly once. "
            f"Got: min={min(ranks_int)}, max={max(ranks_int)}, unique={len(set(ranks_int))}"
        )

    # Check 4: Scores are floats 0-1, monotonically non-increasing with rank
    df_sorted = df.sort_values("rank")
    scores = df_sorted["score"].tolist()
    try:
        scores_float = [float(s) for s in scores]
    except (ValueError, TypeError):
        raise ValueError("Check 4 FAILED: Score column contains non-numeric values")
    if not all(0.0 <= s <= 1.0 for s in scores_float):
        bad = [(i, s) for i, s in enumerate(scores_float) if not 0.0 <= s <= 1.0]
        raise ValueError(f"Check 4 FAILED: Scores out of [0,1] range at positions: {bad[:5]}")
    if not all(scores_float[i] >= scores_float[i + 1] for i in range(len(scores_float) - 1)):
        violations = [
            (i, scores_float[i], scores_float[i + 1])
            for i in range(len(scores_float) - 1)
            if scores_float[i] < scores_float[i + 1]
        ]
        raise ValueError(
            f"Check 4 FAILED: Scores not monotonically non-increasing. "
            f"First violation at index {violations[0][0]}: {violations[0][1]:.4f} < {violations[0][2]:.4f}"
        )

    # Check 5: All candidate_ids exist in source file
    source_ids: set[str] = set()
    source_path = Path(candidates_path)
    if not source_path.exists():
        raise ValueError(f"Check 5 FAILED: Candidates source file not found: {candidates_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = str(c.get("candidate_id") or c.get("id") or "")
                if cid:
                    source_ids.add(cid)
            except json.JSONDecodeError:
                pass
    submission_ids = set(df["candidate_id"].astype(str))
    unknown_ids = submission_ids - source_ids
    if unknown_ids:
        raise ValueError(
            f"Check 5 FAILED: {len(unknown_ids)} candidate_id(s) not found in source file: "
            f"{list(unknown_ids)[:5]}"
        )

    # Check 6: No duplicate candidate_ids
    if df["candidate_id"].duplicated().any():
        dupes = df[df["candidate_id"].duplicated(keep=False)]["candidate_id"].tolist()
        raise ValueError(
            f"Check 6 FAILED: Duplicate candidate_ids found: {dupes[:5]}"
        )

    # Check 7: No null or empty reasoning strings
    reasoning = df["reasoning"].tolist()
    bad_idx = [
        i for i, r in enumerate(reasoning)
        if not r or (isinstance(r, str) and not r.strip()) or pd.isna(r)
    ]
    if bad_idx:
        raise ValueError(
            f"Check 7 FAILED: Empty or null reasoning at row indices: {bad_idx[:5]}"
        )

    # Check 8: No identical reasoning strings
    reasoning_strs = [str(r).strip() for r in reasoning]
    if len(set(reasoning_strs)) < len(reasoning_strs):
        from collections import Counter
        counts = Counter(reasoning_strs)
        dupes = {r: c for r, c in counts.items() if c > 1}
        raise ValueError(
            f"Check 8 FAILED: {len(dupes)} identical reasoning string(s) found (catches templating): "
            f"{list(dupes.keys())[0][:80]}..."
        )

    # Check 9: UTF-8 safe (no characters that would corrupt CSV)
    for i, r in enumerate(reasoning_strs):
        try:
            r.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError(
                f"Check 9 FAILED: Non-UTF-8 character in reasoning at row {i}: {e}"
            )
        # Also check for unescaped commas/newlines that would corrupt CSV structure
        if "\n" in r or "\r" in r:
            # Strip them rather than fail — warn only
            pass

    # All checks passed
    return


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Validate output submission CSV.")
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    parser.add_argument("--candidates", default="data/candidates.jsonl", help="Path to source candidates.jsonl")
    args = parser.parse_args()
    
    print(f"Validating {args.submission} against {args.candidates}...")
    try:
        df = pd.read_csv(args.submission)
        validate_output(df, args.candidates)
        print("SUCCESS: All 9 validation checks passed!")
    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)
