"""
Precompute Orchestrator
Runs all four precompute scripts in order with timing.

Usage:
    python precompute/run_all.py --candidates ./data/candidates.jsonl --jd ./data/jd.txt
"""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

# Ensure repo root is on sys.path so src/ is importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
STATIC_DIR = REPO_ROOT / "static"
MODEL_DIR = REPO_ROOT / "models" / "minilm"
PRECOMPUTE_DIR = REPO_ROOT / "precompute"

# All artifact paths are hardcoded here
HONEYPOT_FLAGS_PATH = ARTIFACTS_DIR / "honeypot_flags.parquet"
FEATURES_PATH = ARTIFACTS_DIR / "features.parquet"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
FAISS_IDS_PATH = ARTIFACTS_DIR / "candidate_ids_faiss.npy"
BM25_PATH = ARTIFACTS_DIR / "bm25.pkl"
BM25_IDS_PATH = ARTIFACTS_DIR / "candidate_ids_bm25.npy"
COMPANY_FOUNDING_YEARS_PATH = STATIC_DIR / "company_founding_years.json"


def _load_module(filename: str, module_name: str):
    """
    Load a Python module from file path using importlib.
    Required because precompute filenames start with digits (01_, 02_, etc.)
    which are not valid Python identifiers and can't be imported directly.
    """
    path = PRECOMPUTE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Precompute script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _step(name: str, fn, *args, **kwargs) -> float:
    """Run one step, print elapsed time. Propagates exceptions."""
    _banner(f"STEP: {name}")
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  [OK] {name} completed in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all precompute steps in order (01 -> 02 -> 03 -> 04)."
    )
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--jd", required=True, help="Path to JD file (.docx/.txt)")
    args = ap.parse_args()

    candidates_path = str(Path(args.candidates).resolve())
    jd_path = str(Path(args.jd).resolve())

    if not Path(candidates_path).exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)
    if not Path(jd_path).exists():
        print(f"ERROR: JD file not found: {jd_path}")
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parse JD upfront to get required skills for BM25 seed scoring in step 01
    jd_required_skills: list[str] | None = None
    try:
        from src.jd_parser import JDParser
        parser = JDParser(static_dir=STATIC_DIR)
        jd = parser.parse(jd_path)
        jd_required_skills = jd.get("required_skills", [])
        print(f"  JD pre-parsed: {len(jd_required_skills)} required skills found")
    except Exception as e:
        print(f"  WARNING: JD pre-parsing failed — BM25 seed will use defaults. Error: {e}")

    # Load modules via importlib (numeric prefixes block direct import)
    _banner("Loading precompute modules")
    mod01 = _load_module("01_honeypot_detector.py", "honeypot_detector")
    mod02 = _load_module("02_feature_extractor.py", "feature_extractor")
    mod03 = _load_module("03_embedder.py", "embedder")
    mod04 = _load_module("04_build_bm25.py", "build_bm25")
    print("  All modules loaded")

    wall_start = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        timings["01_honeypot_detection"] = _step(
            "01 - Honeypot Detection",
            mod01.detect_honeypots,
            candidates_path,
            str(HONEYPOT_FLAGS_PATH),
            str(COMPANY_FOUNDING_YEARS_PATH),
            jd_required_skills,
        )

        timings["02_feature_extraction"] = _step(
            "02 - Feature Extraction",
            mod02.extract_features,
            candidates_path,
            str(HONEYPOT_FLAGS_PATH),
            str(STATIC_DIR),
            str(FEATURES_PATH),
        )

        timings["03_embedding_faiss"] = _step(
            "03 - Embedding + FAISS Index",
            mod03.build_embeddings,
            str(FEATURES_PATH),
            str(MODEL_DIR),
            str(FAISS_INDEX_PATH),
            str(FAISS_IDS_PATH),
        )

        timings["04_bm25_index"] = _step(
            "04 - BM25 Index",
            mod04.build_bm25,
            str(FEATURES_PATH),
            str(BM25_PATH),
            str(BM25_IDS_PATH),
        )

    except Exception as e:
        print(f"\n  [ERROR] FATAL ERROR in precompute: {e}")
        traceback.print_exc()
        sys.exit(1)

    total = time.perf_counter() - wall_start
    _banner("PRECOMPUTE COMPLETE")
    for step, t in timings.items():
        print(f"  {step:<35} {t:.1f}s")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<35} {total:.1f}s")
    print(f"\n  Artifacts written to: {ARTIFACTS_DIR}")
    print(f"  Files:")
    for p in [HONEYPOT_FLAGS_PATH, FEATURES_PATH, FAISS_INDEX_PATH, FAISS_IDS_PATH, BM25_PATH, BM25_IDS_PATH]:
        size_mb = p.stat().st_size / 1_048_576 if p.exists() else 0
        status = f"{size_mb:.1f} MB" if p.exists() else "MISSING"
        print(f"    {p.name:<40} {status}")


if __name__ == "__main__":
    main()
