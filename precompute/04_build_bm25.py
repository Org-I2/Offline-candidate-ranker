"""
BM25 Index Builder — Pre-computation Step 4
Builds a BM25 index over candidate text for fast lexical retrieval at runtime.
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25. Lowercase + strip punctuation + split.
    Deliberately does NOT remove stopwords — skill names like 'go' or 'r' matter.
    Preserves hyphens so 'machine-learning' stays intact.
    """
    text = text.lower()
    # Replace punctuation with spaces but keep alphanumeric and hyphens
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.strip().split()
    return tokens


def build_bm25(
    features_path: str,
    output_bm25_path: str,
    output_ids_path: str,
) -> None:
    """
    Build BM25Okapi index over candidate raw_profile_text.

    Args:
        features_path: Path to artifacts/features.parquet
        output_bm25_path: Path to write artifacts/bm25.pkl
        output_ids_path: Path to write artifacts/candidate_ids_bm25.npy
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} candidate rows")

    texts = df["raw_profile_text"].fillna("").tolist()
    candidate_ids = df["candidate_id"].astype(str).tolist()

    logger.info("Tokenizing corpus...")
    tokenized_corpus: list[list[str]] = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokenized_corpus.append(_tokenize(text))

    logger.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built")

    # Pack both BM25 object and tokenized corpus together
    payload = {
        "bm25": bm25,
        "corpus": tokenized_corpus,
    }

    out_bm25 = Path(output_bm25_path)
    out_bm25.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bm25, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"BM25 index saved to {out_bm25}")

    out_ids = Path(output_ids_path)
    np.save(str(out_ids), np.array(candidate_ids, dtype=object))
    logger.info(f"Candidate IDs saved to {out_ids} ({len(candidate_ids)} entries)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features.parquet")
    ap.add_argument("--output-bm25", default="artifacts/bm25.pkl")
    ap.add_argument("--output-ids", default="artifacts/candidate_ids_bm25.npy")
    args = ap.parse_args()
    build_bm25(args.features, args.output_bm25, args.output_ids)
