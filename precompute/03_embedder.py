"""
Embedder — Pre-computation Step 3
Generates semantic embeddings for all non-flagged candidates and builds a FAISS index.
"""

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def build_embeddings(
    features_path: str,
    model_dir: str,
    output_index_path: str,
    output_ids_path: str,
    batch_size: int = 256,
) -> None:
    """
    Generate sentence embeddings for all candidates and build a FAISS IndexFlatIP.

    Args:
        features_path: Path to artifacts/features.parquet
        model_dir: Path to local MiniLM model directory (models/minilm/)
        output_index_path: Path to write artifacts/faiss.index
        output_ids_path: Path to write artifacts/candidate_ids_faiss.npy
        batch_size: Batch size for embedding (default 256 for CPU efficiency)
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} candidate feature rows")

    # Load model from local directory — NO network access
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"MiniLM model not found at {model_path}. "
            f"Download once with: python -c \""
            f"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('{model_path}')\""
        )
    logger.info(f"Loading SentenceTransformer from {model_path}")
    model = SentenceTransformer(str(model_path), device="cpu")

    texts = df["raw_profile_text"].fillna("").tolist()
    candidate_ids = df["candidate_id"].astype(str).tolist()
    n = len(texts)
    dim = 384  # all-MiniLM-L6-v2 output dimension

    logger.info(f"Embedding {n} candidates in batches of {batch_size}...")
    all_embeddings = np.zeros((n, dim), dtype=np.float32)

    for start in tqdm(range(0, n, batch_size), desc="Embedding candidates"):
        batch = texts[start: start + batch_size]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize so IP = cosine similarity
        )
        all_embeddings[start: start + len(batch)] = embs

    # L2 normalize all embeddings (belt-and-suspenders — ST already normalizes)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    all_embeddings = all_embeddings / norms

    logger.info("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    logger.info(f"FAISS index contains {index.ntotal} vectors")

    # Save FAISS index
    out_index = Path(output_index_path)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index))
    logger.info(f"FAISS index saved to {out_index}")

    # Save candidate_ids array in same order as FAISS index
    out_ids = Path(output_ids_path)
    np.save(str(out_ids), np.array(candidate_ids, dtype=object))
    logger.info(f"Candidate IDs saved to {out_ids} ({len(candidate_ids)} entries)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features.parquet")
    ap.add_argument("--model-dir", default="models/minilm")
    ap.add_argument("--output-index", default="artifacts/faiss.index")
    ap.add_argument("--output-ids", default="artifacts/candidate_ids_faiss.npy")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    build_embeddings(args.features, args.model_dir, args.output_index, args.output_ids, args.batch_size)
