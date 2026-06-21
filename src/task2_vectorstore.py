"""
Task 2 — Section 4: Vector Store Indexing
Build a persisted FAISS index from the chunk embeddings, storing per-chunk
metadata so retrieved vectors can be traced back to their source complaint.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "chunk_embeddings.npy"
)
METADATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "chunk_metadata.csv"
)
VECTOR_STORE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "vector_store"
)
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "complaints_sample.index")
INDEX_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "complaints_sample_metadata.json")


def load_embeddings(path: str) -> np.ndarray:
    """Load the saved embeddings array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found at '{path}'. Run task2_embed.py first.")
    try:
        embeddings = np.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from '{path}': {e}") from e

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array, got shape {embeddings.shape}.")

    logger.info(f"Loaded embeddings of shape {embeddings.shape}.")
    return embeddings


def load_metadata(path: str) -> pd.DataFrame:
    """Load the chunk metadata CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at '{path}'. Run task2_embed.py first.")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from '{path}': {e}") from e

    if df.empty:
        raise ValueError(f"'{path}' has 0 rows.")

    logger.info(f"Loaded metadata for {len(df):,} chunks.")
    return df


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index over the embeddings using cosine similarity
    (via L2-normalized vectors + inner product index).
    """
    try:
        dim = embeddings.shape[1]

        # Normalize embeddings to unit length so inner product == cosine similarity
        normalized = embeddings.copy().astype("float32")
        faiss.normalize_L2(normalized)

        index = faiss.IndexFlatIP(dim)  # Inner Product index
        index.add(normalized)
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}") from e

    logger.info(f"Built FAISS index with {index.ntotal:,} vectors, dimension {dim}.")
    return index


def save_index(index: faiss.Index, index_path: str) -> None:
    """Persist the FAISS index to disk."""
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to '{index_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index to '{index_path}': {e}") from e


def save_index_metadata(df: pd.DataFrame, out_path: str) -> None:
    """
    Save chunk metadata as a JSON list, where list position i corresponds to
    FAISS index position i — this positional mapping is how we trace a
    retrieved vector back to its source complaint.
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        records = df.to_dict(orient="records")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved index metadata ({len(records):,} records) to '{out_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save index metadata to '{out_path}': {e}") from e


if __name__ == "__main__":
    try:
        embeddings = load_embeddings(EMBEDDINGS_PATH)
        metadata_df = load_metadata(METADATA_PATH)

        if len(embeddings) != len(metadata_df):
            raise RuntimeError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_df)} metadata rows. "
                f"Aborting to avoid building a misaligned index."
            )

        index = build_faiss_index(embeddings)
        save_index(index, FAISS_INDEX_PATH)
        save_index_metadata(metadata_df, INDEX_METADATA_PATH)

        logger.info("Vector store build complete.")
    except Exception as err:
        logger.error(f"Vector store build failed: {err}")