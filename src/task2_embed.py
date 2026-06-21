"""
Task 2 — Section 3: Embedding Generation
Generate vector embeddings for each text chunk using sentence-transformers
(all-MiniLM-L6-v2), matching the model used in the pre-built vector store.
"""

import os
import logging
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_CHUNKS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "chunked_complaints.csv"
)
OUTPUT_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "chunk_embeddings.npy"
)
OUTPUT_METADATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "chunk_metadata.csv"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
CHUNK_TEXT_COL = "chunk_text"


def load_chunks(csv_path: str) -> pd.DataFrame:
    """Load the chunked complaints dataset produced in Section 2."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Chunked complaints file not found at '{csv_path}'. Run task2_chunk.py first."
        )
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load '{csv_path}': {e}") from e

    if df.empty:
        raise ValueError(f"'{csv_path}' has 0 rows.")
    if CHUNK_TEXT_COL not in df.columns:
        raise KeyError(f"Column '{CHUNK_TEXT_COL}' not found. Columns: {list(df.columns)}")

    # Drop any rows with missing/empty chunk text (shouldn't normally happen, but be safe)
    before = len(df)
    df = df[df[CHUNK_TEXT_COL].notna() & (df[CHUNK_TEXT_COL].astype(str).str.strip() != "")].copy()
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after:,} rows with empty chunk text.")

    logger.info(f"Loaded {len(df):,} chunks for embedding.")
    return df


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    try:
        logger.info(f"Loading embedding model '{model_name}' ...")
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e

    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(model: SentenceTransformer, texts: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Generate embeddings for a list of text chunks, batched for efficiency."""
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}") from e

    logger.info(f"Generated embeddings of shape {embeddings.shape}.")
    return embeddings


def save_embeddings(embeddings: np.ndarray, out_path: str) -> None:
    """Save embeddings as a .npy file."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, embeddings)
        logger.info(f"Saved embeddings to '{out_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save embeddings to '{out_path}': {e}") from e


def save_metadata(df: pd.DataFrame, out_path: str) -> None:
    """Save chunk metadata (aligned by row order with the embeddings array)."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved chunk metadata to '{out_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save metadata to '{out_path}': {e}") from e


if __name__ == "__main__":
    try:
        df_chunks = load_chunks(INPUT_CHUNKS_PATH)
        model = load_embedding_model()
        embeddings = generate_embeddings(model, df_chunks[CHUNK_TEXT_COL].tolist())

        # Sanity check: embeddings row count must match metadata row count exactly,
        # since the vector store relies on positional alignment between the two.
        if len(embeddings) != len(df_chunks):
            raise RuntimeError(
                f"Mismatch between embeddings ({len(embeddings)}) and metadata "
                f"({len(df_chunks)}) row counts — aborting save to avoid misaligned index."
            )

        save_embeddings(embeddings, OUTPUT_EMBEDDINGS_PATH)
        save_metadata(df_chunks, OUTPUT_METADATA_PATH)
    except Exception as err:
        logger.error(f"Embedding run failed: {err}")