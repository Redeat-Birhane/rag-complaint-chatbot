"""
Task 2 — Section 2: Text Chunking
Split long complaint narratives into smaller overlapping chunks suitable for
embedding, using LangChain's RecursiveCharacterTextSplitter.
"""

import os
import logging
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "stratified_sample.csv"
)
OUTPUT_CHUNKS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "chunked_complaints.csv"
)

# Chosen to roughly match the pre-built vector store spec (500 chars / 50 overlap)
# so our own pipeline is directly comparable to the production-scale store used in Task 3-4.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

NARRATIVE_COL = "cleaned_narrative"
ID_COL = "Complaint ID"
PRODUCT_COL = "product_category"


def load_sample(csv_path: str) -> pd.DataFrame:
    """Load the stratified sample produced in Section 1."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Stratified sample not found at '{csv_path}'. Run task2_sample.py first."
        )
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load '{csv_path}': {e}") from e

    if df.empty:
        raise ValueError(f"'{csv_path}' has 0 rows.")

    logger.info(f"Loaded {len(df):,} sampled complaints.")
    return df


def chunk_narratives(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE,
                      chunk_overlap: int = CHUNK_OVERLAP) -> pd.DataFrame:
    """
    Split each narrative into overlapping text chunks and attach per-chunk metadata
    (complaint ID, product category, chunk index, total chunks) for traceability.

    Returns:
        A new DataFrame, one row per chunk, with columns:
        complaint_id, product_category, chunk_index, total_chunks, chunk_text
    """
    try:
        if NARRATIVE_COL not in df.columns:
            raise KeyError(f"Column '{NARRATIVE_COL}' not found. Columns: {list(df.columns)}")
        if ID_COL not in df.columns:
            raise KeyError(f"Column '{ID_COL}' not found. Columns: {list(df.columns)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to initialize text splitter: {e}") from e

    records = []
    failed_rows = 0

    for _, row in df.iterrows():
        try:
            narrative = row[NARRATIVE_COL]
            if not isinstance(narrative, str) or narrative.strip() == "":
                continue

            chunks = splitter.split_text(narrative)
            total_chunks = len(chunks)

            for idx, chunk_text in enumerate(chunks):
                records.append({
                    "complaint_id": row[ID_COL],
                    "product_category": row.get(PRODUCT_COL, None),
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "chunk_text": chunk_text,
                })
        except Exception as e:
            failed_rows += 1
            logger.warning(f"Failed to chunk complaint ID {row.get(ID_COL, '?')}: {e}")
            continue

    if failed_rows:
        logger.warning(f"{failed_rows:,} rows failed chunking and were skipped.")

    if not records:
        raise RuntimeError("No chunks were produced — check input data and chunking config.")

    chunks_df = pd.DataFrame(records)
    logger.info(
        f"Produced {len(chunks_df):,} chunks from {df[ID_COL].nunique():,} complaints "
        f"(avg {len(chunks_df) / df[ID_COL].nunique():.1f} chunks/complaint)."
    )
    return chunks_df


def save_chunks(df: pd.DataFrame, out_path: str) -> None:
    """Save the chunked dataset to CSV."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df):,} chunks to '{out_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save chunks to '{out_path}': {e}") from e


if __name__ == "__main__":
    try:
        df_sample = load_sample(INPUT_SAMPLE_PATH)
        df_chunks = chunk_narratives(df_sample)
        save_chunks(df_chunks, OUTPUT_CHUNKS_PATH)
    except Exception as err:
        logger.error(f"Chunking run failed: {err}")