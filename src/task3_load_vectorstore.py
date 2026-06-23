import os
import gc
import logging
import numpy as np
import pandas as pd
import faiss
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "complaint_embeddings.parquet"
)

EMBEDDING_COL = "embedding"
BATCH_SIZE    = 10_000   
MAX_ROWS      = 500_000  


def load_vectorstore(
    parquet_path: str = PARQUET_PATH,
    batch_size: int = BATCH_SIZE,
    max_rows: int = MAX_ROWS,
):
    """
    Load parquet in small batches up to max_rows, build FAISS index
    incrementally, keep only metadata (no embedding vectors) in DataFrame.

    Returns:
        index       - FAISS IndexFlatIP (cosine via L2-normalized vectors)
        metadata_df - DataFrame with text + metadata only (no embeddings)
                      Row i corresponds exactly to FAISS index position i.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet file not found at '{parquet_path}'. "
            f"Place complaint_embeddings.parquet in data/."
        )

    try:
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows   = parquet_file.metadata.num_rows
        effective    = min(total_rows, max_rows)
        logger.info(
            f"Parquet has {total_rows:,} rows total. "
            f"Will index first {effective:,} rows (cap={max_rows:,}, "
            f"batch={batch_size:,})."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open parquet file: {e}") from e

    index           = None
    metadata_chunks = []
    rows_processed  = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):

           
            if rows_processed >= max_rows:
                break

            df_batch = batch.to_pandas()

            remaining = max_rows - rows_processed
            if len(df_batch) > remaining:
                df_batch = df_batch.iloc[:remaining].copy()

            
            try:
                vectors = np.vstack(df_batch[EMBEDDING_COL].values).astype("float32")
            except KeyError as e:
                raise KeyError(
                    f"Embedding column '{EMBEDDING_COL}' not found. "
                    f"Columns: {list(df_batch.columns)}"
                ) from e

            faiss.normalize_L2(vectors)

            if index is None:
                dim   = vectors.shape[1]
                index = faiss.IndexFlatIP(dim)
                logger.info(f"Initialized FAISS IndexFlatIP (dim={dim}).")

            index.add(vectors)
            del vectors
            gc.collect()

            # --- Keep metadata only ---
            meta_cols = [c for c in df_batch.columns if c != EMBEDDING_COL]
            metadata_chunks.append(df_batch[meta_cols].copy())
            del df_batch
            gc.collect()

            rows_processed += len(metadata_chunks[-1])
            logger.info(f"Indexed {rows_processed:,} / {effective:,} chunks ...")

    except MemoryError as e:
        raise RuntimeError(
            f"Out of memory at row {rows_processed:,}. "
            f"Lower MAX_ROWS (currently {max_rows:,}) to 300_000 and retry."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error during batched loading at row {rows_processed:,}: {e}"
        ) from e

    if index is None or index.ntotal == 0:
        raise RuntimeError("No vectors were added to the FAISS index.")

    try:
        logger.info("Concatenating metadata batches ...")
        metadata_df = pd.concat(metadata_chunks, ignore_index=True)
        del metadata_chunks
        gc.collect()
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate metadata batches: {e}") from e

    logger.info(
        f"Vector store ready — "
        f"FAISS index: {index.ntotal:,} vectors | "
        f"Metadata rows: {len(metadata_df):,}"
    )
    return index, metadata_df


if __name__ == "__main__":
    try:
        index, metadata_df = load_vectorstore()
        logger.info(f"Columns available: {list(metadata_df.columns)}")
        logger.info(f"Sample row:\n{metadata_df.iloc[0].to_dict()}")
    except Exception as err:
        logger.error(f"Vector store load failed: {err}")