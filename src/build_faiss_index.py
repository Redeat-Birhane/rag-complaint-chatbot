import os
import numpy as np
import pandas as pd
import faiss
import pyarrow.parquet as pq
import logging
import gc

logging.basicConfig(level=logging.INFO)

PARQUET_PATH = "../data/complaint_embeddings.parquet"

INDEX_PATH = "../data/complaints.faiss"
META_PATH  = "../data/metadata.pkl"

BATCH_SIZE = 2000   # safer than 5000
MAX_ROWS   = 100000


def build_index():
    pf = pq.ParquetFile(PARQUET_PATH)

    index = None
    metadata_chunks = []
    rows = 0

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["document", "embedding", "metadata"]
    ):

        if rows >= MAX_ROWS:
            break

        df = batch.to_pandas()

        remaining = MAX_ROWS - rows
        df = df.iloc[:remaining]

        # -------------------------------
        # SAFE EMBEDDING CONVERSION
        # -------------------------------
        embeddings = df["embedding"].values

        dim = len(embeddings[0])
        vectors = np.empty((len(embeddings), dim), dtype="float32")

        for i, emb in enumerate(embeddings):
            vectors[i] = np.asarray(emb, dtype="float32")

        faiss.normalize_L2(vectors)

        # -------------------------------
        # INIT FAISS INDEX
        # -------------------------------
        if index is None:
            index = faiss.IndexFlatIP(dim)

        index.add(vectors)

        # -------------------------------
        # STORE METADATA ONLY
        # -------------------------------
        metadata_chunks.append(df[["document", "metadata"]].copy())

        rows += len(df)

        logging.info(f"Indexed {rows}/{MAX_ROWS}")

        # -------------------------------
        # CLEAN MEMORY EACH BATCH
        # -------------------------------
        del df
        del embeddings
        del vectors
        gc.collect()

    # -------------------------------
    # FINAL MERGE
    # -------------------------------
    metadata_df = pd.concat(metadata_chunks, ignore_index=True)

    faiss.write_index(index, INDEX_PATH)
    metadata_df.to_pickle(META_PATH)

    logging.info("Index saved successfully!")


if __name__ == "__main__":
    build_index()