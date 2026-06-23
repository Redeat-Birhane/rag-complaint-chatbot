"""
Task 3 — Section 2: Retriever
Embeds a user question with all-MiniLM-L6-v2 and performs cosine similarity
search against the FAISS index to return the top-k most relevant chunks.
"""

import logging
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K        = 5
DOCUMENT_COL = "document"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the same embedding model used to build the vector store."""
    try:
        logger.info(f"Loading embedding model '{model_name}' ...")
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}") from e
    return model


def embed_question(model: SentenceTransformer, question: str) -> np.ndarray:
    """
    Embed a single user question into a normalized float32 vector.

    Args:
        model:    Loaded SentenceTransformer model.
        question: Plain-English question string.

    Returns:
        A (1, 384) float32 numpy array, L2-normalized for cosine search.
    """
    if not isinstance(question, str) or question.strip() == "":
        raise ValueError("Question must be a non-empty string.")

    try:
        vector = model.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vector)
    except Exception as e:
        raise RuntimeError(f"Failed to embed question: {e}") from e

    return vector


def retrieve(
    question: str,
    index: faiss.Index,
    metadata_df: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given question.

    Args:
        question:    User's plain-English question.
        index:       FAISS index (from task3_load_vectorstore.load_vectorstore).
        metadata_df: Aligned metadata DataFrame (row i = FAISS index position i).
        model:       Loaded SentenceTransformer embedding model.
        top_k:       Number of chunks to retrieve.

    Returns:
        List of dicts, each containing:
            - score:           cosine similarity score
            - document:        chunk text
            - complaint_id:    source complaint ID
            - product_category
            - issue
            - company
            - state
            - date_received
            - chunk_index
    """
    try:
        question_vector = embed_question(model, question)
        scores, indices = index.search(question_vector, top_k)
    except Exception as e:
        raise RuntimeError(f"FAISS search failed: {e}") from e

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 for unfilled slots
        try:
            row = metadata_df.iloc[idx]
            # metadata is stored as a nested dict in the 'metadata' column
            meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
            results.append({
                "score":            float(score),
                "document":         row[DOCUMENT_COL],
                "complaint_id":     meta.get("complaint_id", ""),
                "product_category": meta.get("product_category", ""),
                "issue":            meta.get("issue", ""),
                "company":          meta.get("company", ""),
                "state":            meta.get("state", ""),
                "date_received":    meta.get("date_received", ""),
                "chunk_index":      meta.get("chunk_index", 0),
            })
        except Exception as e:
            logger.warning(f"Failed to process result at index {idx}: {e}")
            continue

    logger.info(f"Retrieved {len(results)} chunks for question: '{question[:60]}...'")
    return results


if __name__ == "__main__":
    from task3_load_vectorstore import load_vectorstore

    try:
        index, metadata_df = load_vectorstore()
        model = load_embedding_model()

        test_question = "Why are customers unhappy with their credit cards?"
        results = retrieve(test_question, index, metadata_df, model)

        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score={r['score']:.4f}) ---")
            print(f"Product : {r['product_category']}")
            print(f"Issue   : {r['issue']}")
            print(f"Company : {r['company']}")
            print(f"Excerpt : {r['document'][:300]}")
    except Exception as err:
        logger.error(f"Retriever test failed: {err}")