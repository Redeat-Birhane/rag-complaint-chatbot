"""
Section 4 (optimized): Clean Consumer Narrative Text — vectorized for large datasets
- Lowercase text
- Remove boilerplate phrases
- Remove special characters / PII redaction artifacts
- Normalize whitespace
"""

import logging
import pandas as pd

from task1_load_data import load_raw_complaints, RAW_CSV_PATH
from task1_filter import run_filtering, NARRATIVE_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    r"i am writing to file a complaint\b",
    r"i am writing this complaint\b",
    r"this is a complaint (against|regarding)\b",
    r"to whom it may concern[,]?",
    r"\bx{2,}\b",
    r"\bxx/xx/xxxx\b",
]


def clean_narratives(df: pd.DataFrame, source_col: str = NARRATIVE_COL,
                      target_col: str = "cleaned_narrative") -> pd.DataFrame:
    """
    Vectorized cleaning of all narratives at once using pandas .str methods,
    instead of per-row .apply(). Much faster on large (millions-of-rows) datasets.
    """
    try:
        df = df.copy()
        cleaned = df[source_col].astype(str).str.lower()

        for pattern in BOILERPLATE_PATTERNS:
            cleaned = cleaned.str.replace(pattern, " ", regex=True)

        # Remove special characters, keep letters/numbers/basic punctuation/spaces
        cleaned = cleaned.str.replace(r"[^a-z0-9.,!?'\s]", " ", regex=True)

        # Collapse whitespace
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()

        df[target_col] = cleaned
    except KeyError as e:
        raise KeyError(f"Column '{source_col}' not found. Columns: {list(df.columns)}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error cleaning narratives: {e}") from e

    before = len(df)
    df = df[df[target_col].str.strip() != ""].copy()
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after:,} rows that were empty after cleaning.")

    logger.info(f"Cleaned narratives for {after:,} rows.")
    return df


if __name__ == "__main__":
    try:
        df_raw = load_raw_complaints(RAW_CSV_PATH)
        df_filtered = run_filtering(df_raw)
        df_clean = clean_narratives(df_filtered)
        print(df_clean[["product_category", "cleaned_narrative"]].head())
        print(df_clean.shape)
    except Exception as err:
        logger.error(f"Cleaning run failed: {err}")