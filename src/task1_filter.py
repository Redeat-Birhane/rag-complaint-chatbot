"""
Section 3: Filter CFPB Complaints
- Map raw 'Product' values to the 4 canonical target categories
- Retain only the 4 target products: Credit Card, Personal Loan, Savings Account, Money Transfer
- Drop records with empty/missing consumer narratives
"""

import logging
import pandas as pd

from task1_load_data import load_raw_complaints, RAW_CSV_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"

# Raw CFPB 'Product' values vary by year/export — map known variants to our
# 4 canonical categories.
PRODUCT_MAP = {
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",
    "Bank account or service": "Savings Account",
    "Checking or savings account": "Savings Account",
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
}

TARGET_PRODUCTS = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]


def map_products(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw 'Product' values into the 4 canonical target categories."""
    try:
        df = df.copy()
        df["product_category"] = df[PRODUCT_COL].map(PRODUCT_MAP).fillna(df[PRODUCT_COL])
    except KeyError as e:
        raise KeyError(f"Column '{PRODUCT_COL}' not found. Columns: {list(df.columns)}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error mapping products: {e}") from e

    return df


def filter_target_products(df: pd.DataFrame, target_products: list = TARGET_PRODUCTS) -> pd.DataFrame:
    """Keep only rows belonging to the target product categories."""
    try:
        if "product_category" not in df.columns:
            raise KeyError("'product_category' column missing — run map_products() first.")
        before = len(df)
        filtered = df[df["product_category"].isin(target_products)].copy()
        after = len(filtered)
    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error filtering target products: {e}") from e

    logger.info(f"Filtered to target products: {before:,} -> {after:,} rows.")

    # Log breakdown per category for the report
    breakdown = filtered["product_category"].value_counts()
    logger.info(f"Breakdown by target category:\n{breakdown}")

    return filtered


def drop_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the consumer narrative is missing or blank."""
    try:
        before = len(df)
        mask = df[NARRATIVE_COL].notna() & (df[NARRATIVE_COL].astype(str).str.strip() != "")
        filtered = df[mask].copy()
        after = len(filtered)
    except KeyError as e:
        raise KeyError(f"Column '{NARRATIVE_COL}' not found. Columns: {list(df.columns)}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error dropping empty narratives: {e}") from e

    logger.info(f"Dropped empty narratives: {before:,} -> {after:,} rows.")
    return filtered


def run_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Run mapping -> product filtering -> empty-narrative drop, in order."""
    try:
        df = map_products(df)
        df = filter_target_products(df)
        df = drop_empty_narratives(df)
    except Exception as e:
        raise RuntimeError(f"Filtering pipeline failed: {e}") from e

    logger.info(f"Filtering complete. Remaining rows: {len(df):,}")
    return df


if __name__ == "__main__":
    try:
        df_raw = load_raw_complaints(RAW_CSV_PATH)
        df_filtered = run_filtering(df_raw)
        print(df_filtered[["product_category", NARRATIVE_COL]].head())
        print(df_filtered.shape)
    except Exception as err:
        logger.error(f"Filtering run failed: {err}")