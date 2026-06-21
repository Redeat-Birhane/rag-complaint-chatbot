"""
Task 2 — Section 1: Stratified Sampling
Create a stratified sample of 10,000-15,000 complaints from the cleaned dataset,
with proportional representation across the 4 product categories.
"""

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered_complaints.csv"
)
OUTPUT_SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "stratified_sample.csv"
)

SAMPLE_SIZE = 12000  # within the 10,000-15,000 target range
RANDOM_STATE = 42


def load_filtered_complaints(csv_path: str) -> pd.DataFrame:
    """Load the cleaned/filtered dataset produced in Task 1."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Filtered complaints file not found at '{csv_path}'. Run Task 1 first."
        )
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load '{csv_path}': {e}") from e

    if df.empty:
        raise ValueError(f"'{csv_path}' has 0 rows.")

    logger.info(f"Loaded {len(df):,} filtered complaints.")
    return df


def stratified_sample(df: pd.DataFrame, target_col: str = "product_category",
                       sample_size: int = SAMPLE_SIZE,
                       random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Draw a stratified sample proportional to each category's share of the full dataset.
    """
    try:
        if target_col not in df.columns:
            raise KeyError(f"Column '{target_col}' not found. Columns: {list(df.columns)}")

        category_counts = df[target_col].value_counts()
        category_proportions = category_counts / category_counts.sum()
        logger.info(f"Category proportions in full dataset:\n{category_proportions}")

        sampled_parts = []
        for category, proportion in category_proportions.items():
            n_for_category = max(1, round(sample_size * proportion))
            category_df = df[df[target_col] == category]
            n_for_category = min(n_for_category, len(category_df))
            sampled_parts.append(
                category_df.sample(n=n_for_category, random_state=random_state)
            )

        sample_df = pd.concat(sampled_parts, ignore_index=True)
        sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error during stratified sampling: {e}") from e

    logger.info(f"Final stratified sample size: {len(sample_df):,} rows.")
    final_breakdown = sample_df[target_col].value_counts()
    logger.info(f"Sample breakdown by category:\n{final_breakdown}")

    return sample_df


def save_sample(df: pd.DataFrame, out_path: str) -> None:
    """Save the stratified sample to CSV."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved stratified sample ({len(df):,} rows) to '{out_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save sample to '{out_path}': {e}") from e


if __name__ == "__main__":
    try:
        df_filtered = load_filtered_complaints(INPUT_CSV_PATH)
        df_sample = stratified_sample(df_filtered)
        save_sample(df_sample, OUTPUT_SAMPLE_PATH)
    except Exception as err:
        logger.error(f"Sampling run failed: {err}")