"""
Section 5: Task 1 — Full Pipeline Runner
Loads raw data -> runs EDA -> filters to target products -> drops empty
narratives -> cleans text -> saves data/filtered_complaints.csv
"""

import logging
import sys
import os

from task1_load_data import load_raw_complaints, RAW_CSV_PATH
from task1_eda import run_eda
from task1_filter import run_filtering
from task1_clean import clean_narratives

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Resolve path relative to this script's own location, not the current working directory
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered_complaints.csv")


def main() -> int:
    """Run the full Task 1 pipeline end to end. Returns exit code (0=success, 1=failure)."""

    # 1. Load raw data
    try:
        df_raw = load_raw_complaints(RAW_CSV_PATH)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to load raw data: {e}")
        return 1

    # 2. Run EDA on the raw data (non-fatal if a sub-step fails)
    try:
        eda_summary = run_eda(df_raw)
        logger.info(f"EDA summary: {eda_summary}")
    except Exception as e:
        logger.error(f"EDA step encountered an unexpected error (continuing): {e}")

    # 3. Filter to target products + drop empty narratives
    try:
        df_filtered = run_filtering(df_raw)
    except RuntimeError as e:
        logger.error(f"Filtering failed: {e}")
        return 1

    # 4. Clean narrative text
    try:
        df_clean = clean_narratives(df_filtered)
    except RuntimeError as e:
        logger.error(f"Cleaning failed: {e}")
        return 1

    # 5. Save final cleaned + filtered dataset
    try:
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        df_clean.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info(f"Saved cleaned dataset ({len(df_clean):,} rows) to '{OUTPUT_CSV_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to save cleaned dataset: {e}")
        return 1

    logger.info("Task 1 pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())