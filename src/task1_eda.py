"""
Section 2: Exploratory Data Analysis on CFPB Complaints
- Distribution of complaints across products
- Narrative word-count distribution (visualized)
- Counts of complaints with vs. without narratives
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from task1_load_data import load_raw_complaints, RAW_CSV_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"
PROCESSED_DIR = "data/processed"


def product_distribution(df: pd.DataFrame) -> pd.Series:
    """Count complaints per product category."""
    try:
        counts = df[PRODUCT_COL].value_counts()
    except KeyError as e:
        raise KeyError(f"Column '{PRODUCT_COL}' not found. Columns: {list(df.columns)}") from e
    except Exception as e:
        raise RuntimeError(f"Error computing product distribution: {e}") from e

    logger.info(f"Found {len(counts)} unique product categories.")
    return counts


def narrative_presence_counts(df: pd.DataFrame) -> dict:
    """Count complaints with vs. without a narrative."""
    try:
        has_narrative = df[NARRATIVE_COL].notna() & (df[NARRATIVE_COL].astype(str).str.strip() != "")
        with_count = int(has_narrative.sum())
        without_count = int(len(df) - with_count)
    except KeyError as e:
        raise KeyError(f"Column '{NARRATIVE_COL}' not found. Columns: {list(df.columns)}") from e
    except Exception as e:
        raise RuntimeError(f"Error counting narrative presence: {e}") from e

    result = {"with_narrative": with_count, "without_narrative": without_count, "total": int(len(df))}
    logger.info(f"Narratives present: {with_count:,} | missing: {without_count:,} | total: {len(df):,}")
    return result


def narrative_word_counts(df: pd.DataFrame) -> pd.Series:
    """Compute word count for each non-null narrative (vectorized)."""
    try:
        narratives = df[NARRATIVE_COL].dropna()
        # str.count of whitespace-separated tokens via vectorized regex — faster than .split().str.len() at scale
        word_counts = narratives.str.count(r'\S+')
    except KeyError as e:
        raise KeyError(f"Column '{NARRATIVE_COL}' not found. Columns: {list(df.columns)}") from e
    except Exception as e:
        raise RuntimeError(f"Error computing word counts: {e}") from e

    logger.info(
        f"Word count — min: {word_counts.min()}, max: {word_counts.max()}, "
        f"mean: {word_counts.mean():.1f}, median: {word_counts.median():.1f}"
    )
    return word_counts


def plot_product_distribution(counts: pd.Series, out_path: str) -> None:
    """Save a bar chart of complaint counts per product."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.head(20).plot(kind="bar", ax=ax, color="#3b6ea5")
        ax.set_title("Complaint Count by Product (Top 20)")
        ax.set_xlabel("Product")
        ax.set_ylabel("Number of Complaints")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved product distribution plot to '{out_path}'.")
    except Exception as e:
        logger.error(f"Failed to save product distribution plot: {e}")


def plot_word_count_histogram(word_counts: pd.Series, out_path: str, max_words: int = 1000) -> None:
    """Save a histogram of narrative word counts (clipped for readability)."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clipped = word_counts[word_counts <= max_words]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(clipped, bins=60, color="#5a9367", edgecolor="black", alpha=0.8)
        ax.set_title(f"Distribution of Narrative Word Counts (clipped at {max_words} words)")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Number of Complaints")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved word count histogram to '{out_path}'.")
    except Exception as e:
        logger.error(f"Failed to save word count histogram: {e}")


def run_eda(df: pd.DataFrame, output_dir: str = PROCESSED_DIR) -> dict:
    """Run the full EDA suite and return a summary dict."""
    summary = {}

    try:
        counts = product_distribution(df)
        summary["product_counts"] = counts
        plot_product_distribution(counts, os.path.join(output_dir, "eda_product_distribution.png"))
    except Exception as e:
        logger.error(f"Product distribution step failed: {e}")
        summary["product_counts"] = None

    try:
        summary["narrative_counts"] = narrative_presence_counts(df)
    except Exception as e:
        logger.error(f"Narrative presence step failed: {e}")
        summary["narrative_counts"] = None

    try:
        wc = narrative_word_counts(df)
        summary["word_count_stats"] = {
            "min": int(wc.min()), "max": int(wc.max()),
            "mean": float(wc.mean()), "median": float(wc.median()),
        }
        plot_word_count_histogram(wc, os.path.join(output_dir, "eda_word_count_histogram.png"))

        very_short = wc[wc <= 5]
        very_long = wc[wc >= 1000]
        logger.info(f"Very short narratives (<=5 words): {len(very_short):,}")
        logger.info(f"Very long narratives (>=1000 words): {len(very_long):,}")
        summary["very_short_count"] = int(len(very_short))
        summary["very_long_count"] = int(len(very_long))
    except Exception as e:
        logger.error(f"Word count step failed: {e}")
        summary["word_count_stats"] = None

    return summary


if __name__ == "__main__":
    try:
        df_raw = load_raw_complaints(RAW_CSV_PATH)
        results = run_eda(df_raw)
        print(results)
    except Exception as err:
        logger.error(f"EDA run failed: {err}")