import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_CSV_PATH = "data/raw/complaints.csv"

USECOLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Complaint ID",
]

CHUNKSIZE = 100_000 


def load_raw_complaints(csv_path: str, usecols: list[str] = USECOLS,
                         chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw complaints file not found at '{csv_path}'.")

    chunks = []
    try:
        logger.info(f"Loading raw complaints from '{csv_path}' in chunks of {chunksize:,} rows ...")
        reader = pd.read_csv(
            csv_path,
            usecols=usecols,
            low_memory=False,
            on_bad_lines="warn",
            chunksize=chunksize,
            engine="c",
        )
        for i, chunk in enumerate(reader, start=1):
            chunks.append(chunk)
            logger.info(f"Loaded chunk {i} ({len(chunk):,} rows) — running total: {sum(len(c) for c in chunks):,}")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"The file at '{csv_path}' is empty.") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse '{csv_path}' as CSV: {e}") from e
    except MemoryError as e:
        raise RuntimeError(
            f"Out of memory while loading '{csv_path}' even in chunks. "
            f"Try lowering chunksize (currently {chunksize:,}) or freeing up RAM."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading '{csv_path}': {e}") from e

    if not chunks:
        raise ValueError(f"No data loaded from '{csv_path}'.")

    try:
        df = pd.concat(chunks, ignore_index=True)
    except MemoryError as e:
        raise RuntimeError(
            "Out of memory while concatenating chunks. The file may be too large "
            "to hold fully in memory even with reduced columns — consider processing "
            "and filtering chunk-by-chunk instead (see load_and_filter_in_chunks())."
        ) from e

    if df.empty:
        raise ValueError(f"Loaded DataFrame from '{csv_path}' has 0 rows.")

    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns total.")
    return df


if __name__ == "__main__":
    try:
        df_raw = load_raw_complaints(RAW_CSV_PATH)
        print(df_raw.head())
        print(df_raw.shape)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(e)