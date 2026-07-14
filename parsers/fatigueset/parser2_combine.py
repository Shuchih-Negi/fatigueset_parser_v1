"""
Parser 2: Combine all participant CSVs into final output.

Reads all 12 per-participant CSV files and concatenates them into
a single fatigueset_final.csv for EDA and model training.

Usage:
    python parser2_combine.py
    python parser2_combine.py --input-dir raw/fatigueset_participants --output-file datastore/fatigueset_final.csv
"""

import argparse
import logging
from pathlib import Path
from typing import List
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def combine_participants(
    input_dir: Path, output_file: Path, expected_subjects: int = 12
) -> None:
    """
    Combine all per-participant CSVs into one final CSV.

    Args:
        input_dir: directory containing P01.csv, P02.csv, ..., P12.csv
        output_file: path to write final combined CSV
        expected_subjects: expected number of subject files (default 12)
    """
    # Find all P*.csv files
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    participant_files = sorted(input_dir.glob("P*.csv"))
    logger.info(f"Found {len(participant_files)} participant files")

    if len(participant_files) != expected_subjects:
        logger.warning(
            f"Expected {expected_subjects} files but found {len(participant_files)}"
        )
        missing = set(range(1, expected_subjects + 1)) - set(
            int(f.stem[1:]) for f in participant_files
        )
        if missing:
            logger.warning(f"Missing subjects: {sorted(missing)}")

    # Load and concatenate
    dfs = []
    for fpath in participant_files:
        logger.info(f"Loading {fpath.name}")
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {fpath}: {e}")
            continue

    if len(dfs) == 0:
        logger.error("No participant files loaded successfully")
        sys.exit(1)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} rows from {len(dfs)} participants")

    # Check for duplicates
    duplicate_cols = ["subject_id", "session_id", "window_start_sec"]
    duplicate_mask = combined_df.duplicated(subset=duplicate_cols, keep=False)
    if duplicate_mask.any():
        logger.warning(f"Found {duplicate_mask.sum()} duplicate rows; removing")
        combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep="first")
        logger.info(f"After deduplication: {len(combined_df)} rows")

    # Write output
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Wrote {len(combined_df)} rows to {output_file}")

    # Print summary statistics
    logger.info(f"Subject distribution:")
    for subject_id in combined_df["subject_id"].unique():
        count = len(combined_df[combined_df["subject_id"] == subject_id])
        logger.info(f"  {subject_id}: {count} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all per-participant CSVs into final output."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="raw/fatigueset_participants",
        help="Directory containing P*.csv files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="datastore/fatigueset_final.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=12,
        help="Expected number of subjects",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    combine_participants(args.input_dir, args.output_file, args.expected_subjects)


if __name__ == "__main__":
    main()
