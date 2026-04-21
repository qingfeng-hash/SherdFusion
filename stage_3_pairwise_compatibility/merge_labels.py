"""Merge positive and negative pair label CSV files."""

from __future__ import annotations

import argparse

import pandas as pd


REQUIRED_COLUMNS = [
    "group_name",
    "label",
    "exterior_image",
    "interior_image",
    "exterior_path",
    "interior_path",
]


def load_and_check(path: str) -> pd.DataFrame:
    """Load one label CSV and validate its required columns."""
    dataframe = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    return dataframe[REQUIRED_COLUMNS].copy()


def main():
    """Merge positive and negative examples into one training CSV."""
    parser = argparse.ArgumentParser(description="Merge positive and negative pair label CSV files.")
    parser.add_argument("--positive_csv", type=str, required=True)
    parser.add_argument("--negative_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="all_labels.csv")
    args = parser.parse_args()

    positive_dataframe = load_and_check(args.positive_csv)
    negative_dataframe = load_and_check(args.negative_csv)

    merged_dataframe = pd.concat([positive_dataframe, negative_dataframe], axis=0, ignore_index=True)
    merged_dataframe = merged_dataframe.drop_duplicates(subset=["group_name", "exterior_path", "interior_path"])
    merged_dataframe.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"Positive samples: {len(positive_dataframe)}")
    print(f"Negative samples: {len(negative_dataframe)}")
    print(f"Merged samples: {len(merged_dataframe)}")
    print(f"Output file: {args.output_csv}")


if __name__ == "__main__":
    main()
