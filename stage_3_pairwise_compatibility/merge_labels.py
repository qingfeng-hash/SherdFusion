"""Prepare and merge labeled pair CSV files for Stage 3 training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR
DEFAULT_POSITIVE_DIR = SCRIPT_DIR / "real_match"
DEFAULT_NEGATIVE_DIR = SCRIPT_DIR / "false_match"
DEFAULT_POSITIVE_CSV = SCRIPT_DIR / "positive_labels.csv"
DEFAULT_NEGATIVE_CSV = SCRIPT_DIR / "negative_labels.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "all_labels.csv"


REQUIRED_COLUMNS = [
    "group_name",
    "label",
    "exterior_image",
    "interior_image",
    "exterior_path",
    "interior_path",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_and_check(path: str | Path) -> pd.DataFrame:
    """Load one label CSV and validate its required columns."""
    dataframe = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    return dataframe[REQUIRED_COLUMNS].copy()


def to_project_relative(path: Path, project_root: Path) -> str:
    """Store relative paths when possible so training can relocate the dataset root."""
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def scan_labeled_folder(input_dir: str | Path, label: int, project_root: str | Path) -> pd.DataFrame:
    """Scan one folder and build labeled exterior/interior image pairs."""
    input_dir = Path(input_dir)
    project_root = Path(project_root)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    grouped = {}
    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = file_path.stem
        if stem.endswith("_exterior"):
            group_name = stem[: -len("_exterior")]
            grouped.setdefault(group_name, {})["exterior"] = file_path
        elif stem.endswith("_interior"):
            group_name = stem[: -len("_interior")]
            grouped.setdefault(group_name, {})["interior"] = file_path

    rows = []
    for group_name, paths in grouped.items():
        if "exterior" not in paths or "interior" not in paths:
            continue

        exterior_path = paths["exterior"]
        interior_path = paths["interior"]
        rows.append(
            {
                "group_name": group_name,
                "label": int(label),
                "exterior_image": exterior_path.name,
                "interior_image": interior_path.name,
                "exterior_path": to_project_relative(exterior_path, project_root),
                "interior_path": to_project_relative(interior_path, project_root),
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def save_optional_csv(dataframe: pd.DataFrame, output_path: str | Path | None) -> None:
    """Save an intermediate CSV only when the caller requests it."""
    if output_path is None:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    """Build positive/negative labels and export one merged training CSV."""
    parser = argparse.ArgumentParser(
        description="Prepare Stage 3 label CSV files either from folders or from existing CSV files."
    )
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "folder", "csv"])
    parser.add_argument("--positive_csv", type=str, default=str(DEFAULT_POSITIVE_CSV), help="Existing positive label CSV.")
    parser.add_argument("--negative_csv", type=str, default=str(DEFAULT_NEGATIVE_CSV), help="Existing negative label CSV.")
    parser.add_argument("--positive_dir", type=str, default=str(DEFAULT_POSITIVE_DIR), help="Folder containing positive image pairs.")
    parser.add_argument("--negative_dir", type=str, default=str(DEFAULT_NEGATIVE_DIR), help="Folder containing negative image pairs.")
    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT), help="Root directory used to store relative image paths.")
    parser.add_argument("--positive_output_csv", type=str, default=str(DEFAULT_POSITIVE_CSV), help="Optional output CSV for generated positive labels.")
    parser.add_argument("--negative_output_csv", type=str, default=str(DEFAULT_NEGATIVE_CSV), help="Optional output CSV for generated negative labels.")
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Merged output CSV for training.")
    args = parser.parse_args()

    positive_dir = Path(args.positive_dir)
    negative_dir = Path(args.negative_dir)
    positive_csv = Path(args.positive_csv)
    negative_csv = Path(args.negative_csv)

    if args.mode == "auto":
        if positive_dir.exists() and negative_dir.exists():
            mode = "folder"
        elif positive_csv.exists() and negative_csv.exists():
            mode = "csv"
        else:
            mode = "folder"
    else:
        mode = args.mode

    if mode == "folder":
        positive_dataframe = scan_labeled_folder(args.positive_dir, label=1, project_root=args.project_root)
        negative_dataframe = scan_labeled_folder(args.negative_dir, label=0, project_root=args.project_root)
        save_optional_csv(positive_dataframe, args.positive_output_csv)
        save_optional_csv(negative_dataframe, args.negative_output_csv)
    else:
        positive_dataframe = load_and_check(args.positive_csv)
        negative_dataframe = load_and_check(args.negative_csv)

    merged_dataframe = pd.concat([positive_dataframe, negative_dataframe], axis=0, ignore_index=True)
    merged_dataframe = merged_dataframe.drop_duplicates(subset=["group_name", "exterior_path", "interior_path"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged_dataframe.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"Positive samples: {len(positive_dataframe)}")
    print(f"Negative samples: {len(negative_dataframe)}")
    print(f"Merged samples: {len(merged_dataframe)}")
    print(f"Output file: {args.output_csv}")
    print(f"Preparation mode: {mode}")


if __name__ == "__main__":
    main()
