"""Dataset utilities for pairwise pottery compatibility classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


REQUIRED_COLUMNS = [
    "group_name",
    "label",
    "exterior_image",
    "interior_image",
    "exterior_path",
    "interior_path",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_path(project_root: Path, relative_or_absolute_path: str) -> Path:
    """Resolve image paths stored in CSV rows."""
    path = Path(str(relative_or_absolute_path).replace("\\", "/"))
    return path if path.is_absolute() else (project_root / path)


def load_samples_from_csv(labels_csv: str | Path, project_root: str | Path):
    """Load labeled training samples from one CSV file."""
    labels_csv = Path(labels_csv)
    project_root = Path(project_root)

    dataframe = pd.read_csv(labels_csv)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{labels_csv} is missing required columns: {missing_columns}")

    samples = []
    for row in dataframe[REQUIRED_COLUMNS].to_dict(orient="records"):
        sample = dict(row)
        sample["label"] = int(sample["label"])
        sample["exterior_path"] = str(_normalize_path(project_root, sample["exterior_path"]))
        sample["interior_path"] = str(_normalize_path(project_root, sample["interior_path"]))
        samples.append(sample)

    return samples


def scan_infer_folder(input_dir: str | Path):
    """Scan one folder for *_exterior / *_interior image pairs."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    grouped = {}
    for file_path in sorted(input_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = file_path.stem
        if stem.endswith("_exterior"):
            group_name = stem[: -len("_exterior")]
            grouped.setdefault(group_name, {})["exterior"] = file_path
        elif stem.endswith("_interior"):
            group_name = stem[: -len("_interior")]
            grouped.setdefault(group_name, {})["interior"] = file_path

    samples = []
    for group_name, paths in grouped.items():
        if "exterior" not in paths or "interior" not in paths:
            continue
        samples.append(
            {
                "group_name": group_name,
                "exterior_image": paths["exterior"].name,
                "interior_image": paths["interior"].name,
                "exterior_path": str(paths["exterior"]),
                "interior_path": str(paths["interior"]),
                "label": -1,
            }
        )

    return samples


class PotteryPairDataset(Dataset):
    """PyTorch dataset for loading exterior/interior pottery image pairs."""

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict(self.samples[index])
        exterior_image = Image.open(sample["exterior_path"]).convert("RGB")
        interior_image = Image.open(sample["interior_path"]).convert("RGB")

        if self.transform is not None:
            exterior_image = self.transform(exterior_image)
            interior_image = self.transform(interior_image)

        return {
            "exterior": exterior_image,
            "interior": interior_image,
            "label": int(sample.get("label", -1)),
            "group_name": sample["group_name"],
            "exterior_image": sample["exterior_image"],
            "interior_image": sample["interior_image"],
            "exterior_path": sample["exterior_path"],
            "interior_path": sample["interior_path"],
        }


def build_train_transform(image_size: int):
    """Build the image augmentation pipeline used during training."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int):
    """Build the deterministic transform used for validation and inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
