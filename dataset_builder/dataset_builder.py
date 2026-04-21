"""Build a single PKL dataset from gold_standard_benchmark images.

The pipeline performs the following steps for each RGBA PNG image:
1. Extract the largest alpha contour.
2. Resample and center the contour.
3. Extract local patch features along the contour with DINOv3.
4. Extract one global foreground feature with DINOv3.
5. Save everything into one pickle file.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import DINOv3_S_Encoder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "gold_standard_benchmark" / "dataset"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "gold_standard_benchmark" / "gold_standard_dataset.pkl"
DEFAULT_WEIGHT_PATH = SCRIPT_DIR / "dinov3_epoch_500.pth"

TARGET_SIZE = 224
MARGIN_RATIO = 0.08
CONTOUR_SPACING = 35
PATCH_SIZE = 48
PROJECTION_DIM = 128
VALID_IMAGE_EXTENSIONS = {".png"}

GLOBAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def signed_area_image_coords(points: np.ndarray) -> float:
    """Compute the signed polygon area in image coordinates."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Contour points must have shape (N, 2).")

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    x_next = np.roll(x_coords, -1)
    y_next = np.roll(y_coords, -1)
    return 0.5 * np.sum(x_coords * y_next - x_next * y_coords)


def enforce_contour_orientation(
    points: np.ndarray, clockwise: bool = True
) -> np.ndarray:
    """Make the contour orientation consistent across all samples."""
    points = np.asarray(points, dtype=np.float64)
    is_clockwise = signed_area_image_coords(points) > 0
    if is_clockwise != clockwise:
        return points[::-1].copy()
    return points


def resample_contour_by_spacing(
    contour: np.ndarray,
    spacing: int = CONTOUR_SPACING,
    clockwise: bool = True,
) -> np.ndarray | None:
    """Resample a contour with approximately fixed point spacing."""
    contour = contour[:, 0, :] if contour.ndim == 3 else contour
    contour = contour.astype(np.float64)

    if len(contour) < 3:
        return None

    contour = enforce_contour_orientation(contour, clockwise=clockwise)

    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_length = cumulative_lengths[-1]

    if total_length <= 0:
        return None

    target_lengths = np.arange(0, total_length, spacing, dtype=np.float64)
    if len(target_lengths) == 0:
        return None

    resampled_points = []
    segment_index = 0
    for target_length in target_lengths:
        while (
            segment_index < len(cumulative_lengths) - 2
            and cumulative_lengths[segment_index + 1] < target_length
        ):
            segment_index += 1

        start_length = cumulative_lengths[segment_index]
        end_length = cumulative_lengths[segment_index + 1]
        start_point = contour[segment_index]
        end_point = contour[segment_index + 1]

        if end_length == start_length:
            alpha = 0.0
        else:
            alpha = (target_length - start_length) / (end_length - start_length)

        resampled_points.append((1 - alpha) * start_point + alpha * end_point)

    return np.array(resampled_points, dtype=np.float32)


def center_contour(contour: np.ndarray) -> np.ndarray:
    """Shift the contour so its mean lies at the origin."""
    center = contour.mean(axis=0)
    return contour - center


def image_sort_key(image_path: Path) -> tuple[int, int, str]:
    """Sort images by numeric id and by exterior before interior."""
    match = re.search(r"(\d+)", image_path.stem)
    numeric_id = int(match.group(1)) if match else 10**9

    if "exterior" in image_path.stem:
        side_rank = 0
    elif "interior" in image_path.stem:
        side_rank = 1
    else:
        side_rank = 2

    return numeric_id, side_rank, image_path.stem


def load_rgba_image(image_path: Path) -> np.ndarray:
    """Load a PNG image as an RGBA numpy array."""
    with Image.open(image_path) as image:
        rgba_image = image.convert("RGBA")
    return np.array(rgba_image)


def find_largest_alpha_contour(image_rgba: np.ndarray) -> np.ndarray | None:
    """Extract the largest contour from the alpha mask."""
    alpha_channel = image_rgba[:, :, 3]
    alpha_mask = (alpha_channel > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        alpha_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_patch_feature(
    patch_rgb_uint8: np.ndarray,
    model: nn.Module,
    device: torch.device,
    output_type: str = "z",
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Extract one DINOv3 feature for a local RGB patch."""
    resized_patch = cv2.resize(
        patch_rgb_uint8,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )
    patch_image = Image.fromarray(resized_patch, mode="RGB")
    patch_tensor = GLOBAL_TRANSFORM(patch_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features, projected = model(patch_tensor)

    if output_type == "feat":
        return features.squeeze(0).detach().cpu().numpy()
    return projected.squeeze(0).detach().cpu().numpy()


def crop_and_pad_foreground(
    image_rgba: np.ndarray,
    target_size: int = TARGET_SIZE,
    margin_ratio: float = MARGIN_RATIO,
) -> Image.Image | None:
    """Crop the visible foreground and pad it into a square RGB canvas."""
    alpha_channel = image_rgba[:, :, 3].astype(np.float32) / 255.0
    foreground_mask = alpha_channel > 0

    y_coords, x_coords = np.where(foreground_mask)
    if len(x_coords) == 0:
        return None

    image_height, image_width = alpha_channel.shape
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    padding = int(margin_ratio * max(x_max - x_min, y_max - y_min))
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width - 1, x_max + padding)
    y_max = min(image_height - 1, y_max + padding)

    rgb_crop = image_rgba[y_min : y_max + 1, x_min : x_max + 1, :3].astype(np.float32)
    alpha_crop = alpha_channel[y_min : y_max + 1, x_min : x_max + 1]

    # Keep only the visible foreground; the background stays black.
    rgb_crop *= alpha_crop[..., None]
    crop_image = Image.fromarray(np.uint8(np.clip(rgb_crop, 0, 255)), mode="RGB")

    crop_width, crop_height = crop_image.size
    scale = min(target_size / crop_width, target_size / crop_height)
    resized_width = max(1, int(crop_width * scale))
    resized_height = max(1, int(crop_height * scale))

    resized_image = crop_image.resize((resized_width, resized_height), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_left = (target_size - resized_width) // 2
    paste_top = (target_size - resized_height) // 2
    canvas.paste(resized_image, (paste_left, paste_top))
    return canvas


def build_dinov3_model(
    weight_path: Path,
    device: torch.device,
    proj_dim: int = PROJECTION_DIM,
) -> nn.Module:
    """Load the trained DINOv3 feature extractor."""
    model = DINOv3_S_Encoder(
        weight_path=str(weight_path),
        proj_dim=proj_dim,
        train_backbone=False,
    )
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def extract_global_feature(
    image_rgba: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Extract one global feature from the full foreground object."""
    processed_image = crop_and_pad_foreground(image_rgba)
    if processed_image is None:
        raise ValueError("The image does not contain visible alpha foreground.")

    image_tensor = GLOBAL_TRANSFORM(processed_image).unsqueeze(0).to(device)
    with torch.no_grad():
        _, projected = model(image_tensor)
    return projected.squeeze(0).detach().cpu().numpy()


def extract_patch_features(
    image_rgba: np.ndarray,
    contour_points: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    output_type: str = "z",
) -> np.ndarray:
    """Extract local features from patches centered on contour points."""
    image_height, image_width = image_rgba.shape[:2]
    half_patch = patch_size // 2
    patch_features = []

    for x_coord, y_coord in contour_points:
        x_coord = int(round(x_coord))
        y_coord = int(round(y_coord))

        x_min = max(0, x_coord - half_patch)
        y_min = max(0, y_coord - half_patch)
        x_max = min(image_width, x_coord + half_patch)
        y_max = min(image_height, y_coord + half_patch)

        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

        paste_x_min = half_patch - (x_coord - x_min)
        paste_y_min = half_patch - (y_coord - y_min)
        paste_x_max = paste_x_min + (x_max - x_min)
        paste_y_max = paste_y_min + (y_max - y_min)

        rgb_region = image_rgba[y_min:y_max, x_min:x_max, :3]
        alpha_region = image_rgba[y_min:y_max, x_min:x_max, 3] > 0

        for channel in range(3):
            patch[paste_y_min:paste_y_max, paste_x_min:paste_x_max, channel][
                alpha_region
            ] = rgb_region[:, :, channel][alpha_region]

        feature = extract_patch_feature(
            patch_rgb_uint8=patch,
            model=model,
            device=device,
            output_type=output_type,
        )
        patch_features.append(feature)

    return np.vstack(patch_features)


def process_single_image(
    image_path: Path,
    model: nn.Module,
    device: torch.device,
    contour_spacing: int = CONTOUR_SPACING,
    patch_size: int = PATCH_SIZE,
) -> dict[str, Any] | None:
    """Convert one image into one dataset record."""
    image_rgba = load_rgba_image(image_path)
    contour = find_largest_alpha_contour(image_rgba)

    if contour is None:
        print(f"Skip {image_path.name}: no alpha contour found.")
        return None

    resampled_contour = resample_contour_by_spacing(contour, spacing=contour_spacing)
    if resampled_contour is None or len(resampled_contour) == 0:
        print(f"Skip {image_path.name}: contour resampling failed.")
        return None

    centered_contour = center_contour(resampled_contour).astype(np.float32)
    patch_features = extract_patch_features(
        image_rgba=image_rgba,
        contour_points=resampled_contour,
        model=model,
        device=device,
        patch_size=patch_size,
        output_type="z",
    ).astype(np.float32)
    global_feature = extract_global_feature(
        image_rgba=image_rgba,
        model=model,
        device=device,
    ).astype(np.float32)

    print(
        f"Processed {image_path.name}: "
        f"contour={centered_contour.shape}, "
        f"patch_features={patch_features.shape}, "
        f"global_feature={global_feature.shape}"
    )

    return {
        "name": image_path.stem,
        "polygon": centered_contour,
        "patch_features": patch_features,
        "global_feature": global_feature.reshape(1, -1),
        "image": image_rgba,
    }


def build_dataset(
    image_dir: Path,
    output_pkl_path: Path,
    model: nn.Module,
    device: torch.device,
    contour_spacing: int = CONTOUR_SPACING,
    patch_size: int = PATCH_SIZE,
) -> None:
    """Build one dataset PKL directly from a directory of RGBA PNG images."""
    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ],
        key=image_sort_key,
    )

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    polygons = []
    patch_features = []
    global_features = []
    images = []
    names = []

    for image_path in image_paths:
        record = process_single_image(
            image_path=image_path,
            model=model,
            device=device,
            contour_spacing=contour_spacing,
            patch_size=patch_size,
        )
        if record is None:
            continue

        polygons.append(record["polygon"])
        patch_features.append(record["patch_features"])
        global_features.append(record["global_feature"])
        images.append(record["image"])
        names.append(record["name"])

    dataset_payload = {
        "polygons": polygons,
        "patch_features": patch_features,
        "global_features": global_features,
        "images": images,
        "names": names,
        "metadata": {
            "image_dir": str(image_dir),
            "output_pkl_path": str(output_pkl_path),
            "contour_spacing": contour_spacing,
            "patch_size": patch_size,
            "feature_type": "DINOv3 projector output",
        },
    }

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl_path.open("wb") as file_handle:
        pickle.dump(dataset_payload, file_handle)

    print(f"\nSaved dataset to: {output_pkl_path}")
    print(f"Samples: {len(names)}")


def main() -> None:
    """Entry point."""
    if not DEFAULT_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DEFAULT_IMAGE_DIR}")
    if not DEFAULT_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Model weight not found: {DEFAULT_WEIGHT_PATH}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_dinov3_model(
        weight_path=DEFAULT_WEIGHT_PATH,
        device=device,
        proj_dim=PROJECTION_DIM,
    )

    build_dataset(
        image_dir=DEFAULT_IMAGE_DIR,
        output_pkl_path=DEFAULT_OUTPUT_PATH,
        model=model,
        device=device,
        contour_spacing=CONTOUR_SPACING,
        patch_size=PATCH_SIZE,
    )


if __name__ == "__main__":
    main()
