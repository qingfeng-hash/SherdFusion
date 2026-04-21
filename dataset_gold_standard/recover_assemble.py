"""Reconstruct paired sherd assemblies from stored transformation records."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ALPHA_THRESHOLD = 0
ALPHA_CROP_PADDING = 10


def parse_action(action: list[float]) -> tuple[float, float, float]:
    """Convert [dx, dy, cos(theta), sin(theta)] into translation and angle."""
    dx, dy, cos_theta, sin_theta = action
    theta = math.atan2(sin_theta, cos_theta)
    return float(dx), float(dy), float(theta)


def compute_alpha_weighted_centroid(image: Image.Image) -> tuple[float, float]:
    """Compute the centroid using the alpha channel as pixel weights."""
    image_array = np.array(image, dtype=np.uint8)
    alpha_channel = image_array[:, :, 3].astype(np.float64)
    alpha_sum = alpha_channel.sum()

    if alpha_sum < 1e-9:
        return image.width / 2.0, image.height / 2.0

    y_coords, x_coords = np.indices(alpha_channel.shape)
    center_x = (x_coords * alpha_channel).sum() / alpha_sum
    center_y = (y_coords * alpha_channel).sum() / alpha_sum
    return float(center_x), float(center_y)


def recenter_image_to_centroid(image: Image.Image) -> Image.Image:
    """Pad the image so its alpha-weighted centroid sits at the canvas center."""
    center_x, center_y = compute_alpha_weighted_centroid(image)
    image_width, image_height = image.size

    left_extent = center_x
    right_extent = image_width - center_x
    top_extent = center_y
    bottom_extent = image_height - center_y

    canvas_width = int(math.ceil(2 * max(left_extent, right_extent)))
    canvas_height = int(math.ceil(2 * max(top_extent, bottom_extent)))

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    paste_x = int(round(canvas_width / 2 - center_x))
    paste_y = int(round(canvas_height / 2 - center_y))
    canvas.paste(image, (paste_x, paste_y))
    return canvas


def find_alpha_bbox(
    image: Image.Image, alpha_threshold: int = ALPHA_THRESHOLD
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-transparent pixels."""
    image_array = np.array(image, dtype=np.uint8)
    alpha_channel = image_array[:, :, 3]
    y_coords, x_coords = np.where(alpha_channel > alpha_threshold)

    if x_coords.size == 0 or y_coords.size == 0:
        return None

    return (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max()) + 1,
        int(y_coords.max()) + 1,
    )


def crop_to_alpha_bbox(
    image: Image.Image,
    padding: int = ALPHA_CROP_PADDING,
    alpha_threshold: int = ALPHA_THRESHOLD,
) -> Image.Image:
    """Crop the canvas to the visible alpha region with optional padding."""
    alpha_bbox = find_alpha_bbox(image, alpha_threshold=alpha_threshold)
    if alpha_bbox is None:
        return image

    left, top, right, bottom = alpha_bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    return image.crop((left, top, right, bottom))


def make_output_name(image_a_path: str, image_b_path: str) -> str:
    """Create a readable output filename for a sherd pair."""
    image_a_name = Path(image_a_path).name
    image_b_name = Path(image_b_path).name

    for suffix in ("_exterior.png", "_interior.png", ".png"):
        image_a_name = image_a_name.replace(suffix, "")
        image_b_name = image_b_name.replace(suffix, "")

    return f"{image_a_name}_vs_{image_b_name}.png"


def load_rgba_image(image_path: Path) -> Image.Image:
    """Load an image as RGBA and recentre it around the alpha centroid."""
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    with Image.open(image_path) as image:
        rgba_image = image.convert("RGBA")

    return recenter_image_to_centroid(rgba_image)


def rotate_image(image: Image.Image, theta_radians: float) -> Image.Image:
    """Rotate an RGBA image using a transparent expanded canvas."""
    return image.rotate(
        math.degrees(theta_radians),
        expand=True,
        resample=Image.BICUBIC,
    )


def compute_top_left(
    center_x: float, center_y: float, image: Image.Image
) -> tuple[float, float]:
    """Convert a placement center into top-left image coordinates."""
    return center_x - image.width / 2.0, center_y - image.height / 2.0


def compute_canvas_bounds(
    image_a_top_left: tuple[float, float],
    image_b_top_left: tuple[float, float],
    image_a: Image.Image,
    image_b: Image.Image,
) -> tuple[float, float, float, float]:
    """Compute the shared canvas bounds that can contain both images."""
    min_x = min(image_a_top_left[0], image_b_top_left[0])
    min_y = min(image_a_top_left[1], image_b_top_left[1])
    max_x = max(image_a_top_left[0] + image_a.width, image_b_top_left[0] + image_b.width)
    max_y = max(
        image_a_top_left[1] + image_a.height,
        image_b_top_left[1] + image_b.height,
    )
    return min_x, min_y, max_x, max_y


def paste_on_canvas(
    canvas: Image.Image,
    image: Image.Image,
    top_left: tuple[float, float],
    min_x: float,
    min_y: float,
) -> None:
    """Paste an RGBA image onto the shared canvas using alpha blending."""
    paste_x = int(round(top_left[0] - min_x))
    paste_y = int(round(top_left[1] - min_y))
    canvas.paste(image, (paste_x, paste_y), image)


def render_record(record: dict[str, Any], dataset_dir: Path) -> Image.Image:
    """Render a single assembled pair from one transformation record."""
    ax, ay, angle_a = parse_action(record["A_action"])
    bx, by, angle_b = parse_action(record["B_action"])

    image_a = load_rgba_image(dataset_dir / record["img_a"])
    image_b = load_rgba_image(dataset_dir / record["img_b"])

    rotated_a = rotate_image(image_a, angle_a)
    rotated_b = rotate_image(image_b, angle_b)

    image_a_top_left = compute_top_left(ax, ay, rotated_a)
    image_b_top_left = compute_top_left(bx, by, rotated_b)

    min_x, min_y, max_x, max_y = compute_canvas_bounds(
        image_a_top_left,
        image_b_top_left,
        rotated_a,
        rotated_b,
    )

    canvas_width = int(math.ceil(max_x - min_x))
    canvas_height = int(math.ceil(max_y - min_y))
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    paste_on_canvas(canvas, rotated_a, image_a_top_left, min_x, min_y)
    paste_on_canvas(canvas, rotated_b, image_b_top_left, min_x, min_y)
    return crop_to_alpha_bbox(canvas)


def main() -> None:
    """Generate assembled pair images for all records in the benchmark JSON."""
    script_dir = Path(__file__).resolve().parent
    ground_truth_json = script_dir / "gt_all.json"
    dataset_dir = script_dir / "dataset"
    output_dir = script_dir / "assemble"

    output_dir.mkdir(parents=True, exist_ok=True)

    with ground_truth_json.open("r", encoding="utf-8") as file_handle:
        records = json.load(file_handle)

    success_count = 0
    failure_count = 0

    for record in records:
        try:
            assembled_image = render_record(record, dataset_dir)
            output_path = output_dir / make_output_name(record["img_a"], record["img_b"])
            assembled_image.save(output_path)
            success_count += 1
            print(f"OK   pair={record.get('pair_id')} -> {output_path.name}")
        except Exception as error:
            failure_count += 1
            print(f"FAIL pair={record.get('pair_id')} | {error!r}")

    print(f"Done. OK={success_count}, FAIL={failure_count}")
    print(f"Input JSON : {ground_truth_json}")
    print(f"Output dir : {output_dir}")


if __name__ == "__main__":
    main()
