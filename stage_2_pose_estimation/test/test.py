"""Evaluate the pose estimator on the same validation split used during training."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DIR = SCRIPT_DIR.parent
TRAIN_DIR = POSE_DIR / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from critic import PolygonPackingTransformer
from sde import init_sde, pc_sampler_state
from train_pair import (
    DATASET_PATHS,
    load_checkpoint_compat,
    load_simulated_pair_dataset,
    set_seed,
    SimulatedPairDataset,
    split_positive_pairs,
)


DEFAULT_CHECKPOINT = TRAIN_DIR / "outputs" / "models" / "critic_best_checkpoint.pt"
DEFAULT_WEIGHTS = TRAIN_DIR / "critic_29.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def action_to_theta(action):
    """Convert [tx, ty, cos, sin] into a rotation angle."""
    return torch.atan2(action[3], action[2])


def rotation_matrix(theta):
    """Create a 2D rotation matrix."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ],
        dim=0,
    )


def apply_action_to_vertices(vertices, action):
    """Apply one rigid transform action to one polygon."""
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices, dtype=torch.float32, device=action.device)
    else:
        vertices = vertices.to(action.device).float()

    theta = action_to_theta(action)
    rotated = vertices @ rotation_matrix(theta).T
    return rotated + action[:2].unsqueeze(0)


def polygon_centroid(points):
    """Compute a polygon centroid, falling back to point mean for degenerate cases."""
    x = points[:, 0]
    y = points[:, 1]
    x1 = torch.roll(x, shifts=-1, dims=0)
    y1 = torch.roll(y, shifts=-1, dims=0)

    cross = x * y1 - x1 * y
    cross_sum = cross.sum()
    if torch.abs(cross_sum) < 1e-8:
        return points.mean(dim=0)

    cx = ((x + x1) * cross).sum() / (3.0 * cross_sum)
    cy = ((y + y1) * cross).sum() / (3.0 * cross_sum)
    return torch.stack([cx, cy], dim=0)


def safe_polygon_np(points):
    """Build a valid Shapely polygon from contour points."""
    polygon = Polygon(np.asarray(points, dtype=np.float32))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def polygon_iou(points_a, points_b):
    """Compute polygon IoU."""
    polygon_a = safe_polygon_np(points_a)
    polygon_b = safe_polygon_np(points_b)
    union = polygon_a.union(polygon_b).area
    if union < 1e-8:
        return 0.0
    return polygon_a.intersection(polygon_b).area / union


def angle_diff_deg(theta_a, theta_b):
    """Compute the wrapped angular difference in degrees."""
    diff = torch.abs(theta_a - theta_b)
    diff = torch.minimum(diff, 2 * torch.pi - diff)
    return (diff * 180.0 / torch.pi).item()


def apply_global_rigid_to_actions(actions, delta_theta, delta_t):
    """Align the predicted pair by applying one global rigid transform."""
    aligned = actions.clone()
    rotation = rotation_matrix(delta_theta)
    aligned[:, :2] = actions[:, :2] @ rotation.T + delta_t.unsqueeze(0)

    old_theta = torch.atan2(actions[:, 3], actions[:, 2])
    new_theta = old_theta + delta_theta
    amplitude = torch.norm(actions[:, 2:4], dim=1).clamp_min(1e-8)
    aligned[:, 2] = amplitude * torch.cos(new_theta)
    aligned[:, 3] = amplitude * torch.sin(new_theta)
    return aligned


def search_best_anchor_alignment(
    pred_anchor_poly,
    gt_anchor_poly,
    coarse_step_deg=5.0,
    fine_step_deg=0.5,
    fine_window_deg=5.0,
):
    """Find the rigid alignment that maximizes anchor IoU."""
    device = pred_anchor_poly.device
    dtype = pred_anchor_poly.dtype
    pred_center = polygon_centroid(pred_anchor_poly)
    gt_center = polygon_centroid(gt_anchor_poly)
    gt_numpy = gt_anchor_poly.detach().cpu().numpy()

    def evaluate_angle(deg):
        theta = torch.tensor(math.radians(deg), device=device, dtype=dtype)
        rotation = rotation_matrix(theta)
        delta_t = gt_center - pred_center @ rotation.T
        pred_aligned = pred_anchor_poly @ rotation.T + delta_t.unsqueeze(0)
        iou = polygon_iou(pred_aligned.detach().cpu().numpy(), gt_numpy)
        return theta, delta_t, iou

    best_deg = 0.0
    best_theta = None
    best_delta_t = None
    best_iou = -1.0

    for deg in np.arange(-180.0, 180.0 + 1e-9, coarse_step_deg):
        theta, delta_t, iou = evaluate_angle(float(deg))
        if iou > best_iou:
            best_iou = iou
            best_deg = float(deg)
            best_theta = theta
            best_delta_t = delta_t

    for deg in np.arange(best_deg - fine_window_deg, best_deg + fine_window_deg + 1e-9, fine_step_deg):
        theta, delta_t, iou = evaluate_angle(float(deg))
        if iou > best_iou:
            best_iou = iou
            best_theta = theta
            best_delta_t = delta_t

    return best_theta, best_delta_t, best_iou


def evaluate_one_alignment(pred_actions, gt_actions, graph_a, graph_b, anchor_idx):
    """Evaluate one prediction using one polygon as the alignment anchor."""
    vertices = [graph_a.poly, graph_b.poly]

    pred_anchor_poly = apply_action_to_vertices(vertices[anchor_idx], pred_actions[anchor_idx])
    gt_anchor_poly = apply_action_to_vertices(vertices[anchor_idx], gt_actions[anchor_idx])
    delta_theta, delta_t, anchor_iou = search_best_anchor_alignment(pred_anchor_poly, gt_anchor_poly)

    pred_aligned = apply_global_rigid_to_actions(pred_actions, delta_theta, delta_t)
    target_idx = 1 - anchor_idx

    pred_target_poly = apply_action_to_vertices(vertices[target_idx], pred_aligned[target_idx])
    gt_target_poly = apply_action_to_vertices(vertices[target_idx], gt_actions[target_idx])

    pred_target_center = polygon_centroid(pred_target_poly)
    gt_target_center = polygon_centroid(gt_target_poly)
    trans_err = torch.norm(pred_target_center - gt_target_center).item()

    theta_pred = action_to_theta(pred_aligned[target_idx])
    theta_gt = action_to_theta(gt_actions[target_idx])
    rot_err_deg = angle_diff_deg(theta_pred, theta_gt)

    target_iou = polygon_iou(
        pred_target_poly.detach().cpu().numpy(),
        gt_target_poly.detach().cpu().numpy(),
    )

    return {
        "anchor_idx": anchor_idx,
        "trans_err": trans_err,
        "rot_err_deg": rot_err_deg,
        "anchor_iou": anchor_iou,
        "target_iou": target_iou,
        "pred_aligned": pred_aligned.detach().cpu(),
    }


def choose_better_result(result_a, result_b):
    """Prefer higher target IoU, then lower translation and rotation errors."""
    score_a = (result_a["target_iou"], -result_a["trans_err"], -result_a["rot_err_deg"])
    score_b = (result_b["target_iou"], -result_b["trans_err"], -result_b["rot_err_deg"])
    return result_b if score_b > score_a else result_a


def sample_multiple(score_model, sde_fn, graph_a, graph_b, num_steps, n_samples):
    """Sample multiple pose predictions for one pair."""
    n_samples = max(1, int(n_samples))
    batch_a = PyGBatch.from_data_list([graph.to("cpu").clone() for graph in graph_a.to_data_list() for _ in range(n_samples)])
    batch_b = PyGBatch.from_data_list([graph.to("cpu").clone() for graph in graph_b.to_data_list() for _ in range(n_samples)])
    batch_a = batch_a.to(next(score_model.parameters()).device)
    batch_b = batch_b.to(next(score_model.parameters()).device)

    _, final_actions = pc_sampler_state(
        score_model=score_model,
        sde_fn=sde_fn,
        g1=batch_a,
        g2=batch_b,
        num_steps=num_steps,
    )
    return [final_actions[idx] for idx in range(final_actions.shape[0])]


def close_polygon(points):
    """Append the first point to close a contour for plotting."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points
    return np.concatenate([points, points[:1]], axis=0)


def draw_pose_pair(ax, vertices_a, vertices_b, actions, title):
    """Draw two transformed fragments in one subplot."""
    transformed_a = apply_action_to_vertices(vertices_a, actions[0]).detach().cpu().numpy()
    transformed_b = apply_action_to_vertices(vertices_b, actions[1]).detach().cpu().numpy()

    polygon_a = close_polygon(transformed_a)
    polygon_b = close_polygon(transformed_b)

    ax.plot(polygon_a[:, 0], polygon_a[:, 1], color="#0B6E4F", linewidth=2.0)
    ax.fill(polygon_a[:, 0], polygon_a[:, 1], color="#0B6E4F", alpha=0.22)
    ax.plot(polygon_b[:, 0], polygon_b[:, 1], color="#C75000", linewidth=2.0)
    ax.fill(polygon_b[:, 0], polygon_b[:, 1], color="#C75000", alpha=0.22)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_case_visualization(case_index, graph_a, graph_b, gt_actions, best_result, output_dir):
    """Save one GT-vs-prediction visualization for one validation pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))

    draw_pose_pair(axes[0], graph_a.poly, graph_b.poly, gt_actions, "GT")
    draw_pose_pair(axes[1], graph_a.poly, graph_b.poly, best_result["pred_aligned"], "Best Pred (aligned)")

    figure.suptitle(
        f"Case {case_index:04d} | IoU={best_result['target_iou']:.4f} | "
        f"Etrans={best_result['trans_err']:.2f} | Erot={best_result['rot_err_deg']:.2f} deg",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"case_{case_index:04d}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_training_configuration(checkpoint_path: Path):
    """Read dataset and split arguments saved during training."""
    if not checkpoint_path.exists():
        return {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    return args_dict if isinstance(args_dict, dict) else {}


def write_metrics_csv(csv_path: Path, metrics: dict):
    """Save overall evaluation metrics."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "num_cases",
                "gt_pairs",
                "recovered_pairs",
                "recall_rate",
                "AEtrans",
                "AErot",
                "Avg_IoU",
                "mean_anchor_iou",
            ]
        )
        writer.writerow(
            [
                metrics["num_cases"],
                metrics["gt_pairs"],
                metrics["recovered_pairs"],
                f"{metrics['recall_rate']:.6f}",
                "" if math.isnan(metrics["AEtrans"]) else f"{metrics['AEtrans']:.6f}",
                "" if math.isnan(metrics["AErot"]) else f"{metrics['AErot']:.6f}",
                "" if math.isnan(metrics["Avg_IoU"]) else f"{metrics['Avg_IoU']:.6f}",
                f"{metrics['mean_anchor_iou']:.6f}",
            ]
        )


def evaluate_validation_split(
    score_model,
    sde_fn,
    test_dataset,
    device,
    num_steps,
    n_samples,
    max_cases,
    trans_thresh,
    rot_thresh,
    iou_thresh,
    output_dir,
    max_visualizations,
):
    """Evaluate only the validation split rebuilt from the training configuration."""
    score_model.eval()

    num_cases = min(max_cases, len(test_dataset))
    correct = 0
    best_anchor_iou = []
    recalled_trans = []
    recalled_rot = []
    recalled_target_iou = []

    with torch.no_grad():
        for case_index in tqdm(range(num_cases), desc="Evaluating validation split"):
            graph_a, graph_b, gt_actions = test_dataset[case_index]
            gt_actions = gt_actions.to(device)

            batch_a = PyGBatch.from_data_list([graph_a]).to(device)
            batch_b = PyGBatch.from_data_list([graph_b]).to(device)

            sampled_actions = sample_multiple(
                score_model=score_model,
                sde_fn=sde_fn,
                graph_a=batch_a,
                graph_b=batch_b,
                num_steps=num_steps,
                n_samples=n_samples,
            )

            best_case_result = None
            success = False

            for pred_actions in sampled_actions:
                pred_actions = pred_actions.to(device)
                result_a = evaluate_one_alignment(pred_actions, gt_actions, graph_a, graph_b, anchor_idx=0)
                result_b = evaluate_one_alignment(pred_actions, gt_actions, graph_a, graph_b, anchor_idx=1)
                result = choose_better_result(result_a, result_b)

                if best_case_result is None:
                    best_case_result = result
                else:
                    best_case_result = choose_better_result(best_case_result, result)

                passed = (
                    result["trans_err"] < trans_thresh
                    and result["rot_err_deg"] < rot_thresh
                    and (iou_thresh <= 0.0 or result["target_iou"] >= iou_thresh)
                )
                if passed:
                    success = True

            if success:
                correct += 1
                recalled_trans.append(best_case_result["trans_err"])
                recalled_rot.append(best_case_result["rot_err_deg"])
                recalled_target_iou.append(best_case_result["target_iou"])

            best_anchor_iou.append(best_case_result["anchor_iou"])

            print(
                f"Case {case_index:04d} | success={int(success)} | anchor={best_case_result['anchor_idx']} | "
                f"trans={best_case_result['trans_err']:.4f} | rot={best_case_result['rot_err_deg']:.4f} deg | "
                f"target_iou={best_case_result['target_iou']:.4f} | anchor_iou={best_case_result['anchor_iou']:.4f}"
            )

            if case_index < max_visualizations:
                save_case_visualization(
                    case_index=case_index,
                    graph_a=graph_a,
                    graph_b=graph_b,
                    gt_actions=gt_actions.cpu(),
                    best_result=best_case_result,
                    output_dir=output_dir / "visualizations",
                )

    metrics = {
        "num_cases": num_cases,
        "gt_pairs": num_cases,
        "recovered_pairs": correct,
        "recall_rate": correct / max(num_cases, 1),
        "AEtrans": float(np.mean(recalled_trans)) if recalled_trans else float("nan"),
        "AErot": float(np.mean(recalled_rot)) if recalled_rot else float("nan"),
        "Avg_IoU": float(np.mean(recalled_target_iou)) if recalled_target_iou else float("nan"),
        "mean_anchor_iou": float(np.mean(best_anchor_iou)) if best_anchor_iou else 0.0,
    }

    print("\n==================== Final Result ====================")
    print(f"Validation cases           : {metrics['num_cases']}")
    print(f"GT pairs                   : {metrics['gt_pairs']}")
    print(f"Recovered pairs            : {metrics['recovered_pairs']}")
    print(
        f"Recall                     : "
        f"{metrics['recovered_pairs']}/{metrics['gt_pairs']} = {metrics['recall_rate'] * 100:.2f}%"
    )
    print(
        "AEtrans                    : "
        + ("N/A" if math.isnan(metrics["AEtrans"]) else f"{metrics['AEtrans']:.6f}")
    )
    print(
        "AErot                      : "
        + ("N/A" if math.isnan(metrics["AErot"]) else f"{metrics['AErot']:.6f} deg")
    )
    print(
        "Avg. IoU                   : "
        + ("N/A" if math.isnan(metrics["Avg_IoU"]) else f"{metrics['Avg_IoU']:.6f}")
    )
    print(f"Mean anchor IoU            : {metrics['mean_anchor_iou']:.6f}")
    print("=====================================================\n")

    write_metrics_csv(output_dir / "test_metrics.csv", metrics)
    return metrics


def build_arg_parser():
    """Create the evaluation CLI."""
    parser = argparse.ArgumentParser(description="Evaluate Pose_Estimation on the training validation split.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--sde-mode", type=str, default=None)
    parser.add_argument("--num-steps-sample", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-test-cases", type=int, default=50)
    parser.add_argument("--max-visualizations", type=int, default=20)
    parser.add_argument("--trans-thresh", type=float, default=50.0)
    parser.add_argument("--rot-thresh", type=float, default=10.0)
    parser.add_argument("--iou-thresh", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main():
    """Run evaluation on the same validation split used during training."""
    parser = build_arg_parser()
    args = parser.parse_args()

    saved_args = load_training_configuration(args.checkpoint)
    split_seed = args.seed if args.seed is not None else int(saved_args.get("seed", 3407))
    test_ratio = args.test_ratio if args.test_ratio is not None else float(saved_args.get("test_ratio", 0.1))
    sde_mode = args.sde_mode or saved_args.get("sde_mode", "ve")

    for dataset_path in DATASET_PATHS:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    set_seed(split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Datasets:")
    for dataset_path in DATASET_PATHS:
        print(f"  - {dataset_path}")
    print(f"Split seed: {split_seed}")
    print(f"Test ratio: {test_ratio}")

    dataset_dict = load_simulated_pair_dataset(DATASET_PATHS)
    _, val_pairs = split_positive_pairs(
        dataset_dict["positive_pair_indices"],
        test_ratio=test_ratio,
        seed=split_seed,
    )
    test_dataset = SimulatedPairDataset(
        dataset_dict["all_polys"],
        dataset_dict["all_local_features"],
        dataset_dict["all_global_features"],
        dataset_dict["all_actions"],
        val_pairs,
    )

    print(f"Validation pairs reconstructed from training split: {len(test_dataset)}")

    _, marginal_prob_fn, sde_fn, _ = init_sde(sde_mode)
    score_model = PolygonPackingTransformer(
        marginal_prob_std_func=marginal_prob_fn,
        device=device,
    ).to(device)

    if args.weights.exists():
        load_checkpoint_compat(score_model, args.weights)
    elif args.checkpoint.exists():
        load_checkpoint_compat(score_model, args.checkpoint)
    else:
        raise FileNotFoundError(
            f"Neither weights nor checkpoint was found: {args.weights}, {args.checkpoint}"
        )

    evaluate_validation_split(
        score_model=score_model,
        sde_fn=sde_fn,
        test_dataset=test_dataset,
        device=device,
        num_steps=args.num_steps_sample,
        n_samples=args.n_samples,
        max_cases=args.max_test_cases,
        trans_thresh=args.trans_thresh,
        rot_thresh=args.rot_thresh,
        iou_thresh=args.iou_thresh,
        output_dir=args.output_dir,
        max_visualizations=args.max_visualizations,
    )


if __name__ == "__main__":
    main()
