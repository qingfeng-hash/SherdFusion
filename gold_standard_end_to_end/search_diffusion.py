"""Diffusion-based polygon alignment and evaluation utilities."""

import csv
import json
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from torch_geometric.data import Batch, Data

from critic import PolygonPackingTransformer
from sde import init_sde, pc_sampler_state

IOU_SUCCESS_THRESHOLD = 0.7
ETRANS_SUCCESS_THRESHOLD = 50.0
EROT_SUCCESS_THRESHOLD = 10.0
DETAILED_RESULTS_FILENAME = "evaluation_metrics_thresholded.csv"
SUMMARY_RESULTS_FILENAME = "evaluation_summary_thresholded.csv"

def append_metrics_to_csv(
    csv_path,
    shard_a,
    shard_b,
    sample_idx,
    iou_B,
    disp_B,
    angle_error_deg_B,
    passes_thresholds,
    is_selected,
):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "shard_a",
                "shard_b",
                "sample_idx",
                "iou_B",
                "disp_B",
                "angle_error_deg_B",
                "passes_thresholds",
                "is_selected",
            ])

        writer.writerow([
            shard_a,
            shard_b,
            sample_idx,
            f"{iou_B:.6f}",
            f"{disp_B:.6f}",
            f"{angle_error_deg_B:.6f}",
            int(passes_thresholds),
            int(is_selected),
        ])


def append_best_metrics_to_csv(
    csv_path,
    shard_a,
    shard_b,
    is_recalled,
    best_sample_idx,
    best_iou_B,
    best_disp_B,
    best_angle_error_deg_B,
):
    """Append one summary row for the best sample of a shard pair."""
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "shard_a",
                "shard_b",
                "is_recalled",
                "best_sample_idx",
                "best_iou_B",
                "best_disp_B",
                "best_angle_error_deg_B",
            ])

        writer.writerow([
            shard_a,
            shard_b,
            int(is_recalled),
            "" if best_sample_idx is None else best_sample_idx,
            "" if best_iou_B is None else f"{best_iou_B:.6f}",
            "" if best_disp_B is None else f"{best_disp_B:.6f}",
            "" if best_angle_error_deg_B is None else f"{best_angle_error_deg_B:.6f}",
        ])


def is_successful_prediction(
    iou_B,
    disp_B,
    angle_error_deg_B,
    iou_threshold=IOU_SUCCESS_THRESHOLD,
    etrans_threshold=ETRANS_SUCCESS_THRESHOLD,
    erot_threshold=EROT_SUCCESS_THRESHOLD,
):
    """Check whether a prediction satisfies the success thresholds."""
    return (
        iou_B >= iou_threshold
        and disp_B <= etrans_threshold
        and angle_error_deg_B <= erot_threshold
    )

def pair_already_processed(csv_path, shard_a, shard_b):
    """
    Check whether the pair already exists in the CSV file.
    """
    if not os.path.exists(csv_path):
        return False

    key = tuple(sorted([shard_a, shard_b]))

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row["shard_a"]
            b = row["shard_b"]
            if tuple(sorted([a, b])) == key:
                return True

    return False

# ================================================================
#   Setup
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, marginal_prob_fn, sde_fn, _ = init_sde("ve")

diff_model = PolygonPackingTransformer(
    marginal_prob_std_func=marginal_prob_fn, device=device
).to(device)
diff_model.load_state_dict(torch.load("./critic.pth", map_location="cpu"), strict=False)
diff_model.eval()

GT_JSON_PATH = "gt.json"


# ================================================================
#   Geometry helpers
# ================================================================
def compute_iou(polyA, polyB):
    """Compute polygon IoU with a safe fallback for invalid shapes."""
    pA, pB = Polygon(polyA), Polygon(polyB)
    if not pA.is_valid or not pB.is_valid:
        return 0
    inter = pA.intersection(pB).area
    union = pA.union(pB).area
    return inter / max(union, 1e-6)


def calculate_angle(p1, p2, p3):
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot = a[0] * b[0] + a[1] * b[1]
    cross = a[0] * b[1] - a[1] * b[0]
    ang = math.atan2(cross, dot)
    ang = abs(ang) * (180 / math.pi)
    if cross < 0:
        ang = 360 - ang
    return ang


def compute_node_features(poly):
    feats = []
    for i in range(len(poly)):
        p1, p2, p3 = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
        angle = calculate_angle(p1, p2, p3)
        feats.append([p2[0], p2[1], angle])
    return feats


def compute_global_features(poly):
    polygon = Polygon(poly)
    return [polygon.area, polygon.length]


# ================================================================
#   Graph construction
# ================================================================
def build_single_graph(coords, gabor_feat, global_feat):
    """Build a PyG graph with the same attributes used during training."""
    coords = np.asarray(coords, dtype=np.float32)
    node_feats = compute_node_features(coords)
    area, perimeter = compute_global_features(coords)

    n = len(coords)
    edges = [(j, (j + 1) % n) for j in range(n)]
    edges += [((j + 1) % n, j) for j in range(n)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        f=torch.tensor(np.asarray(gabor_feat), dtype=torch.float),
        g=torch.tensor(np.asarray(global_feat), dtype=torch.float),
        edge_index=edge_index,
        area=torch.tensor(area, dtype=torch.float),
        perm=torch.tensor(perimeter, dtype=torch.float),
        poly=torch.tensor(coords, dtype=torch.float),
        action=torch.zeros(4, dtype=torch.float)
    )
    return data


def make_pair_batches(query_coords, target_coords, q_feat, t_feat, q_global_feat, t_global_feat, device):
    g1 = build_single_graph(query_coords, q_feat, q_global_feat)
    g2 = build_single_graph(target_coords, t_feat, t_global_feat)
    bg1 = Batch.from_data_list([g1]).to(device)
    bg2 = Batch.from_data_list([g2]).to(device)
    return bg1, bg2


# ================================================================
#   RGBA visualization helpers
# ================================================================
def extract_foreground_mask_and_center(img):
    alpha = img[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return np.array([0, 0]), np.zeros_like(alpha, bool)
    coords = np.stack([xs, ys], 1).astype(np.float32)
    center = coords.mean(0)
    mask = alpha > 0
    return center, mask


def build_affine_matrix(R, t, center_src, center_dst):
    M = np.zeros((2,3), np.float32)
    M[:, :2] = R
    M[:, 2] = t + center_dst - (R @ center_src)
    return M


def rotate_to_R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], np.float32)


def visualize_predicted_images_from_arrays(poly_ids, predict, mask, images, fig_name, save_dir="vis_test"):
    """Render transformed RGBA shards into a single preview image."""

    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    transformed_images = []
    all_coords = []

    for i, pid in enumerate(poly_ids):
        if mask[i] < 0:
            continue
        img = np.array(images[pid])
        if img.ndim != 3 or img.shape[2] != 4:
            continue

        center, mask_fg = extract_foreground_mask_and_center(img)
        if not np.any(mask_fg):
            continue

        theta = np.arctan2(predict[i, 3], predict[i, 2])
        R = rotate_to_R(theta)
        t = predict[i, :2]
        M = build_affine_matrix(R, t, center, np.zeros(2))

        ys, xs = np.where(mask_fg)
        coords = np.stack([xs, ys], 1).astype(np.float32)
        coords_trans = (coords - center) @ R.T + t
        all_coords.append(coords_trans)

        transformed_images.append((img, M, R, t, center))

    all_coords = np.vstack(all_coords)
    min_xy = all_coords.min(0)
    max_xy = all_coords.max(0)
    pad = 20
    out_w = int(max_xy[0] - min_xy[0]) + 2 * pad
    out_h = int(max_xy[1] - min_xy[1]) + 2 * pad

    canvas = np.zeros((out_h, out_w, 4), np.uint8)

    for img, M, R, t, center in transformed_images:
        M_shift = M.copy()
        M_shift[:, 2] += pad - min_xy
        warped = cv2.warpAffine(img, M_shift, (out_w, out_h), borderValue=(0,0,0,0))

        alpha = warped[:,:,3:4] / 255.
        canvas[:,:,:3] = canvas[:,:,:3]*(1-alpha) + warped[:,:,:3]*alpha
        canvas[:,:,3] = np.clip(canvas[:,:,3] + warped[:,:,3], 0,255)

    Image.fromarray(canvas).save(os.path.join(save_dir, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")


# ================================================================
#   Diffusion inference for the exterior side
# ================================================================
@torch.no_grad()
def run_diffusion_refinement(
    query_coords, target_coords,
    query_feat, target_feat,
    query_global_feat, target_global_feat,
    query_image=None, target_image=None,
    save_tag="exterior_align",
    save_dir="./vis_diffusion",
    num_steps=256
):
    """Run score-based sampling for one exterior shard pair."""

    os.makedirs(save_dir, exist_ok=True)

    bg1, bg2 = make_pair_batches(
        query_coords,
        target_coords,
        query_feat,
        target_feat,
        query_global_feat,
        target_global_feat,
        device,
    )

    _, final = pc_sampler_state(diff_model, sde_fn, bg1, bg2, num_steps)
    pred = final[0]  # [2,4]

    # ---- Visualization ----
    if query_image is not None and target_image is not None:
        visualize_predicted_images_from_arrays(
            poly_ids=[0, 1],
            predict=pred,
            mask=torch.zeros(2),
            images=[query_image, target_image],
            fig_name=save_tag,
            save_dir=save_dir
        )

    return pred   # [2,4]


# ================================================================
#   Interior alignment derived from exterior predictions
# ================================================================
def _apply_pred_to_polygon(coords, pred_row, neg_theta=False):
    coords = np.asarray(coords, dtype=np.float32)
    t = pred_row[:2]
    d = pred_row[2:]

    theta = math.atan2(float(d[1]), float(d[0]) + 1e-8)
    if neg_theta:
        theta = -theta

    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], np.float32)
    return coords @ R.T + t, theta, t, R



def _mirror_polygon(poly):
    Fy = np.array([[1, 0], [0, -1]], np.float32)
    return np.asarray(poly) @ Fy.T


def _search_best_rotation(poly_src_mirror, coords_other, follow_center, step=2):
    coords_other = np.asarray(coords_other, float)
    follow_center = np.asarray(follow_center, float)

    center0 = coords_other.mean(0)

    best_iou, best_theta = -1, 0
    for deg in np.arange(-180, 181, step):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]], float)
        poly_rot = (coords_other - center0) @ R.T + follow_center
        iou = compute_iou(poly_rot, poly_src_mirror)
        if iou > best_iou:
            best_iou, best_theta = iou, th

    return best_theta


def follow_interior_by_exterior(pred_ext, query_ext, target_ext, query_int, target_int):
    """
    Args:
        pred_ext: diffusion prediction for the exterior pair, shape [2, 4].
        query_ext, target_ext: exterior polygons.
        query_int, target_int: interior polygons.

    Returns:
        Interior pose predictions with shape [2, 4].
    """
    pred_int = np.zeros_like(pred_ext, float)

    ext_list = [query_ext, target_ext]
    int_list = [query_int, target_int]

    for i in range(2):
        ext_coords = np.asarray(ext_list[i])
        int_coords = np.asarray(int_list[i])

        # Exterior polygon in world coordinates after alignment.
        poly_ext_world, _, _, _ = _apply_pred_to_polygon(ext_coords, pred_ext[i])

        # Mirror it to derive the interior alignment center.
        poly_ext_mirror = _mirror_polygon(poly_ext_world)
        follow_center = poly_ext_mirror.mean(0)

        # Search for the best interior rotation.
        best_theta = _search_best_rotation(poly_ext_mirror, int_coords, follow_center)

        pred_int[i, :2] = follow_center
        pred_int[i, 2:] = [math.cos(best_theta), math.sin(best_theta)]

    return pred_int


def run_diffusion_repeat_5_times(
    query_coords_ext, target_coords_ext,
    query_feat_ext, target_feat_ext,
    query_img_ext, target_img_ext,
    query_global_ext, target_global_ext,
    query_coords_int, target_coords_int,
    query_feat_int, target_feat_int,
    query_img_int, target_img_int,
    query_global_int, target_global_int,
    save_tag="pair_result",
    save_dir="./vis_diffusion",

):
    all_results = []

    num_repeat = 5

    for k in range(num_repeat):

        print(f"\n[Diffusion] Running sample {k} ...")

        result = run_diffusion_with_bidirectional_alignment(
            query_coords_ext=query_coords_ext,
            target_coords_ext=target_coords_ext,
            query_feat_ext=query_feat_ext,
            target_feat_ext=target_feat_ext,
            query_img_ext=query_img_ext,
            target_img_ext=target_img_ext,
            query_global_ext=query_global_ext,
            target_global_ext=target_global_ext,
            query_coords_int=query_coords_int,
            target_coords_int=target_coords_int,
            query_feat_int=query_feat_int,
            target_feat_int=target_feat_int,
            query_img_int=query_img_int,
            target_img_int=target_img_int,
            query_global_int=query_global_int,
            target_global_int=target_global_int,
            save_tag=f"{save_tag}_sample{k}",
            save_dir=save_dir
        )

        all_results.append(result)

    return all_results


# ================================================================
#   Main pipeline: exterior diffusion + interior follow-up
# ================================================================
def run_diffusion_with_bidirectional_alignment(
    query_coords_ext, target_coords_ext,
    query_feat_ext, target_feat_ext,
    query_img_ext, target_img_ext,
    query_global_ext, target_global_ext,
    query_coords_int, target_coords_int,
    query_feat_int, target_feat_int,
    query_img_int, target_img_int,
    query_global_int, target_global_int,
    save_tag="pair_result",
    save_dir="./vis_diffusion"
):
    """
    Exterior poses are predicted with diffusion.
    Interior poses follow the mirrored exterior alignment.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Kept in the signature for dataset symmetry and future extensions.
    _ = query_global_int, target_global_int

    pred_ext = run_diffusion_refinement(
        query_coords_ext, target_coords_ext,
        query_feat_ext, target_feat_ext,
        query_global_ext, target_global_ext,
        query_img_ext, target_img_ext,
        save_tag=save_tag + "_exterior",
        save_dir=save_dir
    )
    pred_ext = pred_ext.detach().cpu().numpy()

    pred_int = follow_interior_by_exterior(
        pred_ext=pred_ext,
        query_ext=query_coords_ext,
        target_ext=target_coords_ext,
        query_int=query_coords_int,
        target_int=target_coords_int
    )

    visualize_predicted_images_from_arrays(
        poly_ids=[0, 1],
        predict=pred_int,
        mask=torch.zeros(2),
        images=[query_img_int, target_img_int],
        fig_name=save_tag + "_interior",
        save_dir=save_dir
    )

    return {
        "pred_exterior": pred_ext,
        "pred_interior": pred_int
    }


# ================================================================
#   GT alignment and evaluation
# ================================================================
def action_to_theta(action4, neg_theta=False):
    """
    action4: [dx, dy, c, s] -> (dx, dy, theta)
    neg_theta=True: theta = -atan2(s, c), used for GT annotations.
    """
    dx, dy, c, s = action4
    theta = math.atan2(float(s), float(c))
    if neg_theta:
        theta = -theta
    return float(dx), float(dy), float(theta)



def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def align_pred_actions_to_gt_by_A(pred_A, pred_B, gt_A):
    """
    Align the predicted pair to GT using shard A as the anchor.
    """
    pred_A = np.asarray(pred_A, np.float32)
    pred_B = np.asarray(pred_B, np.float32)
    gt_A   = np.asarray(gt_A,   np.float32)

    px, py, pth = action_to_theta(pred_A, neg_theta=False)
    gx, gy, gth = action_to_theta(gt_A, neg_theta=True)

    dth = gth - pth
    R = rot2(dth)

    pA = np.array([px, py], np.float32)
    gA = np.array([gx, gy], np.float32)

    t_delta = gA - (R @ pA)

    def apply(action):
        x, y, th = action_to_theta(action, neg_theta=False)
        p = np.array([x, y], np.float32)
        p2 = (R @ p) + t_delta
        th2 = th + dth
        return np.array([p2[0], p2[1], math.cos(th2), math.sin(th2)], np.float32)

    pred_A_aligned = apply(pred_A)
    pred_B_aligned = apply(pred_B)
    return pred_A_aligned, pred_B_aligned



def _strip_name_to_shard_id(img_name: str):
    """
    Convert filenames such as ``JDX-16_exterior.png`` to ``JDX-16``.
    """
    base = os.path.basename(img_name)
    base = base.replace("_exterior.png", "").replace("_interior.png", "").replace(".png", "")
    return base


def load_gt_pose_by_names(shard_a: str, shard_b: str, json_path=GT_JSON_PATH):
    """
    Look up the GT pose pair by shard names.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rec in data:
        if rec.get("side", "exterior") != "exterior":
            continue
        img_a = rec["img_a"]
        img_b = rec["img_b"]
        a_id = _strip_name_to_shard_id(img_a)
        b_id = _strip_name_to_shard_id(img_b)
        if a_id == shard_a and b_id == shard_b:
            A_action = np.asarray(rec["A_action"], np.float32)
            B_action = np.asarray(rec["B_action"], np.float32)
            return A_action, B_action

    raise ValueError(f"Could not find GT poses for {shard_a} and {shard_b}.")


def metrics_for_one_prediction(
    target_coords_ext,
    gt_A_action,
    gt_B_action,
    pred_A_action,
    pred_B_action,
):
    """
    Evaluate one prediction after aligning the pair with GT shard A.

    Metrics:
        1) IoU of shard B in world coordinates.
        2) Center displacement of shard B.
        3) Rotation error of shard B in degrees.
    """
    gt_A_action   = np.asarray(gt_A_action,   np.float32)
    gt_B_action   = np.asarray(gt_B_action,   np.float32)
    pred_A_action = np.asarray(pred_A_action, np.float32)
    pred_B_action = np.asarray(pred_B_action, np.float32)

    # 1) Align the predicted pair with a rigid transform.
    pred_A_aligned, pred_B_aligned = align_pred_actions_to_gt_by_A(
        pred_A_action, pred_B_action, gt_A_action
    )

    # 2) Build world-space polygons for GT B and predicted B.
    poly_gt_B_world, _, _, _ = _apply_pred_to_polygon(target_coords_ext, gt_B_action, neg_theta=True)
    poly_pred_B_world, _, _, _ = _apply_pred_to_polygon(target_coords_ext, pred_B_aligned, neg_theta=False)

    # 3) IoU
    iou_B = compute_iou(poly_gt_B_world, poly_pred_B_world)

    # 4) Center displacement.
    center_gt   = np.mean(poly_gt_B_world, axis=0)
    center_pred = np.mean(poly_pred_B_world, axis=0)
    disp_B = float(np.linalg.norm(center_gt - center_pred))

    # 5) Rotation error in degrees.
    _, _, theta_gt_B = action_to_theta(gt_B_action, neg_theta=True)
    _, _, theta_pred_B = action_to_theta(pred_B_aligned, neg_theta=False)

    # Normalize to [-pi, pi].
    angle_diff = theta_pred_B - theta_gt_B
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    angle_diff_deg = angle_diff * 180.0 / math.pi
    angle_error_deg_B = float(abs(angle_diff_deg))

    return iou_B, disp_B, angle_error_deg_B


def evaluate_all_samples_for_one_pair(
    target_coords_ext,
    gt_A_action,
    gt_B_action,
    preds_list,
):
    all_metrics = []

    for k, pred in enumerate(preds_list):
        pred = np.asarray(pred, np.float32)
        pred_A = pred[0]
        pred_B = pred[1]

        iou_B, disp_B, angle_error_deg_B = metrics_for_one_prediction(
            target_coords_ext,
            gt_A_action,
            gt_B_action,
            pred_A,
            pred_B,
        )

        print(
            f"[Eval] sample {k}: "
            f"IoU(B)={iou_B:.4f}, "
            f"disp(B)={disp_B:.4f}, "
            f"angleError_deg(B)={angle_error_deg_B:.4f}"
        )

        all_metrics.append({
            "idx": k,
            "iou": iou_B,
            "disp": disp_B,
            "angle_error_deg": angle_error_deg_B,
            "passes_thresholds": is_successful_prediction(
                iou_B,
                disp_B,
                angle_error_deg_B,
            ),
        })

    return all_metrics



def run_diffusion_repeat_5_times_with_gt_by_name(
    shard_a, shard_b,
    query_coords_ext, target_coords_ext,
    query_feat_ext, target_feat_ext,
    query_img_ext, target_img_ext,
    query_global_ext, target_global_ext,
    query_coords_int, target_coords_int,
    query_feat_int, target_feat_int,
    query_img_int, target_img_int,
    query_global_int, target_global_int,
    save_tag="pair_result",
    save_dir="./vis_diffusion",
    gt_json_path=GT_JSON_PATH,
):
    detailed_csv_path = os.path.join(save_dir, DETAILED_RESULTS_FILENAME)
    summary_csv_path = os.path.join(save_dir, SUMMARY_RESULTS_FILENAME)

    if pair_already_processed(detailed_csv_path, shard_a, shard_b):
        print(f"[Skip Pair] {shard_a} - {shard_b} already processed (or reversed)")
        return {
            "all_results": None,
            "selected_index": None,
            "best_pred_exterior": None,
            "selected_iou_B": None,
            "selected_disp_B": None,
            "selected_angle_error_deg_B": None,
            "is_recalled": None,
            "detailed_csv_path": detailed_csv_path,
            "summary_csv_path": summary_csv_path,
        }
    all_results = run_diffusion_repeat_5_times(
        query_coords_ext=query_coords_ext,
        target_coords_ext=target_coords_ext,
        query_feat_ext=query_feat_ext,
        target_feat_ext=target_feat_ext,
        query_img_ext=query_img_ext,
        target_img_ext=target_img_ext,
        query_global_ext=query_global_ext,
        target_global_ext=target_global_ext,
        query_coords_int=query_coords_int,
        target_coords_int=target_coords_int,
        query_feat_int=query_feat_int,
        target_feat_int=target_feat_int,
        query_img_int=query_img_int,
        target_img_int=target_img_int,
        query_global_int=query_global_int,
        target_global_int=target_global_int,
        save_tag=save_tag,
        save_dir=save_dir,
    )

    try:
        gt_A_action, gt_B_action = load_gt_pose_by_names(
            shard_a, shard_b, json_path=gt_json_path
        )
    except ValueError as e:
        print(f"[GT Missing] {e}")
        return {
            "all_results": all_results,
            "selected_index": None,
            "best_pred_exterior": None,
            "selected_iou_B": None,
            "selected_disp_B": None,
            "selected_angle_error_deg_B": None,
            "is_recalled": None,
            "detailed_csv_path": detailed_csv_path,
            "summary_csv_path": summary_csv_path,
        }

    preds_list = [res["pred_exterior"] for res in all_results]

    print("\n[Select] Evaluate all samples w.r.t GT ...")
    all_metrics = evaluate_all_samples_for_one_pair(
        target_coords_ext=target_coords_ext,
        gt_A_action=gt_A_action,
        gt_B_action=gt_B_action,
        preds_list=preds_list,
    )
    selected_metrics = [m for m in all_metrics if m["passes_thresholds"]]
    is_recalled = len(selected_metrics) > 0

    if is_recalled:
        selected_metric = max(selected_metrics, key=lambda item: item["iou"])
        selected_index = selected_metric["idx"]
    else:
        selected_metric = None
        selected_index = None

    for m in all_metrics:
        append_metrics_to_csv(
            csv_path=detailed_csv_path,
            shard_a=shard_a,
            shard_b=shard_b,
            sample_idx=m["idx"],
            iou_B=m["iou"],
            disp_B=m["disp"],
            angle_error_deg_B=m["angle_error_deg"],
            passes_thresholds=m["passes_thresholds"],
            is_selected=int(selected_index is not None and m["idx"] == selected_index),
        )

    append_best_metrics_to_csv(
        csv_path=summary_csv_path,
        shard_a=shard_a,
        shard_b=shard_b,
        is_recalled=is_recalled,
        best_sample_idx=selected_index,
        best_iou_B=None if selected_metric is None else selected_metric["iou"],
        best_disp_B=None if selected_metric is None else selected_metric["disp"],
        best_angle_error_deg_B=None if selected_metric is None else selected_metric["angle_error_deg"],
    )

    return {
        "all_results": all_results,
        "selected_index": selected_index,
        "best_pred_exterior": None if selected_index is None else preds_list[selected_index],
        "selected_iou_B": None if selected_metric is None else selected_metric["iou"],
        "selected_disp_B": None if selected_metric is None else selected_metric["disp"],
        "selected_angle_error_deg_B": None if selected_metric is None else selected_metric["angle_error_deg"],
        "is_recalled": is_recalled,
        "detailed_csv_path": detailed_csv_path,
        "summary_csv_path": summary_csv_path,
    }

