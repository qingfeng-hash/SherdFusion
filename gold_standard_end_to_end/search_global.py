"""Global retrieval and diffusion-based verification pipeline."""

import csv
import math
import os
import pickle
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from PolygonMatchingNet import PolygonMatchingNet
from search_diffusion import run_diffusion_repeat_5_times_with_gt_by_name

# ------------------ Data processing ------------------
def calculate_angle(p1, p2, p3):
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = a[0] * b[0] + a[1] * b[1]
    cross_product = a[0] * b[1] - a[1] * b[0]
    angle = math.atan2(cross_product, dot_product)
    angle = abs(angle) * (180.0 / math.pi)
    if cross_product < 0:
        angle = 360 - angle
    return angle

def compute_node_features(poly):
    node_features = []
    for i in range(len(poly)):
        p1, p2, p3 = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
        internal_angle = calculate_angle(p1, p2, p3)
        node_features.append([p2[0], p2[1], internal_angle])
    return node_features

def compute_global_features(poly):
    polygon = Polygon(poly)
    area = polygon.area
    perimeter = polygon.length
    return [area, perimeter]


def polygon_to_graph(poly, local_feat, global_feat):
    coords = np.array(poly)
    num_nodes = coords.shape[0]
    x = compute_node_features(coords)
    x = torch.tensor(x, dtype=torch.float32)
    f = torch.tensor(local_feat, dtype=torch.float32)
    g = torch.tensor(global_feat, dtype=torch.float32)
    area, perm = compute_global_features(poly)
    area_tensor = torch.tensor(area, dtype=torch.float32)
    perm_tensor = torch.tensor(perm, dtype=torch.float32)
    edge_indices = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_indices += [((i + 1) % num_nodes, i) for i in range(num_nodes)]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    data = Data(x=x, f=f, g=g, edge_index=edge_index, area=area_tensor, perm=perm_tensor)
    data.coords = coords
    return data

def load_patch_dataset_with_images(pkl_path):
    """Load either the new dict-based dataset format or the legacy format."""
    with open(pkl_path, "rb") as f:
        first_obj = pickle.load(f)

        if isinstance(first_obj, dict):
            all_polys = first_obj["polygons"]
            all_features = first_obj["patch_features"]
            all_global_features = first_obj["global_features"]
            all_images = first_obj["images"]
            all_names = first_obj["names"]
        else:
            all_polys = first_obj
            all_features = pickle.load(f)
            all_global_features = pickle.load(f)
            all_images = pickle.load(f)
            all_names = pickle.load(f)

    graph_list = []
    for poly, local_feat, global_feat in zip(all_polys, all_features, all_global_features):
        graph = polygon_to_graph(poly, local_feat, global_feat)
        graph_list.append(graph)

    return graph_list, all_images, all_names


def load_gt_pairs(pair_csv_path):
    """Load GT shard pairs as unordered pair keys."""
    gt_pairs = set()
    if not os.path.exists(pair_csv_path):
        return gt_pairs

    with open(pair_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_pairs.add(tuple(sorted((row["shard_a"], row["shard_b"]))))
    return gt_pairs


def summarize_recall_metrics(summary_csv_path, pair_csv_path, output_csv_path=None):
    """Aggregate recall and averaged metrics from per-pair summaries."""
    gt_pairs = load_gt_pairs(pair_csv_path)
    if not gt_pairs or not os.path.exists(summary_csv_path):
        return None

    summary_rows = {}
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_key = tuple(sorted((row["shard_a"], row["shard_b"])))
            if pair_key in gt_pairs:
                summary_rows[pair_key] = row

    recalled_rows = [
        row for row in summary_rows.values()
        if int(row["is_recalled"]) == 1
    ]
    total_pairs = len(gt_pairs)
    recalled_pairs = len(recalled_rows)
    recall_rate = recalled_pairs / total_pairs if total_pairs > 0 else 0.0

    if recalled_rows:
        ae_trans = sum(float(row["best_disp_B"]) for row in recalled_rows) / recalled_pairs
        ae_rot = sum(float(row["best_angle_error_deg_B"]) for row in recalled_rows) / recalled_pairs
        avg_iou = sum(float(row["best_iou_B"]) for row in recalled_rows) / recalled_pairs
    else:
        ae_trans = float("nan")
        ae_rot = float("nan")
        avg_iou = float("nan")

    metrics = {
        "total_gt_pairs": total_pairs,
        "recalled_pairs": recalled_pairs,
        "recall_rate": recall_rate,
        "AEtrans": ae_trans,
        "AErot": ae_rot,
        "Avg_IoU": avg_iou,
    }

    if output_csv_path is not None:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "total_gt_pairs",
                "recalled_pairs",
                "recall_rate",
                "AEtrans",
                "AErot",
                "Avg_IoU",
            ])
            writer.writerow([
                metrics["total_gt_pairs"],
                metrics["recalled_pairs"],
                f"{metrics['recall_rate']:.6f}",
                "" if math.isnan(metrics["AEtrans"]) else f"{metrics['AEtrans']:.6f}",
                "" if math.isnan(metrics["AErot"]) else f"{metrics['AErot']:.6f}",
                "" if math.isnan(metrics["Avg_IoU"]) else f"{metrics['Avg_IoU']:.6f}",
            ])

    return metrics


def load_model_checkpoint_compat(model, checkpoint_path, device):
    """Load a checkpoint while tolerating stale keys from older model versions."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.unexpected_keys:
        print(f"Ignored unexpected keys from {checkpoint_path}: {incompatible.unexpected_keys}")
    if incompatible.missing_keys:
        print(f"Missing keys when loading {checkpoint_path}: {incompatible.missing_keys}")

    return checkpoint


def compute_all_embeddings(graph_list, model, device, batch_size=64):
    """Encode all polygons into retrieval descriptors."""
    model.eval()
    all_nodes, all_global = [], []
    with torch.no_grad():
        for i in range(0, len(graph_list), batch_size):
            batch_graphs = graph_list[i:i+batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)
            feat_nodes, _, ga_proj, _ = model(batch, batch)
            emb_nodes = global_mean_pool(feat_nodes, batch.batch)
            all_nodes.append(emb_nodes)
            all_global.append(ga_proj)
    all_nodes = torch.cat(all_nodes, dim=0)
    all_global = torch.cat(all_global, dim=0)
    return all_nodes, all_global

def compute_pairwise_logits_from_embeddings(emb_nodes, ga_proj, geom_weight=0.3, global_weight=0.7):
    """Fuse local geometry similarity and global descriptor similarity."""
    emb_nodes = emb_nodes / emb_nodes.norm(dim=-1, keepdim=True)
    ga_proj = ga_proj / ga_proj.norm(dim=-1, keepdim=True)
    sim_geom = torch.matmul(emb_nodes, emb_nodes.t())
    sim_global = torch.matmul(ga_proj, ga_proj.t())
    logits_matrix = geom_weight * sim_geom + global_weight * sim_global
    return logits_matrix.cpu()


def visualize_search_images_combined(query_idx, topk_idx, images, names, save_dir="vis_search_images"):
    """Save a transparent panel with the query and its top-k candidates."""
    os.makedirs(save_dir, exist_ok=True)
    total_cols = 1 + len(topk_idx)
    fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4), dpi=150)
    fig.patch.set_alpha(0.0)
    if total_cols == 1:
        axes = [axes]
    for ax in axes:
        ax.patch.set_alpha(0.0)

    # Helper: crop the visible foreground.
    def extract_bbox(img, pad=10):
        if img.shape[2] == 4:
            rgba = img.copy()
            alpha = rgba[:, :, 3]
            fg_mask = alpha > 0
        else:
            fg_mask = np.any(img != 0, axis=2)
            rgba = np.dstack([img, fg_mask.astype(np.uint8) * 255])
        ys, xs = np.where(fg_mask)
        if len(xs) == 0 or len(ys) == 0:
            return rgba, (0, rgba.shape[1], 0, rgba.shape[0])
        x_min, x_max = max(xs.min() - pad, 0), min(xs.max() + pad, rgba.shape[1])
        y_min, y_max = max(ys.min() - pad, 0), min(ys.max() + pad, rgba.shape[0])
        return rgba[y_min:y_max, x_min:x_max], (x_min, x_max, y_min, y_max)

    # Crop the query and top-k candidates.
    all_idx = [query_idx] + topk_idx
    crops = []
    max_h, max_w = 0, 0
    for idx in all_idx:
        crop, bbox = extract_bbox(images[idx])
        crops.append(crop)
        h, w = crop.shape[:2]
        max_h, max_w = max(max_h, h), max(max_w, w)

    # Show each cropped foreground on a centered canvas.
    def show_foreground(ax, crop_img):
        canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        h, w = crop_img.shape[:2]
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = crop_img
        ax.imshow(canvas)
        ax.axis("off")

    # Show the query image.
    show_foreground(axes[0], crops[0])
    axes[0].set_title(f"Query\n{names[query_idx]}", fontsize=10)

    # Show the top-k retrieved images.
    for i, idx in enumerate(topk_idx):
        show_foreground(axes[i + 1], crops[i + 1])
        axes[i + 1].set_title(f"Top{i + 1}\n{names[idx]}", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"query{query_idx}_topk_fg.png")
    plt.savefig(save_path, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved retrieval visualization to {save_path}")

def process_all_clusters(base_dir="./dataset", model_path="stage2_search_epoch.pth", topk=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_csv_path = os.path.join(base_dir, "pair_shard_ids.csv")

    # === Unified output directory ===
    save_folder = "./dataset/results"
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving results to: {save_folder}")

    for name in sorted(os.listdir(base_dir)):
        if not name.startswith("gold_standard_dataset.pkl") or not name.endswith(".pkl"):
            continue

        dataset_pkl = os.path.join(base_dir, name)
        print(f"\n=== Processing {name} ===")

        # === Load data ===
        graph_list, images, names = load_patch_dataset_with_images(dataset_pkl)
        print(f"Loaded {len(graph_list)} polygons and {len(images)} images")

        # === Split exterior and interior sides ===
        exterior_idx = [i for i, n in enumerate(names) if "_exterior" in n]
        interior_idx = [i for i, n in enumerate(names) if "_interior" in n]
        exterior_graphs = [graph_list[i] for i in exterior_idx]
        interior_graphs = [graph_list[i] for i in interior_idx]
        exterior_images = [images[i] for i in exterior_idx]
        interior_images = [images[i] for i in interior_idx]
        exterior_names = [names[i] for i in exterior_idx]
        interior_names = [names[i] for i in interior_idx]
        print(f"Exterior count: {len(exterior_graphs)}, Interior count: {len(interior_graphs)}")

        # === Load retrieval model ===
        model = PolygonMatchingNet(node_feat_dim=3, patch_feat_dim=128, global_feat_dim=128).to(device)
        load_model_checkpoint_compat(model, model_path, device)
        model.eval()
        print(f"Loaded model: {model_path}")

        # === Build retrieval similarity matrices ===
        emb_nodes_ext, ga_proj_ext = compute_all_embeddings(exterior_graphs, model, device)
        emb_nodes_int, ga_proj_int = compute_all_embeddings(interior_graphs, model, device)
        logits_ext = compute_pairwise_logits_from_embeddings(
            emb_nodes_ext, ga_proj_ext, geom_weight=0.3, global_weight=0.7
        )
        logits_int = compute_pairwise_logits_from_embeddings(
            emb_nodes_int, ga_proj_int, geom_weight=0.3, global_weight=0.7
        )

        # === Build shard id mappings ===
        def base_id(name): return name.split("_")[0]
        base_ext = [base_id(n) for n in exterior_names]
        base_int = [base_id(n) for n in interior_names]
        unique_ids = sorted(set(base_ext) & set(base_int))

        for main_id in unique_ids:
            start_time = time.time()
            try:
                q_ext_idx = base_ext.index(main_id)
                q_int_idx = base_int.index(main_id)

                # === Search on both sides, then fuse scores ===
                sims_ext = logits_ext[q_ext_idx].clone()
                sims_int = logits_int[q_int_idx].clone()
                sims_ext[q_ext_idx] = -1e9
                sims_int[q_int_idx] = -1e9

                combined_sims = torch.zeros_like(sims_ext)
                for j, tgt_id in enumerate(base_ext):
                    combined_sims[j] = (
                        0.5 * sims_ext[j] + 0.5 * sims_int[base_int.index(tgt_id)]
                        if tgt_id in base_int else sims_ext[j]
                    )

                # === Keep non-negative matches only ===
                topk_values, topk_indices = torch.topk(combined_sims, k=topk)
                topk_idx, topk_sims = [], []
                for i, score in zip(topk_indices.tolist(), topk_values.tolist()):
                    score_percent = score * 100 if score <= 1 else score
                    if score_percent >= 0:
                        topk_idx.append(i)
                        topk_sims.append(score_percent)

                if len(topk_idx) == 0:
                    print(f"[Skip] {main_id}: no match with score >= 0.")
                    continue

                print(f"\n[Query] {main_id}")
                print(f"Top{len(topk_idx)} (>= 0): {[exterior_names[i] for i in topk_idx]}")
                print(f"Scores: {topk_sims}")

                # === Visualize retrieval results ===
                vis_dir_ext = os.path.join(save_folder, "exterior")
                vis_dir_int = os.path.join(save_folder, "interior")
                os.makedirs(vis_dir_ext, exist_ok=True)
                os.makedirs(vis_dir_int, exist_ok=True)

                visualize_search_images_combined(
                    q_ext_idx, topk_idx, exterior_images, exterior_names,
                    save_dir=vis_dir_ext
                )
                visualize_search_images_combined(
                    q_int_idx,
                    [base_int.index(base_ext[i]) for i in topk_idx if base_ext[i] in base_int],
                    interior_images, interior_names,
                    save_dir=vis_dir_int
                )

                # === Diffusion-based pose refinement on both sides ===
                query_coords_ext = exterior_graphs[q_ext_idx].coords
                query_feat_ext = exterior_graphs[q_ext_idx].f.cpu().numpy()
                query_img_ext = exterior_images[q_ext_idx]
                query_global_ext = exterior_graphs[q_ext_idx].g.cpu().numpy()

                query_coords_int = interior_graphs[q_int_idx].coords
                query_feat_int = interior_graphs[q_int_idx].f.cpu().numpy()
                query_img_int = interior_images[q_int_idx]
                query_global_int = interior_graphs[q_int_idx].g.cpu().numpy()

                for cid in topk_idx:
                    target_coords_ext = exterior_graphs[cid].coords
                    target_feat_ext = exterior_graphs[cid].f.cpu().numpy()
                    target_img_ext = exterior_images[cid]
                    target_global_ext = exterior_graphs[cid].g.cpu().numpy()

                    if base_ext[cid] in base_int:
                        target_int_idx = base_int.index(base_ext[cid])
                        target_coords_int = interior_graphs[target_int_idx].coords
                        target_feat_int = interior_graphs[target_int_idx].f.cpu().numpy()
                        target_img_int = interior_images[target_int_idx]
                        target_global_int = interior_graphs[target_int_idx].g.cpu().numpy()
                    else:
                        continue

                    run_diffusion_repeat_5_times_with_gt_by_name(
                        shard_a=main_id,
                        shard_b=base_ext[cid],
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
                        save_tag=f"{main_id}_vs_{base_ext[cid]}",
                        save_dir=save_folder
                    )
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"[Done] Query {main_id} took {elapsed:.3f}s")

            except Exception as e:
                print(f"[Error] {main_id} search failed: {e}")

    summary_csv_path = os.path.join(save_folder, "evaluation_summary_thresholded.csv")
    overall_csv_path = os.path.join(save_folder, "overall_metrics_thresholded.csv")
    overall_metrics = summarize_recall_metrics(
        summary_csv_path=summary_csv_path,
        pair_csv_path=pair_csv_path,
        output_csv_path=overall_csv_path,
    )
    if overall_metrics is not None:
        print("\nFinal metrics:")
        print(
            f"Recall = {overall_metrics['recalled_pairs']}/{overall_metrics['total_gt_pairs']} "
            f"({overall_metrics['recall_rate']:.4f})"
        )
        print(f"AEtrans = {overall_metrics['AEtrans']:.4f}")
        print(f"AErot = {overall_metrics['AErot']:.4f}")
        print(f"Avg. IoU = {overall_metrics['Avg_IoU']:.4f}")
        print(f"Saved overall metrics to: {overall_csv_path}")

    print("Finished processing all clusters.")


if __name__ == "__main__":
    process_all_clusters()


