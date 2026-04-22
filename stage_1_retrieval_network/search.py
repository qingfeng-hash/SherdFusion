"""Train and evaluate the retrieval network on paired fragment data."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from PolygonMatchingNet import PolygonMatchingNet

SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_PATHS = [
    SCRIPT_DIR / "../dataset_simulated_sample/simulated_dataset_pairs1.pkl",
    SCRIPT_DIR / "../dataset_simulated_sample/simulated_dataset_pairs2.pkl",
    SCRIPT_DIR / "../dataset_simulated_sample/simulated_dataset_pairs3.pkl",
]
PRETRAINED_PATH = SCRIPT_DIR / "None"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
VIS_DIR = SCRIPT_DIR / "vis_top5"
METRICS_CSV_PATH = SCRIPT_DIR / "retrieval_metrics.csv"

TRAIN_RATIO = 0.85
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-5
GEOM_WEIGHT = 0.3
GLOBAL_WEIGHT = 0.7
TEMPERATURE = 0.07
SAVE_EVERY = 10
VIS_TOPK = 5
MAX_VISUALIZATIONS = 10


@dataclass
class FragmentRecord:
    """Container for one unique fragment in the retrieval dataset."""

    fragment_id: int
    poly: np.ndarray
    local_feature: np.ndarray
    global_feature: np.ndarray
    action: np.ndarray
    image: np.ndarray | None = None
    source_path: Path | None = None
    local_fragment_id: int | None = None


def calculate_angle(p1, p2, p3):
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot = a[0] * b[0] + a[1] * b[1]
    cross = a[0] * b[1] - a[1] * b[0]
    angle = math.atan2(cross, dot)
    angle = abs(angle) * (180.0 / math.pi)
    if cross < 0:
        angle = 360 - angle
    return angle


def compute_node_features(poly):
    feats = []
    for idx in range(len(poly)):
        p1 = poly[idx - 1]
        p2 = poly[idx]
        p3 = poly[(idx + 1) % len(poly)]
        feats.append([p2[0], p2[1], calculate_angle(p1, p2, p3)])
    return feats


def polygon_to_graph(record: FragmentRecord):
    """Convert one fragment record into a PyG graph."""
    coords = np.asarray(record.poly, dtype=np.float32)
    num_nodes = coords.shape[0]
    node_features = torch.tensor(compute_node_features(coords), dtype=torch.float32)
    local_features = torch.tensor(record.local_feature, dtype=torch.float32)
    global_features = torch.tensor(record.global_feature, dtype=torch.float32)

    edge_indices = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_indices += [((i + 1) % num_nodes, i) for i in range(num_nodes)]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    graph = Data(
        x=node_features,
        f=local_features,
        g=global_features.view(1, -1),
        edge_index=edge_index,
    )
    graph.fragment_id = record.fragment_id
    return graph


def load_pair_dataset(pkl_path: Path, load_images=True):
    """Load the compact simulated_dataset_pairs dataset format."""
    with pkl_path.open("rb") as file_handle:
        all_polys = pickle.load(file_handle)
        all_local_features = pickle.load(file_handle)
        all_global_features = pickle.load(file_handle)
        all_actions = pickle.load(file_handle)
        positive_pair_indices = pickle.load(file_handle)
        all_fragment_images = pickle.load(file_handle) if load_images else None

    return {
        "all_polys": all_polys,
        "all_local_features": all_local_features,
        "all_global_features": all_global_features,
        "all_actions": all_actions,
        "positive_pair_indices": positive_pair_indices,
        "all_fragment_images": all_fragment_images,
    }


def make_fragment_identity_key(poly, local_feature, global_feature, action):
    """Build a stable key for fragment identity without relying on image bytes."""
    hasher = hashlib.sha1()
    for value in (poly, local_feature, global_feature, action):
        array = np.ascontiguousarray(value)
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def build_fragment_catalog(pkl_paths, load_images=False):
    """Build a fragment catalog from one or more compact simulated_dataset_pairs files."""
    fragments = {}
    neighbors = {}
    positive_pairs = set()

    next_fragment_id = 0
    for pkl_path in pkl_paths:
        dataset = load_pair_dataset(pkl_path, load_images=load_images)

        all_polys = dataset["all_polys"]
        all_local_features = dataset["all_local_features"]
        all_global_features = dataset["all_global_features"]
        all_actions = dataset["all_actions"]
        all_fragment_images = dataset["all_fragment_images"]
        positive_pair_indices = dataset["positive_pair_indices"] or []

        fragment_id_offset = next_fragment_id

        for local_fragment_id in range(len(all_polys)):
            fragment_id = fragment_id_offset + local_fragment_id
            image = None
            if all_fragment_images is not None:
                image = all_fragment_images[local_fragment_id]

            fragments[fragment_id] = FragmentRecord(
                fragment_id=fragment_id,
                poly=np.asarray(all_polys[local_fragment_id], dtype=np.float32),
                local_feature=np.asarray(all_local_features[local_fragment_id], dtype=np.float32),
                global_feature=np.asarray(all_global_features[local_fragment_id], dtype=np.float32),
                action=np.asarray(all_actions[local_fragment_id], dtype=np.float32),
                image=image,
                source_path=pkl_path,
                local_fragment_id=local_fragment_id,
            )
            neighbors[fragment_id] = set()

        for local_fragment_a, local_fragment_b in positive_pair_indices:
            if local_fragment_a == local_fragment_b:
                continue
            fragment_a = fragment_id_offset + local_fragment_a
            fragment_b = fragment_id_offset + local_fragment_b
            neighbors[fragment_a].add(fragment_b)
            neighbors[fragment_b].add(fragment_a)
            positive_pairs.add((fragment_a, fragment_b))
            positive_pairs.add((fragment_b, fragment_a))

        next_fragment_id += len(all_polys)

    return fragments, neighbors, sorted(positive_pairs)


def attach_fragment_images(fragments):
    """Populate images lazily for all source datasets when visualization is needed."""
    image_cache = {}

    for record in fragments.values():
        if record.image is not None or record.source_path is None or record.local_fragment_id is None:
            continue

        if record.source_path not in image_cache:
            dataset = load_pair_dataset(record.source_path, load_images=True)
            image_cache[record.source_path] = dataset["all_fragment_images"] or []

        dataset_images = image_cache[record.source_path]
        if record.local_fragment_id < len(dataset_images):
            record.image = dataset_images[record.local_fragment_id]


def split_fragment_ids(positive_pairs, train_ratio=TRAIN_RATIO):
    """Split fragment ids by shuffling the unique ids appearing in all pair items."""
    all_ids = set()
    for fragment_a, fragment_b in positive_pairs:
        all_ids.add(fragment_a)
        all_ids.add(fragment_b)

    all_ids = list(all_ids)
    random.shuffle(all_ids)

    split_idx = int(len(all_ids) * train_ratio)
    train_ids = set(all_ids[:split_idx])
    test_ids = set(all_ids[split_idx:])
    return train_ids, test_ids


def filter_pairs_by_split(positive_pairs, valid_ids):
    """Keep only positive directed pairs fully inside one split."""
    return [
        (fragment_a, fragment_b)
        for fragment_a, fragment_b in positive_pairs
        if fragment_a in valid_ids and fragment_b in valid_ids
    ]


def build_graph_dict(fragments):
    """Prebuild PyG graphs for all unique fragments."""
    return {
        fragment_id: polygon_to_graph(record)
        for fragment_id, record in fragments.items()
    }


def batch_positive_pairs(positive_pairs, batch_size):
    """Yield mini-batches of positive pairs."""
    for idx in range(0, len(positive_pairs), batch_size):
        yield positive_pairs[idx : idx + batch_size]


def load_model_checkpoint_compat(model, checkpoint_path, device):
    """Load older checkpoints while tolerating stale parameter keys."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.unexpected_keys:
        print(f"Ignored unexpected keys from {checkpoint_path}: {incompatible.unexpected_keys}")
    if incompatible.missing_keys:
        print(f"Missing keys when loading {checkpoint_path}: {incompatible.missing_keys}")


def encode_graph_batch(model, graph_dict, fragment_ids, device):
    """Encode a list of fragment ids into retrieval embeddings."""
    graphs = [graph_dict[fragment_id] for fragment_id in fragment_ids]
    batch = Batch.from_data_list(graphs).to(device)
    node_features, global_features = model.encode_graph(batch)
    pooled_features = global_mean_pool(node_features, batch.batch)
    return pooled_features, global_features


def build_logits(emb_a, emb_b, global_a, global_b):
    """Fuse local and global similarity into one retrieval logits matrix."""
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    global_a = F.normalize(global_a, dim=-1)
    global_b = F.normalize(global_b, dim=-1)

    sim_geom = emb_a @ emb_b.T
    sim_global = global_a @ global_b.T
    return (GEOM_WEIGHT * sim_geom + GLOBAL_WEIGHT * sim_global) / TEMPERATURE


def multi_pos_info_nce(logits, ids_a, ids_b, neighbors):
    """Compute multi-positive InfoNCE loss within a training batch."""
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    for row_idx, fragment_a in enumerate(ids_a):
        positive_ids = neighbors.get(fragment_a, set())
        for col_idx, fragment_b in enumerate(ids_b):
            if fragment_b in positive_ids:
                target_mask[row_idx, col_idx] = True

    log_prob = torch.log_softmax(logits, dim=-1)
    positive_log_prob = log_prob[target_mask]
    if positive_log_prob.numel() == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    return -positive_log_prob.mean()


def compute_batch_retrieval_metrics(logits, ids_a, ids_b, neighbors, topk=(5, 10)):
    """Compute batch-local retrieval metrics following the legacy evaluation."""
    ranked_indices = torch.argsort(logits, dim=-1, descending=True)
    batch_candidates = set(ids_b)
    metrics = {}

    for k in topk:
        recall_sum = 0.0
        ndcg_sum = 0.0
        valid_queries = 0

        for row_idx, query_id in enumerate(ids_a):
            batch_gt_set = neighbors.get(query_id, set()) & batch_candidates
            if not batch_gt_set:
                continue

            valid_queries += 1
            ranked_pred_ids = []
            seen_pred_ids = set()
            for col_idx in ranked_indices[row_idx].tolist():
                pred_id = ids_b[col_idx]
                if pred_id == query_id or pred_id in seen_pred_ids:
                    continue
                ranked_pred_ids.append(pred_id)
                seen_pred_ids.add(pred_id)
            pred_ids = ranked_pred_ids[:k]

            hit_set = set(pred_ids) & batch_gt_set
            recall_sum += len(hit_set) / len(batch_gt_set)

            seen = set()
            relevance = []
            for pred_id in pred_ids:
                if pred_id in seen:
                    relevance.append(0)
                else:
                    relevance.append(1 if pred_id in batch_gt_set else 0)
                    seen.add(pred_id)

            dcg = sum(
                relevance[idx] / math.log2(idx + 2)
                for idx in range(len(relevance))
            )
            ideal_hits = min(len(batch_gt_set), k)
            idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

        if valid_queries > 0:
            metrics[f"Recall@{k}"] = recall_sum / valid_queries
            metrics[f"NDCG@{k}"] = ndcg_sum / valid_queries
        else:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0

    return metrics


def build_batch_rankings(logits, ids_a, ids_b, neighbors):
    """Build batch-local rankings for visualization."""
    ranked_indices = torch.argsort(logits, dim=-1, descending=True)
    batch_candidates = set(ids_b)
    rankings = []

    for row_idx, query_id in enumerate(ids_a):
        batch_gt_set = neighbors.get(query_id, set()) & batch_candidates
        if not batch_gt_set:
            continue

        top_ids = []
        seen_top_ids = set()
        for col_idx in ranked_indices[row_idx].tolist():
            pred_id = ids_b[col_idx]
            if pred_id == query_id or pred_id in seen_top_ids:
                continue
            top_ids.append(pred_id)
            seen_top_ids.add(pred_id)
        rankings.append({
            "query_id": query_id,
            "gt_ids": set(batch_gt_set),
            "top_ids": top_ids,
        })

    return rankings


def average_metric_logs(metric_logs, topk=(5, 10)):
    """Average metric dictionaries the same way as the legacy training script."""
    if not metric_logs:
        metrics = {}
        for k in topk:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0
        return metrics

    return {
        metric_name: sum(item[metric_name] for item in metric_logs) / len(metric_logs)
        for metric_name in metric_logs[0]
    }


def visualize_top5_results(rankings, fragments, output_dir, topk=VIS_TOPK, max_visualizations=MAX_VISUALIZATIONS):
    """Visualize query + top5 retrieval results and highlight successful recalls."""
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_rankings = [
        item for item in rankings if any(fragment_id in item["gt_ids"] for fragment_id in item["top_ids"][:topk])
    ]
    selected_rankings = successful_rankings[:max_visualizations]
    if not selected_rankings:
        selected_rankings = rankings[:max_visualizations]

    for vis_idx, ranking in enumerate(selected_rankings):
        query_record = fragments[ranking["query_id"]]
        top_ids = ranking["top_ids"][:topk]

        fig, axes = plt.subplots(1, topk + 1, figsize=(3 * (topk + 1), 3), dpi=150)
        axes = np.atleast_1d(axes)

        axes[0].imshow(decode_fragment_image(query_record.image))
        axes[0].set_title(f"Query\nID {query_record.fragment_id}")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for spine in axes[0].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor("dodgerblue")

        for rank_idx in range(topk):
            ax = axes[rank_idx + 1]
            if rank_idx < len(top_ids):
                candidate_id = top_ids[rank_idx]
                candidate_record = fragments[candidate_id]
                ax.imshow(decode_fragment_image(candidate_record.image))
                is_gt = candidate_id in ranking["gt_ids"]
                title = f"Top{rank_idx + 1}\nID {candidate_id}"
                if is_gt:
                    title += "\nGT"
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(4)
                        spine.set_edgecolor("limegreen")
                    ax.set_title(
                        title,
                        color="limegreen",
                        fontweight="bold",
                    )
                else:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.5)
                        spine.set_edgecolor("lightgray")
                    ax.set_title(title)
            else:
                ax.set_axis_off()
                continue

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        save_path = output_dir / f"query_{query_record.fragment_id:05d}_top{topk}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def decode_fragment_image(image_value):
    """Decode fragment images stored either as PNG bytes or RGBA arrays."""
    if isinstance(image_value, (bytes, bytearray)):
        import io
        from PIL import Image

        with Image.open(io.BytesIO(image_value)) as image:
            return np.array(image.convert("RGBA"))
    return np.asarray(image_value)


def write_metrics_csv(metrics_path, train_metrics, test_metrics):
    """Save the final retrieval metrics to CSV."""
    with metrics_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["split", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10"])
        writer.writerow([
            "train",
            f"{train_metrics.get('Recall@5', 0.0):.6f}",
            f"{train_metrics.get('Recall@10', 0.0):.6f}",
            f"{train_metrics.get('NDCG@5', 0.0):.6f}",
            f"{train_metrics.get('NDCG@10', 0.0):.6f}",
        ])
        writer.writerow([
            "test",
            f"{test_metrics.get('Recall@5', 0.0):.6f}",
            f"{test_metrics.get('Recall@10', 0.0):.6f}",
            f"{test_metrics.get('NDCG@5', 0.0):.6f}",
            f"{test_metrics.get('NDCG@10', 0.0):.6f}",
        ])


def print_metrics(prefix, metrics):
    """Print a compact metric summary."""
    print(
        f"{prefix} | "
        f"Recall@5={metrics.get('Recall@5', 0.0):.4f} | "
        f"Recall@10={metrics.get('Recall@10', 0.0):.4f} | "
        f"NDCG@5={metrics.get('NDCG@5', 0.0):.4f} | "
        f"NDCG@10={metrics.get('NDCG@10', 0.0):.4f}"
    )


def main():
    for dataset_path in DATASET_PATHS:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fragments, neighbors, positive_pairs = build_fragment_catalog(DATASET_PATHS, load_images=False)
    graph_dict = build_graph_dict(fragments)

    train_ids, test_ids = split_fragment_ids(positive_pairs)
    train_pairs = filter_pairs_by_split(positive_pairs, train_ids)
    test_pairs = filter_pairs_by_split(positive_pairs, test_ids)

    print(f"Unique fragments: {len(fragments)}")
    print(f"Train fragments: {len(train_ids)}, Test fragments: {len(test_ids)}")
    print(f"Train positive pairs: {len(train_pairs)}, Test positive pairs: {len(test_pairs)}")

    model = PolygonMatchingNet(node_feat_dim=3, patch_feat_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if PRETRAINED_PATH.exists():
        print(f"Loading pretrained weights from {PRETRAINED_PATH} ...")
        load_model_checkpoint_compat(model, PRETRAINED_PATH, device)
    else:
        print("No pretrained checkpoint found; training from scratch.")

    best_test_recall5 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train_pairs)
        epoch_loss = 0.0
        num_batches = 0
        train_metric_logs = []

        for pair_batch in batch_positive_pairs(train_pairs, BATCH_SIZE):
            ids_a = [fragment_a for fragment_a, _ in pair_batch]
            ids_b = [fragment_b for _, fragment_b in pair_batch]

            emb_a, global_a = encode_graph_batch(model, graph_dict, ids_a, device)
            emb_b, global_b = encode_graph_batch(model, graph_dict, ids_b, device)
            logits = build_logits(emb_a, emb_b, global_a, global_b)
            loss = multi_pos_info_nce(logits, ids_a, ids_b, neighbors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            train_metric_logs.append(
                compute_batch_retrieval_metrics(
                    logits.detach().cpu(),
                    ids_a,
                    ids_b,
                    neighbors,
                )
            )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[Epoch {epoch}] Train Loss={avg_loss:.6f}")

        train_metrics = average_metric_logs(train_metric_logs)

        model.eval()
        test_loss = 0.0
        test_metric_logs = []
        test_rankings = []
        test_batches = 0

        with torch.no_grad():
            for pair_batch in batch_positive_pairs(test_pairs, BATCH_SIZE):
                ids_a = [fragment_a for fragment_a, _ in pair_batch]
                ids_b = [fragment_b for _, fragment_b in pair_batch]

                emb_a, global_a = encode_graph_batch(model, graph_dict, ids_a, device)
                emb_b, global_b = encode_graph_batch(model, graph_dict, ids_b, device)
                logits = build_logits(emb_a, emb_b, global_a, global_b)
                loss = multi_pos_info_nce(logits, ids_a, ids_b, neighbors)

                test_loss += loss.item()
                test_batches += 1
                cpu_logits = logits.cpu()
                test_metric_logs.append(
                    compute_batch_retrieval_metrics(
                        cpu_logits,
                        ids_a,
                        ids_b,
                        neighbors,
                    )
                )
                if len(test_rankings) < MAX_VISUALIZATIONS:
                    test_rankings.extend(
                        build_batch_rankings(
                            cpu_logits,
                            ids_a,
                            ids_b,
                            neighbors,
                        )
                    )

        avg_test_loss = test_loss / max(test_batches, 1)
        print(f"[Epoch {epoch}] Test Loss={avg_test_loss:.6f}")
        test_metrics = average_metric_logs(test_metric_logs)

        print_metrics(f"[Epoch {epoch}] Train", train_metrics)
        print_metrics(f"[Epoch {epoch}] Test", test_metrics)

        if epoch % SAVE_EVERY == 0:
            best_test_recall5 = max(best_test_recall5, test_metrics.get("Recall@5", 0.0))
            if any(record.image is None for record in fragments.values()):
                print("Loading fragment images for visualization ...")
                attach_fragment_images(fragments)
            visualize_top5_results(test_rankings, fragments, VIS_DIR, topk=VIS_TOPK)
            write_metrics_csv(METRICS_CSV_PATH, train_metrics, test_metrics)
            print(f"Updated top5 visualizations in: {VIS_DIR}")
            print(f"Saved metrics to: {METRICS_CSV_PATH}")

        if epoch % SAVE_EVERY == 0:
            checkpoint_path = CHECKPOINT_DIR / f"retrieval_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
