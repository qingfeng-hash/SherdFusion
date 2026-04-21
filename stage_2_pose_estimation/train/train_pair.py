"""Train the diffusion pose estimator on simulated_dataset_pairs.pkl."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data
from tqdm import tqdm

from critic import PolygonPackingTransformer
from sde import ExponentialMovingAverage, init_sde, lossFun, pc_sampler_state


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset_simulated_sample_1000" / "simulated_dataset_pairs.pkl"
DEFAULT_PRETRAINED_PATH = SCRIPT_DIR / "critic_29.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_angle(p1, p2, p3):
    """Compute the internal angle at p2 in degrees."""
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = a[0] * b[0] + a[1] * b[1]
    cross_product = a[0] * b[1] - a[1] * b[0]
    angle = math.atan2(cross_product, dot_product)
    angle = abs(angle) * (180.0 / math.pi)
    if cross_product < 0:
        angle = 360.0 - angle
    return angle


def compute_node_features(poly: np.ndarray) -> np.ndarray:
    """Build the 3D node feature used by the GNN encoder."""
    node_features = []
    for idx in range(len(poly)):
        prev_point = poly[idx - 1]
        point = poly[idx]
        next_point = poly[(idx + 1) % len(poly)]
        node_features.append([point[0], point[1], calculate_angle(prev_point, point, next_point)])
    return np.asarray(node_features, dtype=np.float32)


def build_cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Create a bidirectional cycle graph for one polygon contour."""
    edge_indices = [(idx, (idx + 1) % num_nodes) for idx in range(num_nodes)]
    edge_indices += [((idx + 1) % num_nodes, idx) for idx in range(num_nodes)]
    return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()


def polygon_to_graph(poly, local_feature, global_feature, action) -> Data:
    """Convert one fragment record into the PyG graph expected by the critic."""
    coords = np.asarray(poly, dtype=np.float32)
    graph = Data(
        x=torch.tensor(compute_node_features(coords), dtype=torch.float32),
        f=torch.tensor(np.asarray(local_feature, dtype=np.float32), dtype=torch.float32),
        g=torch.tensor(np.asarray(global_feature, dtype=np.float32), dtype=torch.float32).view(1, -1),
        edge_index=build_cycle_edge_index(len(coords)),
        action=torch.tensor(np.asarray(action, dtype=np.float32), dtype=torch.float32),
    )
    graph.poly = torch.tensor(coords, dtype=torch.float32)
    return graph


def action_to_theta(action: torch.Tensor) -> torch.Tensor:
    """Convert the cosine-sine action head back to a rotation angle."""
    return torch.atan2(action[3], action[2])


def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Create a 2D rotation matrix from one angle."""
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta, cos_theta]),
        ],
        dim=0,
    )


def apply_action_to_vertices(vertices, action) -> torch.Tensor:
    """Rotate and translate one polygon with one predicted or GT action."""
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices, dtype=torch.float32, device=action.device)
    else:
        vertices = vertices.to(action.device).float()

    rotated = vertices @ rotation_matrix(action_to_theta(action)).T
    return rotated + action[:2].unsqueeze(0)


def close_polygon(points: np.ndarray) -> np.ndarray:
    """Append the first point to the end for plotting a closed contour."""
    if len(points) == 0:
        return points
    return np.concatenate([points, points[:1]], axis=0)


def draw_pose_pair(ax, vertices_a, vertices_b, actions, title: str) -> None:
    """Draw two transformed fragments in one subplot."""
    transformed_a = apply_action_to_vertices(vertices_a, actions[0]).detach().cpu().numpy()
    transformed_b = apply_action_to_vertices(vertices_b, actions[1]).detach().cpu().numpy()

    poly_a = close_polygon(transformed_a)
    poly_b = close_polygon(transformed_b)

    ax.plot(poly_a[:, 0], poly_a[:, 1], color="#0B6E4F", linewidth=2.0)
    ax.fill(poly_a[:, 0], poly_a[:, 1], color="#0B6E4F", alpha=0.22)
    ax.plot(poly_b[:, 0], poly_b[:, 1], color="#C75000", linewidth=2.0)
    ax.fill(poly_b[:, 0], poly_b[:, 1], color="#C75000", alpha=0.22)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_pose_visualization(
    model: torch.nn.Module,
    sde_fn,
    dataset: Dataset,
    device: torch.device,
    epoch: int,
    split_name: str,
    output_dir: Path,
    num_steps_sample: int,
) -> None:
    """Sample one pair from the dataset and save GT/prediction visualizations."""
    if len(dataset) == 0:
        return

    sample_index = random.randrange(len(dataset))
    graph_a, graph_b, gt_actions = dataset[sample_index]
    batch_a = PyGBatch.from_data_list([graph_a]).to(device)
    batch_b = PyGBatch.from_data_list([graph_b]).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, sampled_actions = pc_sampler_state(
            score_model=model,
            sde_fn=sde_fn,
            g1=batch_a,
            g2=batch_b,
            num_steps=num_steps_sample,
        )
    if was_training:
        model.train()

    predicted_actions = sampled_actions[0].to(device)

    vertices_a = graph_a.poly
    vertices_b = graph_b.poly

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_pose_pair(axes[0], vertices_a, vertices_b, gt_actions, f"{split_name} GT")
    draw_pose_pair(axes[1], vertices_a, vertices_b, predicted_actions, f"{split_name} Pred")

    figure.suptitle(f"Epoch {epoch:04d} | sample {sample_index}", fontsize=13)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{split_name.lower()}_epoch_{epoch:04d}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_simulated_pair_dataset(pkl_path: Path):
    """Load the compact simulated dataset format saved with six pickle dumps."""
    with pkl_path.open("rb") as file_handle:
        all_polys = pickle.load(file_handle)
        all_local_features = pickle.load(file_handle)
        all_global_features = pickle.load(file_handle)
        all_actions = pickle.load(file_handle)
        positive_pair_indices = pickle.load(file_handle)
        all_fragment_images = pickle.load(file_handle)

    expected_length = len(all_polys)
    for name, values in [
        ("all_local_features", all_local_features),
        ("all_global_features", all_global_features),
        ("all_actions", all_actions),
        ("all_fragment_images", all_fragment_images),
    ]:
        if len(values) != expected_length:
            raise ValueError(
                f"{name} has length {len(values)}, expected {expected_length}."
            )

    if not positive_pair_indices:
        raise ValueError("positive_pair_indices is empty.")

    return {
        "all_polys": all_polys,
        "all_local_features": all_local_features,
        "all_global_features": all_global_features,
        "all_actions": all_actions,
        "positive_pair_indices": [tuple(map(int, pair)) for pair in positive_pair_indices],
    }


def split_positive_pairs(positive_pairs, test_ratio: float, seed: int):
    """Split pair indices into train and validation subsets."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    shuffled_indices = list(range(len(positive_pairs)))
    rng = random.Random(seed)
    rng.shuffle(shuffled_indices)

    if len(shuffled_indices) == 1:
        return [positive_pairs[shuffled_indices[0]]], [positive_pairs[shuffled_indices[0]]]

    train_count = int(round(len(shuffled_indices) * (1.0 - test_ratio)))
    train_count = max(1, min(len(shuffled_indices) - 1, train_count))

    train_pairs = [positive_pairs[idx] for idx in shuffled_indices[:train_count]]
    val_pairs = [positive_pairs[idx] for idx in shuffled_indices[train_count:]]
    return train_pairs, val_pairs


class SimulatedPairDataset(Dataset):
    """Pair dataset that builds two polygon graphs and one [2,4] action tensor."""

    def __init__(
        self,
        all_polys,
        all_local_features,
        all_global_features,
        all_actions,
        positive_pairs,
    ) -> None:
        self.positive_pairs = list(positive_pairs)
        self.graph_bank = [
            polygon_to_graph(poly, local_feature, global_feature, action)
            for poly, local_feature, global_feature, action in zip(
                all_polys,
                all_local_features,
                all_global_features,
                all_actions,
            )
        ]
        self.action_bank = [
            torch.tensor(np.asarray(action, dtype=np.float32), dtype=torch.float32)
            for action in all_actions
        ]

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, index: int):
        fragment_idx_a, fragment_idx_b = self.positive_pairs[index]
        graph_a = self.graph_bank[fragment_idx_a].clone()
        graph_b = self.graph_bank[fragment_idx_b].clone()
        actions = torch.stack(
            [self.action_bank[fragment_idx_a], self.action_bank[fragment_idx_b]],
            dim=0,
        )
        return graph_a, graph_b, actions


def pair_collate_fn(batch):
    """Collate a batch of positive pairs into two PyG batches."""
    graph_a_list, graph_b_list, action_list = zip(*batch)
    batch_a = PyGBatch.from_data_list(list(graph_a_list))
    batch_b = PyGBatch.from_data_list(list(graph_b_list))
    actions = torch.stack(action_list, dim=0)
    return batch_a, batch_b, actions


def move_batch_to_device(batch_a, batch_b, actions, device):
    """Move one mini-batch onto the selected device."""
    return batch_a.to(device), batch_b.to(device), actions.to(device)


def load_checkpoint_compat(model: torch.nn.Module, checkpoint_path: Path) -> bool:
    """Load a checkpoint while tolerating stale keys from older model versions."""
    if not checkpoint_path.exists():
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key[7:] if key.startswith("module.") else key] = value

    current_state_dict = model.state_dict()
    matched_state_dict = {}

    for key, value in cleaned_state_dict.items():
        if key not in current_state_dict:
            continue
        if current_state_dict[key].shape != value.shape:
            continue
        matched_state_dict[key] = value

    current_state_dict.update(matched_state_dict)
    model.load_state_dict(current_state_dict, strict=False)
    print(
        f"Loaded {len(matched_state_dict)}/{len(current_state_dict)} compatible parameters "
        f"from {checkpoint_path.name}"
    )
    return True


def write_history_header(csv_path: Path) -> None:
    """Create the training history CSV with a header row."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])


def append_history_row(csv_path: Path, epoch: int, train_loss: float, val_loss: float, learning_rate: float) -> None:
    """Append one epoch summary row to the training history CSV."""
    with csv_path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{learning_rate:.8f}"])


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    ema: ExponentialMovingAverage,
    epoch: int,
    global_step: int,
    args,
) -> None:
    """Save a resumable training checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": ema.state_dict(),
        },
        checkpoint_path,
    )


def save_model_weights(weights_path: Path, model: torch.nn.Module) -> None:
    """Save only the model weights for easier downstream loading."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)


def run_train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    ema: ExponentialMovingAverage,
    marginal_prob_fn,
    device: torch.device,
    args,
    epoch: int,
    global_step: int,
):
    """Run one training epoch and return the mean loss and updated global step."""
    model.train()
    running_loss = 0.0
    sample_count = 0

    progress = tqdm(dataloader, desc=f"Train {epoch:04d}", leave=False)
    for batch_a, batch_b, actions in progress:
        batch_a, batch_b, actions = move_batch_to_device(batch_a, batch_b, actions, device)

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for _ in range(args.repeat_num):
            loss, _ = lossFun(model, batch_a, batch_b, actions, marginal_prob_fn)
            (loss / args.repeat_num).backward()
            batch_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        if args.warmup_steps > 0:
            warmup_scale = min((global_step + 1) / args.warmup_steps, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup_scale

        optimizer.step()
        ema.update(model.parameters())

        batch_size = actions.size(0)
        mean_batch_loss = batch_loss / args.repeat_num
        running_loss += mean_batch_loss * batch_size
        sample_count += batch_size
        global_step += 1

        progress.set_postfix(loss=f"{mean_batch_loss:.6f}")

    mean_loss = running_loss / max(sample_count, 1)
    return mean_loss, global_step


def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    marginal_prob_fn,
    device: torch.device,
) -> float:
    """Evaluate the diffusion training loss on the validation split."""
    model.eval()
    running_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_a, batch_b, actions in dataloader:
            batch_a, batch_b, actions = move_batch_to_device(batch_a, batch_b, actions, device)
            loss, _ = lossFun(model, batch_a, batch_b, actions, marginal_prob_fn)
            batch_size = actions.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

    return running_loss / max(sample_count, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI for diffusion training."""
    parser = argparse.ArgumentParser(description="Train the pose diffusion model on simulated_dataset_pairs.pkl.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--sde-mode", type=str, default="ve")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--repeat-num", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--visualize-every", type=int, default=10)
    parser.add_argument("--num-steps-sample", type=int, default=128)
    return parser


def main() -> None:
    """Entry point for training."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dict = load_simulated_pair_dataset(args.dataset)
    train_pairs, val_pairs = split_positive_pairs(
        dataset_dict["positive_pair_indices"],
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_dataset = SimulatedPairDataset(
        dataset_dict["all_polys"],
        dataset_dict["all_local_features"],
        dataset_dict["all_global_features"],
        dataset_dict["all_actions"],
        train_pairs,
    )
    val_dataset = SimulatedPairDataset(
        dataset_dict["all_polys"],
        dataset_dict["all_local_features"],
        dataset_dict["all_global_features"],
        dataset_dict["all_actions"],
        val_pairs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pair_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pair_collate_fn,
        drop_last=False,
    )

    print(f"Fragments: {len(dataset_dict['all_polys'])}")
    print(f"Positive pairs: {len(dataset_dict['positive_pair_indices'])}")
    print(f"Train pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(val_dataset)}")

    _, marginal_prob_fn, sde_fn, _ = init_sde(args.sde_mode)
    model = PolygonPackingTransformer(
        marginal_prob_std_func=marginal_prob_fn,
        device=device,
    ).to(device)

    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.exists():
            load_checkpoint_compat(model, pretrained_path)
        else:
            print(f"Pretrained checkpoint not found, training from scratch: {pretrained_path}")

    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    vis_dir = output_dir / "visualizations"
    history_csv_path = output_dir / "train_history.csv"
    write_history_header(history_csv_path)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.n_epochs + 1):
        train_loss, global_step = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            ema=ema,
            marginal_prob_fn=marginal_prob_fn,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )

        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            marginal_prob_fn=marginal_prob_fn,
            device=device,
        )
        ema.restore(model.parameters())

        current_lr = optimizer.param_groups[0]["lr"]
        append_history_row(history_csv_path, epoch, train_loss, val_loss, current_lr)

        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={current_lr:.6e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            save_model_weights(model_dir / "critic_best.pth", model)
            save_checkpoint(
                model_dir / "critic_best_checkpoint.pt",
                model,
                optimizer,
                ema,
                epoch,
                global_step,
                args,
            )
            ema.restore(model.parameters())

        if epoch % args.save_every == 0:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            save_model_weights(model_dir / f"critic_epoch_{epoch}.pth", model)
            save_checkpoint(
                model_dir / f"critic_epoch_{epoch}.pt",
                model,
                optimizer,
                ema,
                epoch,
                global_step,
                args,
            )
            ema.restore(model.parameters())

        if args.visualize_every > 0 and epoch % args.visualize_every == 0:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            save_pose_visualization(
                model=model,
                sde_fn=sde_fn,
                dataset=train_dataset,
                device=device,
                epoch=epoch,
                split_name="Train",
                output_dir=vis_dir,
                num_steps_sample=args.num_steps_sample,
            )
            save_pose_visualization(
                model=model,
                sde_fn=sde_fn,
                dataset=val_dataset,
                device=device,
                epoch=epoch,
                split_name="Val",
                output_dir=vis_dir,
                num_steps_sample=args.num_steps_sample,
            )
            ema.restore(model.parameters())

    print(f"Training finished. Best validation loss: {best_val_loss:.6f}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
