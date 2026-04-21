"""Train the pairwise pottery compatibility classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PotteryPairDataset, build_eval_transform, build_train_transform, load_samples_from_csv
from model import PairClassifier


def set_seed(seed: int = 42):
    """Set all random seeds used in training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Collate one mini-batch of paired image samples."""
    exterior = torch.stack([item["exterior"] for item in batch], dim=0)
    interior = torch.stack([item["interior"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return {
        "exterior": exterior,
        "interior": interior,
        "label": labels,
        "meta": batch,
    }


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute binary classification metrics from probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    """Run one training or validation epoch."""
    model.train(train)
    total_loss = 0.0
    y_true, y_prob = [], []

    iterator = tqdm(loader, desc="train" if train else "valid", ncols=100)
    for batch in iterator:
        exterior = batch["exterior"].to(device)
        interior = batch["interior"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(exterior, interior)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(probabilities.tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        total_loss += loss.item() * labels.size(0)

    average_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = float(average_loss)
    return metrics


def save_curve(history, output_dir: Path):
    """Save loss and accuracy curves to disk."""
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=200)
    plt.close()


def main():
    """Train the model and save the best validation checkpoint."""
    parser = argparse.ArgumentParser(description="Train a binary classifier for pottery pair compatibility.")
    parser.add_argument("--labels_csv", type=str, required=True, help="Merged label CSV.")
    parser.add_argument("--project_root", type=str, default=".", help="Root directory used to resolve image paths.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for checkpoints and metrics.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples_from_csv(args.labels_csv, args.project_root)
    if len(all_samples) < 10:
        raise ValueError("Too few samples. A dataset with at least 10 items is recommended for training.")

    labels = [sample["label"] for sample in all_samples]
    if len(set(labels)) < 2:
        raise ValueError("Training data must contain both positive and negative samples.")

    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    pd.DataFrame(train_samples).to_csv(output_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_samples).to_csv(output_dir / "val_split.csv", index=False, encoding="utf-8-sig")

    train_dataset = PotteryPairDataset(train_samples, transform=build_train_transform(args.image_size))
    val_dataset = PotteryPairDataset(val_samples, transform=build_eval_transform(args.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PairClassifier(backbone_name=args.backbone, pretrained=True).to(device)

    train_labels = np.array([sample["label"] for sample in train_samples])
    positive_count = int((train_labels == 1).sum())
    negative_count = int((train_labels == 0).sum())
    positive_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = -math.inf
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        row: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        current_metric = val_metrics["f1"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "image_size": args.image_size,
                "best_val_f1": best_metric,
                "project_root": str(Path(args.project_root).resolve()),
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"Saved best model at epoch {epoch}, val_f1={best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. patience={patience_counter}/{args.patience}")

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
        save_curve(history, output_dir)

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_metric,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_positive_train": positive_count,
        "num_negative_train": negative_count,
        "device": device,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    print("\nTraining finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
