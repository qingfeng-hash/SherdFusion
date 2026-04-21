"""Run inference for the pairwise pottery compatibility classifier."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PotteryPairDataset, build_eval_transform, scan_infer_folder
from model import PairClassifier


def collate_fn(batch):
    """Collate inference samples into one mini-batch."""
    exterior = torch.stack([item["exterior"] for item in batch], dim=0)
    interior = torch.stack([item["interior"] for item in batch], dim=0)
    return {
        "exterior": exterior,
        "interior": interior,
        "meta": batch,
    }


@torch.no_grad()
def main():
    """Load a checkpoint and run batch inference on one folder."""
    parser = argparse.ArgumentParser(description="Run batch inference for pairwise pottery compatibility.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing *_exterior and *_interior images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained best_model.pt checkpoint.")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV for inference results.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the positive class.")
    args = parser.parse_args()

    samples = scan_infer_folder(args.input_dir)
    if not samples:
        raise ValueError("No complete exterior/interior image pairs were found in the input folder.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = checkpoint.get("backbone", "resnet18")
    image_size = checkpoint.get("image_size", 224)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PairClassifier(backbone_name=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = PotteryPairDataset(samples, transform=build_eval_transform(image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    rows = []
    for batch in tqdm(dataloader, desc="infer", ncols=100):
        exterior = batch["exterior"].to(device)
        interior = batch["interior"].to(device)
        logits = model(exterior, interior)
        probabilities = torch.sigmoid(logits).cpu().numpy().tolist()

        for meta, probability in zip(batch["meta"], probabilities):
            predicted_label = 1 if probability >= args.threshold else 0
            rows.append(
                {
                    "group_name": meta["group_name"],
                    "exterior_image": meta["exterior_image"],
                    "interior_image": meta["interior_image"],
                    "exterior_path": meta["exterior_path"],
                    "interior_path": meta["interior_path"],
                    "prob_success": round(float(probability), 6),
                    "score_0_100": round(float(probability) * 100.0, 2),
                    "pred_label": predicted_label,
                    "pred_text": "compatible" if predicted_label == 1 else "incompatible",
                }
            )

    dataframe = pd.DataFrame(rows).sort_values(by="prob_success", ascending=False)
    dataframe.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"Inference finished. Processed {len(dataframe)} pairs.")
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
