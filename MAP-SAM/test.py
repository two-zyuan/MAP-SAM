"""Evaluate a trained MAP-SAM C1 checkpoint on Vaihingen."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from config import MODEL_CONFIG
from experiments.scripts.ablation_model import build_ablation_model
from experiments.scripts.checkpoints import load_compact_model_state, read_checkpoint
from experiments.scripts.common import CLASS_NAMES, atomic_json, dataset, seed_everything
from utils.eval import ConfusionMatrix
from utils.reproducibility import fixed_torch_seed


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    amp = device.type == "cuda"
    seed_everything(42, device)
    checkpoint = read_checkpoint(args.checkpoint)
    if checkpoint.get("configuration") != "C1":
        raise ValueError("Expected a Vaihingen C1 checkpoint.")
    model = build_ablation_model("C1", MODEL_CONFIG.img_size, args.sam_checkpoint)
    load_compact_model_state(model, checkpoint["model"])
    model.prompt_generator.ddim.n_steps = args.ddim_steps
    model.to(device).eval()

    test_data = dataset(args.root, "test", image_size=MODEL_CONFIG.img_size)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=amp)
    predictions = args.output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    confusion = ConfusionMatrix(MODEL_CONFIG.num_classes, CLASS_NAMES)
    with fixed_torch_seed(42, device):
        for batch in loader:
            rgb = batch["img"].to(device, dtype=torch.float32, non_blocking=amp)
            dsm = batch["dsm"].to(device, dtype=torch.float32, non_blocking=amp)
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(rgb, dsm, is_train=False)[0]
            masks = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for sample_id, mask, target in zip(batch["id"], masks, batch["mask"][:, 0].numpy()):
                Image.fromarray(mask).save(predictions / f"{sample_id}.png")
                confusion.update(mask, target)

    rows, _, _, _ = confusion.metrics(range(MODEL_CONFIG.num_classes))
    _, miou, mf1, oa = confusion.metrics(range(5))
    result = dict(mIoU=miou, mF1=mf1, OA=oa, unit="percent",
                  foreground_class_ids=list(range(5)), test_images=len(test_data),
                  best_epoch=checkpoint.get("best_epoch"), ddim_steps=args.ddim_steps,
                  noise_seed=42, amp=amp)
    np.savetxt(args.output / "confusion_matrix.csv", confusion.matrix, delimiter=",", fmt="%d")
    with (args.output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "iou", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/vaihingen-c1/test"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddim-steps", type=int, default=5, choices=range(1, 1001), metavar="1-1000")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    result = evaluate(args)
    print(f"mIoU: {result['mIoU']:.3f}%  mF1: {result['mF1']:.3f}%  OA: {result['OA']:.3f}%")


if __name__ == "__main__":
    main()
