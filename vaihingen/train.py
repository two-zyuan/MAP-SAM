"""Train the full MAP-SAM C1 configuration on Vaihingen."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config import MODEL_CONFIG, TRAINING_CONFIG
from utils.eval import eval_net
from utils.loss import ClsLoss
from experiments.scripts.ablation_model import build_ablation_model
from experiments.scripts.checkpoints import compact_model_state, load_compact_model_state, read_checkpoint
from experiments.scripts.common import DSM_STATS, atomic_json, dataset, get_config, restore_rng_state, rng_state, seed_everything
from utils.DataSet import PotsdamDataset


def auxiliary_weights(epoch: int) -> tuple[float, float, float]:
    start, end = TRAINING_CONFIG.auxiliary_decay_start_epoch, TRAINING_CONFIG.auxiliary_decay_end_epoch
    progress = 0.0 if epoch <= start else 1.0 if epoch >= end else (epoch - start) / (end - start)
    interpolate = lambda initial, final: initial + progress * (final - initial)
    return (
        interpolate(TRAINING_CONFIG.diffusion_weight, TRAINING_CONFIG.diffusion_weight_final),
        interpolate(TRAINING_CONFIG.prompt_segmentation_weight, TRAINING_CONFIG.prompt_segmentation_weight_final),
        interpolate(TRAINING_CONFIG.contrastive_weight, TRAINING_CONFIG.contrastive_weight_final),
    )


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()], force=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Prepared Vaihingen data directory")
    parser.add_argument("--sam-checkpoint", required=True, type=Path, help="SAM 2 Hiera-S initialization weights")
    parser.add_argument("--output", type=Path, default=Path("runs/vaihingen-c1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    experiment = get_config("C1")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    configure_logging(output / "train.log")
    device = torch.device(args.device)
    seed_everything(42, device)
    model = build_ablation_model(experiment.name, MODEL_CONFIG.img_size, args.sam_checkpoint).to(device)
    pretrained_ids = {id(p) for module in (model.prompt_encoder, model.mask_decoder) for p in module.parameters() if p.requires_grad}
    pretrained = [p for p in model.parameters() if p.requires_grad and id(p) in pretrained_ids]
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids]
    optimizer = optim.AdamW([{"params": new, "lr": 1.6e-4}, {"params": pretrained, "lr": 1.6e-5}], weight_decay=TRAINING_CONFIG.weight_decay)
    warm = TRAINING_CONFIG.warm_epochs
    def factor(index: int) -> float:
        if index < warm:
            return 0.02 + 0.95 * index / warm
        progress = (index - warm) / max(100 - warm, 1)
        return ((1 + math.cos(progress * math.pi)) / 2) * 0.95 + 0.02
    scheduler = lr_scheduler.LambdaLR(optimizer, factor)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_weights = PotsdamDataset.compute_class_weights(str(args.root), "train", MODEL_CONFIG.num_classes, TRAINING_CONFIG.class_weight_mode)
    criterion = ClsLoss(ignore_index=255, weight=class_weights, boundary_weight=TRAINING_CONFIG.boundary_pixel_weight)
    train_loader = DataLoader(dataset(args.root, "train", training=True), batch_size=2, shuffle=True, num_workers=args.num_workers, pin_memory=amp)
    val_loader = DataLoader(dataset(args.root, "val"), batch_size=2, shuffle=False, num_workers=args.num_workers, pin_memory=amp)
    best, best_epoch, start, no_improvement = float("-inf"), None, 0, 0
    history = []
    last_path, best_path = output / "last.pth", output / "best.pth"
    if last_path.is_file():
        checkpoint = read_checkpoint(last_path)
        if checkpoint.get("configuration") != experiment.name:
            raise RuntimeError("Resume checkpoint belongs to another configuration")
        load_compact_model_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = int(checkpoint["epoch"]) + 1
        best, best_epoch = float(checkpoint["best_metric"]), checkpoint.get("best_epoch")
        no_improvement, history = int(checkpoint["no_improvement_epochs"]), checkpoint.get("history", [])
        restore_rng_state(checkpoint["rng_state"], device)
        logging.info("Resuming %s at epoch %d", experiment.name, start + 1)
    logging.info("Config=%s description=%s train=%d val=%d device=%s", experiment.name, experiment.description, len(train_loader.dataset), len(val_loader.dataset), device)
    for epoch_index in range(start, 100):
        epoch_number = epoch_index + 1
        diffusion_weight, prompt_weight, contrastive_weight = auxiliary_weights(epoch_number)
        model.train()
        totals = {key: 0.0 for key in ("total", "segmentation", "diffusion", "prompt", "contrastive")}
        for batch in train_loader:
            image = batch["img"].to(device, dtype=torch.float32, non_blocking=True)
            dsm = batch["dsm"].to(device, dtype=torch.float32, non_blocking=True)
            target = batch["mask"].to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # DDIM runs without gradients before the supervised denoising pass.
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp, cache_enabled=False):
                logits, _, diffusion, contrastive, prompt = model(image, dsm, target, True)
                segmentation = criterion(logits, target[:, 0])
                loss = segmentation + diffusion_weight * diffusion + prompt_weight * prompt + contrastive_weight * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch_number}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(new + pretrained, TRAINING_CONFIG.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            for key, value in zip(totals, (loss, segmentation, diffusion, prompt, contrastive)):
                totals[key] += float(value.detach())
        val_miou, val_mf1, val_oa, _, _ = eval_net(model, val_loader, device, False, noise_seed=42)
        scheduler.step()
        row = {"epoch": epoch_number, "lr": optimizer.param_groups[0]["lr"], **{key: value / len(train_loader) for key, value in totals.items()}, "val_mIoU": val_miou, "val_mF1": val_mf1, "val_OA": val_oa}
        history.append(row)
        improved = val_miou > best
        if improved:
            best, best_epoch = val_miou, epoch_number
        if val_miou > best - TRAINING_CONFIG.early_stopping_min_delta:
            no_improvement = 0 if improved else no_improvement + 1
        else:
            no_improvement += 1
        base = {
            "checkpoint_version": 1, "configuration": experiment.name, "experiment": experiment.__dict__,
            "model": compact_model_state(model), "model_config": {**MODEL_CONFIG.to_dict(), "img_size": MODEL_CONFIG.img_size},
            "training_config": TRAINING_CONFIG.to_dict(), "data_config": {"dsm_stats": DSM_STATS, "dsm_normalization": TRAINING_CONFIG.dsm_normalization},
            "epoch": epoch_index, "best_metric": best, "best_epoch": best_epoch, "no_improvement_epochs": no_improvement, "history": history,
        }
        if improved:
            torch.save(base, best_path)
        last = {**base, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng_state": rng_state(device)}
        torch.save(last, last_path)
        atomic_json(output / "progress.json", {"configuration": experiment.name, "epoch": epoch_number, "best_epoch": best_epoch, "best_val_mIoU": best, "no_improvement_epochs": no_improvement, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        logging.info("epoch=%d loss=%.6f val_mIoU=%.4f best=%.4f", epoch_number, row["total"], val_miou, best)
        if no_improvement >= TRAINING_CONFIG.early_stopping_patience:
            logging.info("Early stopping at epoch %d", epoch_number)
            break
    atomic_json(output / "training_complete.json", {"configuration": experiment.name, "best_epoch": best_epoch, "best_val_mIoU": best, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


if __name__ == "__main__":
    main()
