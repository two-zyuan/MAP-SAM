"""Shared paths, configuration loading, data, and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from config import MODEL_CONFIG, TRAINING_CONFIG
from utils.DataSet import PotsdamDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "configurations.json"
DSM_STATS = (283.86812541651716, 22.373971373014403)
CLASS_NAMES = (
    "Impervious surfaces", "Building", "Low vegetation", "Tree", "Car",
    "Clutter/background",
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    stage: str
    train: bool
    modalities: str
    sca: bool
    sra: bool
    cross_direction: str
    contrastive: str
    prompt: str
    decoder: str
    description: str
    full_alias: bool = False

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(name=name, **values)


def load_configs() -> Dict[str, ExperimentConfig]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {name: ExperimentConfig.from_mapping(name, values) for name, values in raw.items()}


def get_config(name: str) -> ExperimentConfig:
    configs = load_configs()
    key = name.upper()
    if key not in configs:
        raise KeyError(f"Unknown experiment configuration: {name}")
    return configs[key]


def atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def dataset(root: Path, split: str, training: bool = False, image_size: int | None = None) -> PotsdamDataset:
    return PotsdamDataset(
        root=str(root),
        query=split,
        if_transform=training,
        image_size=image_size or MODEL_CONFIG.img_size,
        dsm_normalization=TRAINING_CONFIG.dsm_normalization,
        dsm_stats=DSM_STATS,
        dsm_clip=TRAINING_CONFIG.dsm_clip,
        augmentation_policy=TRAINING_CONFIG.augmentation_policy if training else "none",
        crop_scale=TRAINING_CONFIG.augmentation_crop_scale,
        hflip_prob=TRAINING_CONFIG.augmentation_hflip_prob,
        vflip_prob=TRAINING_CONFIG.augmentation_vflip_prob,
        rot90_prob=TRAINING_CONFIG.augmentation_rot90_prob,
        color_jitter_prob=TRAINING_CONFIG.augmentation_color_jitter_prob,
        color_jitter_strength=TRAINING_CONFIG.augmentation_color_jitter_strength,
    )

