"""Compact ablation checkpoints that reconstruct immutable SAM2 weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch


FROZEN_PREFIX = "image_encoder.image_encoder."


def compact_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if not key.startswith(FROZEN_PREFIX)}


def load_compact_model_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = {key for key in model.state_dict() if not key.startswith(FROZEN_PREFIX)}
    actual = set(state)
    if expected != actual:
        raise RuntimeError(f"Compact checkpoint mismatch: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    incompatible = model.load_state_dict(dict(state), strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith(FROZEN_PREFIX)]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint load mismatch: missing={bad_missing}, unexpected={incompatible.unexpected_keys}")


def read_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict) or "model" not in value:
        raise TypeError(f"Invalid ablation checkpoint: {path}")
    return value

