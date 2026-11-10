"""Cross-modal contrastive supervision for RGB and DSM features."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDSMContrastiveLoss(nn.Module):
    """Align paired RGB/DSM tokens by position and semantic class.

    Args:
        channels: RGB/DSM feature channel count ``C``.
        projection_dim: Normalized contrastive embedding width ``D``.
        temperature: Similarity softmax temperature.
        position_weight: Relative weight of exact spatial-pair positives.
        class_weight: Relative weight of all same-class positives.
        ignore_index: Label value omitted from token sampling.
        num_classes: Valid semantic class count ``K``.
        max_samples: Maximum valid tokens selected from the batch.

    Input/Output:
        Matching RGB/DSM features ``[B,C,h,w]`` and labels ``[B,H,W]`` or
        ``[B,1,H,W]`` -> one scalar loss. Labels are nearest-resized to ``h,w``.
    """

    def __init__(
        self,
        channels: int = 256,
        projection_dim: int = 64,
        temperature: float = 0.07,
        position_weight: float = 1.0,
        class_weight: float = 1.0,
        ignore_index: int = 255,
        num_classes: int = 6,
        max_samples: int = 2048,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.position_weight = position_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.rgb_projection = nn.Linear(channels, projection_dim)
        self.dsm_projection = nn.Linear(channels, projection_dim)

    def _prepare_labels(
        self, labels: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize labels to ``size`` and flatten them to ``[B*h*w]``."""
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3:
            raise ValueError(f"labels must have shape [B, H, W], got {labels.shape}")
        if labels.shape[-2:] != size:
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1)
        return labels.long().flatten()

    def _class_balanced_indices(self, labels: torch.Tensor) -> torch.Tensor:
        """Return up to ``max_samples`` approximately class-balanced indices."""
        if labels.numel() <= self.max_samples:
            return torch.arange(labels.numel(), device=labels.device)
        present_classes = torch.unique(labels)
        quota = max(self.max_samples // present_classes.numel(), 1)
        selected = []
        for class_id in present_classes:
            class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
            order = torch.randperm(class_indices.numel(), device=labels.device)
            selected.append(class_indices[order[:quota]])
        indices = torch.cat(selected)
        if indices.numel() < self.max_samples:
            selected_mask = torch.zeros(
                labels.numel(), dtype=torch.bool, device=labels.device
            )
            selected_mask[indices] = True
            remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
            order = torch.randperm(remaining.numel(), device=labels.device)
            indices = torch.cat(
                (indices, remaining[order[: self.max_samples - indices.numel()]])
            )
        return indices[: self.max_samples]

    @staticmethod
    def _multi_positive_loss(
        logits: torch.Tensor, positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """Average log probability over every positive in each similarity row."""
        log_probabilities = F.log_softmax(logits, dim=1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        return -(
            (log_probabilities * positive_mask).sum(dim=1) / positive_count
        ).mean()

    def forward(
        self,
        rgb_feature: torch.Tensor,
        dsm_feature: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Project sampled tokens, build ``[N,N]`` similarities, and reduce loss."""
        if rgb_feature.shape != dsm_feature.shape:
            raise ValueError("RGB and DSM contrastive features must have identical shapes")
        _, channels, height, width = rgb_feature.shape
        token_labels = self._prepare_labels(labels, (height, width))
        valid = (
            token_labels.ne(self.ignore_index)
            & token_labels.ge(0)
            & token_labels.lt(self.num_classes)
        )
        if not valid.any():
            return (rgb_feature.sum() + dsm_feature.sum()) * 0.0

        rgb_tokens = rgb_feature.permute(0, 2, 3, 1).reshape(-1, channels)[valid]
        dsm_tokens = dsm_feature.permute(0, 2, 3, 1).reshape(-1, channels)[valid]
        token_labels = token_labels[valid]
        indices = self._class_balanced_indices(token_labels)
        rgb_tokens = F.normalize(self.rgb_projection(rgb_tokens[indices]), dim=-1)
        dsm_tokens = F.normalize(self.dsm_projection(dsm_tokens[indices]), dim=-1)
        token_labels = token_labels[indices]

        logits = rgb_tokens @ dsm_tokens.transpose(0, 1) / self.temperature
        position_targets = torch.arange(logits.shape[0], device=logits.device)
        position_loss = 0.5 * (
            F.cross_entropy(logits, position_targets)
            + F.cross_entropy(logits.transpose(0, 1), position_targets)
        )
        same_class = token_labels[:, None].eq(token_labels[None, :]).to(logits.dtype)
        class_loss = 0.5 * (
            self._multi_positive_loss(logits, same_class)
            + self._multi_positive_loss(logits.transpose(0, 1), same_class)
        )
        weight_sum = self.position_weight + self.class_weight
        if weight_sum <= 0:
            raise ValueError("At least one contrastive loss weight must be positive")
        return (
            self.position_weight * position_loss + self.class_weight * class_loss
        ) / weight_sum
