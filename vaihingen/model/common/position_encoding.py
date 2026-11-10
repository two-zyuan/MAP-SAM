"""Position encodings required by the local SAM2 backbone and prompt encoder.

Adapted from Meta SAM2. See ``model/SAM2/LICENSE``.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """Create deterministic 2-D sine/cosine embeddings for FPN features.

    Args:
        num_pos_feats: Total returned channel count; must be even.
        temperature: Frequency progression base.
        normalize: Normalize x/y coordinates to the configured scale.
        scale: Maximum normalized coordinate; defaults to ``2*pi``.

    Input/Output:
        Feature ``[B, C, H, W]`` -> position encoding
        ``[B, num_pos_feats, H, W]``.
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_pos_feats % 2 != 0:
            raise ValueError("num_pos_feats must be even")
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be enabled when scale is provided")
        self.scale = 2 * math.pi if scale is None else scale
        self.cache = {}

    @torch.no_grad()
    def _position_encoding(
        self, batch_size: int, device: torch.device, height: int, width: int
    ) -> torch.Tensor:
        """Return cached/generated positions ``[B,num_pos_feats,H,W]``."""
        cache_key = (height, width)
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(batch_size, 1, 1, 1)

        y_embed = torch.arange(
            1, height + 1, dtype=torch.float32, device=device
        ).view(1, -1, 1).repeat(batch_size, 1, width)
        x_embed = torch.arange(
            1, width + 1, dtype=torch.float32, device=device
        ).view(1, 1, -1).repeat(batch_size, height, 1)
        if self.normalize:
            eps = 1.0e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        position = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = position[0].cpu()
        return position

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a position tensor matching the BCHW batch and spatial size."""
        return self._position_encoding(
            x.shape[0], x.device, x.shape[-2], x.shape[-1]
        )


class PositionEmbeddingRandom(nn.Module):
    """Encode normalized 2-D coordinates with random Fourier features.

    Args:
        num_pos_feats: Features per sine/cosine branch. The returned channel
            dimension is ``2 * num_pos_feats``.
        scale: Standard-deviation multiplier of the fixed Gaussian matrix.

    Shapes:
        Dense grid ``(H, W) -> [2*num_pos_feats, H, W]``. Explicit pixel
        coordinates ``[B, N, 2] -> [B, N, 2*num_pos_feats]``.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        scale = 1.0 if scale is None or scale <= 0.0 else scale
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map normalized ``[...,2]`` coordinates to ``[...,2*num_pos_feats]``."""
        coordinates = 2 * coordinates - 1
        coordinates = coordinates @ self.positional_encoding_gaussian_matrix
        coordinates = 2 * np.pi * coordinates
        return torch.cat((torch.sin(coordinates), torch.cos(coordinates)), dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Create dense positions ``[2*num_pos_feats,H,W]`` for ``size``."""
        height, width = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((height, width), device=device, dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / height
        x_embed = (grid.cumsum(dim=1) - 0.5) / width
        position = self._encode(torch.stack((x_embed, y_embed), dim=-1))
        return position.permute(2, 0, 1)

    def forward_with_coords(
        self, coordinates: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Encode pixel coordinates ``[B,N,2]`` as ``[B,N,2*num_pos_feats]``."""
        coordinates = coordinates.clone()
        coordinates[:, :, 0] /= image_size[1]
        coordinates[:, :, 1] /= image_size[0]
        return self._encode(coordinates.float())
