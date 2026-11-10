"""Windowing and patch-embedding helpers for Hiera.

Adapted from Meta SAM2. See ``model/SAM2/LICENSE``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad and split a channel-last feature map into square windows.

    Args:
        x: Hiera feature ``[B, H, W, C]``.
        window_size: Spatial side length ``S`` of each attention window.

    Returns:
        Windows ``[B*num_windows, S, S, C]`` and padded size ``(Hp, Wp)``.
    """
    batch_size, height, width, channels = x.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height or pad_width:
        x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))
    padded_height = height + pad_height
    padded_width = width + pad_width
    x = x.view(
        batch_size,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window_size, window_size, channels
    )
    return windows, (padded_height, padded_width)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded_size: Tuple[int, int],
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """Merge local windows and remove padding added by ``window_partition``.

    Args:
        windows: ``[B*num_windows, S, S, C]`` window features.
        window_size: Window side length ``S``.
        padded_size: Padded spatial size ``(Hp, Wp)``.
        original_size: Desired unpadded size ``(H, W)``.

    Returns:
        Reconstructed channel-last feature ``[B, H, W, C]``.
    """
    padded_height, padded_width = padded_size
    height, width = original_size
    batch_size = windows.shape[0] // (
        padded_height * padded_width // window_size // window_size
    )
    x = windows.reshape(
        batch_size,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        batch_size, padded_height, padded_width, -1
    )
    return x[:, :height, :width, :]


class PatchEmbed(nn.Module):
    """Convert an image tensor into a channel-last Hiera patch grid.

    Args:
        kernel_size: Convolution kernel height and width.
        stride: Spatial downsampling stride.
        padding: Zero-padding height and width.
        in_chans: Input image channels, normally three.
        embed_dim: Output embedding channels.

    Input/Output:
        ``[B, in_chans, H, W] -> [B, H/stride_h, W/stride_w, embed_dim]``
        for divisible inputs. The default maps 1024 x 1024 to 256 x 256.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (7, 7),
        stride: Tuple[int, int] = (4, 4),
        padding: Tuple[int, int] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project BCHW input and return a channel-last patch grid."""
        return self.proj(x).permute(0, 2, 3, 1)
