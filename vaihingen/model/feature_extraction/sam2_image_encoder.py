# Adapted from Meta SAM2. See model/SAM2/LICENSE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hiera import Hiera
from ..common.position_encoding import PositionEmbeddingSine


class ImageEncoder(nn.Module):
    """Compose a Hiera trunk and FPN neck into the native SAM2 image encoder.

    Args:
        trunk: Hiera module returning four BCHW stage tensors.
        neck: FPN module projecting all stages to 256 channels.
        scalp: Number of lowest-resolution FPN levels removed from public output.

    Input/Output:
        Image ``[B,3,H,W]`` -> dictionary containing ``backbone_fpn`` and
        matching positional encodings. The configured ``scalp=1`` native path
        exposes strides 4, 8, and 16, while MulMoSAM accesses the neck directly
        to retain all four levels.
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        """Encode ``[B,3,H,W]`` and return SAM2 vision/FPN feature dictionaries."""
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """Project Hiera stages to one channel width and apply top-down fusion.

    Args:
        position_encoding: Module mapping each BCHW FPN tensor to equal-size
            positional features.
        d_model: Output channels for every FPN level, currently 256.
        backbone_channel_list: Expected stage channels in low-to-high resolution
            processing order: ``[768,384,192,96]`` for Hiera-S.
        kernel_size: Lateral projection convolution kernel size.
        stride: Lateral projection convolution stride.
        padding: Lateral projection convolution padding.
        fpn_interp_model: Interpolation mode for top-down resizing.
        fuse_type: Combine lateral/top-down features using ``"sum"`` or ``"avg"``.
        fpn_top_down_levels: Stage indices receiving top-down propagation.

    Input/Output:
        Four Hiera tensors at strides 4/8/16/32 -> four 256-channel tensors
        with the same spatial sizes plus equally shaped positional encodings.
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        """Fuse four BCHW stage features and return ``(features, positions)``."""

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


def build_hiera_small_image_encoder() -> ImageEncoder:
    """Build the exact Hiera-S + 256-channel FPN used by the bundled weights."""
    trunk = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
        return_interm_layers=True,
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
        ),
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)
