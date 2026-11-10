"""Task-specific semantic decoder over the SAM2 feature pyramid."""

from __future__ import annotations

from typing import Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalization(norm: Type[nn.Module], channels: int) -> nn.Module:
    """Instantiate a normalization class for ``channels`` BCHW channels."""
    return norm(channels)


def _activation(act: Type[nn.Module]) -> nn.Module:
    """Instantiate an activation, enabling in-place mode for ``nn.ReLU``."""
    return act(inplace=True) if act is nn.ReLU else act()


class DecoderBlock(nn.Module):
    """Reduce channels, upsample by two, and project to an output width.

    Args:
        in_channels: Input channel count.
        out_channels: Returned channel count.
        norm: BCHW normalization class.
        act: Activation class.

    Input/Output:
        ``[B,in_channels,H,W] -> [B,out_channels,2H,2W]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_channels = max(in_channels // 4, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _normalization(norm, hidden_channels),
            _activation(act),
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            _normalization(norm, hidden_channels),
            _activation(act),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _normalization(norm, out_channels),
            _activation(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample and project one BCHW feature tensor."""
        return self.decoder(x)


class SingleClassDecoder(nn.Module):
    """Predict one semantic-class logit channel from a shared feature map.

    Args:
        in_channels: Shared decoder feature width.
        hidden_channels: Intermediate convolution width.
        norm: BCHW normalization class.
        act: Activation class.

    Input/Output:
        ``[B,in_channels,H,W] -> [B,1,H,W]``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            _normalization(norm, hidden_channels),
            _activation(act),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one class-specific logit map."""
        return self.block(x)


class ClassDecoder(nn.Module):
    """Run independent binary heads and concatenate them as semantic logits.

    Args:
        in_channels: Shared input feature width.
        num_classes: Number of independent output heads ``K``.
        hidden_channels: Width inside every class-specific head.
        norm: BCHW normalization class.
        act: Activation class.

    Input/Output:
        ``[B,in_channels,H,W] -> [B,K,H,W]``.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 6,
        hidden_channels: int = 32,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.decoders = nn.ModuleList(
            SingleClassDecoder(
                in_channels,
                hidden_channels=hidden_channels,
                norm=norm,
                act=act,
            )
            for _ in range(num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate ``K`` single-channel class predictions."""
        return torch.cat([decoder(x) for decoder in self.decoders], dim=1)


class DSMResidualFusion(nn.Module):
    """Inject one DSM FPN level into its aligned RGB level with a small gate.

    The spatial gate is predicted from RGB, DSM, and their absolute difference.
    A learnable scalar bounds the residual through ``tanh``; its 8-4 default is
    0.05, so DSM participates immediately without overwhelming RGB features.
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 64,
        residual_scale_init: float = 0.05,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.dsm_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _normalization(norm, channels),
            _activation(act),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, kernel_size=1, bias=False),
            _normalization(norm, hidden_channels),
            _activation(act),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.last_gate_mean = None

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        if rgb.shape != dsm.shape:
            raise ValueError(
                "Aligned RGB/DSM decoder features must have equal shapes, got "
                f"{tuple(rgb.shape)} and {tuple(dsm.shape)}"
            )
        gate_input = torch.cat((rgb, dsm, torch.abs(rgb - dsm)), dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        self.last_gate_mean = gate.detach().mean()
        update = gate * self.dsm_projection(dsm)
        return rgb + torch.tanh(self.residual_scale) * update

    def mean_gate(self) -> torch.Tensor:
        if self.last_gate_mean is not None:
            return self.last_gate_mean
        return torch.sigmoid(self.gate[-1].bias).mean()


class FeatureDecoder(nn.Module):
    """Fuse SAM2 FPN levels with the native MaskDecoder spatial feature.

    Args:
        num_classes: Number of returned semantic-logit channels ``K``.
        norm: BCHW normalization class used by new task layers.
        act: Activation class used by new task layers.

    Input:
        Four aligned RGB and DSM FPN tensors, all with 256 channels at strides
        4/8/16/32,
        plus ``sam_decoder_feature [B,32,H/4,W/4]``. For 1024 x 1024 these
        shapes are ``[B,256,256,256]``, ``[B,256,128,128]``,
        ``[B,256,64,64]``, ``[B,256,32,32]``, and ``[B,32,256,256]``.

    Output:
        The top-down FPN produces ``[B,64,H/4,W/4]``. Concatenation with the
        SAM feature produces ``[B,96,H/4,W/4]``; two decoder blocks return
        logits ``[B,K,H,W]``.
    """

    def __init__(
        self,
        num_classes: int = 6,
        dsm_residual_scale_init: float = 0.05,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.dsm_fusion = nn.ModuleList(
            DSMResidualFusion(
                channels=256,
                residual_scale_init=dsm_residual_scale_init,
                norm=norm,
                act=act,
            )
            for _ in range(4)
        )
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, bias=False),
                _normalization(norm, 64),
                _activation(act),
            )
            for _ in range(4)
        )
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                _normalization(norm, 64),
                _activation(act),
            )
            for _ in range(4)
        )
        self.output_decoder = nn.Sequential(
            DecoderBlock(96, 32, norm=norm, act=act),
            DecoderBlock(32, 32, norm=norm, act=act),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            _normalization(norm, 32),
            _activation(act),
        )
        self.class_decoder = ClassDecoder(
            in_channels=32,
            num_classes=num_classes,
            hidden_channels=32,
            norm=norm,
            act=act,
        )

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        dsm_features: Sequence[torch.Tensor],
        sam_decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-resolution semantic logits from FPN and SAM features."""
        if len(rgb_features) != 4 or len(dsm_features) != 4:
            raise ValueError(
                "FeatureDecoder expects four RGB and four DSM features, got "
                f"{len(rgb_features)} and {len(dsm_features)}"
            )
        fused_features = [
            fusion(rgb, dsm)
            for fusion, rgb, dsm in zip(
                self.dsm_fusion, rgb_features, dsm_features
            )
        ]
        pyramid = [
            lateral(feature)
            for lateral, feature in zip(self.lateral, fused_features)
        ]
        top_down = self.refine[3](pyramid[3])
        for index in (2, 1, 0):
            top_down = F.interpolate(
                top_down,
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            top_down = self.refine[index](pyramid[index] + top_down)
        if sam_decoder_feature.shape[-2:] != top_down.shape[-2:]:
            sam_decoder_feature = F.interpolate(
                sam_decoder_feature,
                size=top_down.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        decoded = self.output_decoder(
            torch.cat((top_down, sam_decoder_feature), dim=1)
        )
        return self.class_decoder(decoded)

    def mean_dsm_weights(self) -> torch.Tensor:
        """Return the most recent mean spatial DSM gate at each FPN level."""
        return torch.stack([fusion.mean_gate() for fusion in self.dsm_fusion])

    def dsm_residual_scales(self) -> torch.Tensor:
        """Return bounded residual scales for the four decoder fusion levels."""
        return torch.stack(
            [torch.tanh(fusion.residual_scale) for fusion in self.dsm_fusion]
        )
