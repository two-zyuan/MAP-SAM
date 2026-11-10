"""Shared frozen SAM2 encoder for RGB and DSM remote-sensing inputs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if __package__ in {None, ""}:
    # Support direct execution: python model/feature_extraction/....py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from model.feature_extraction.multimodal_blocks import (
        CrossModalQuery,
        SemanticResponseAdapter,
        SpatialContextAdapter,
    )
    from model.feature_extraction.sam2_image_encoder import (
        build_hiera_small_image_encoder,
    )
else:
    from .multimodal_blocks import (
        CrossModalQuery,
        SemanticResponseAdapter,
        SpatialContextAdapter,
    )
    from .sam2_image_encoder import build_hiera_small_image_encoder


DEFAULT_MODEL_CONFIG = "hiera_small"
_SAM2_ROOT = Path(__file__).resolve().parents[1] / "SAM2"
DEFAULT_CHECKPOINT = _SAM2_ROOT / "checkpoints" / "sam2_hiera_small.pt"


class SAM2Encoder(nn.Module):
    """Extract and fuse RGB/DSM features with a shared frozen SAM2 backbone.

    Args:
        model_config: Backbone variant name. This implementation supports only
            ``"hiera_small"`` because its modules match the bundled checkpoint.
        checkpoint: SAM2 checkpoint path. ``None`` uses the bundled Hiera-S file.
        adapter_reduction: Bottleneck ratio used by every modality adapter.
        sca_kernel_sizes: Parallel depth-wise convolution kernels used by SCA.
        sra_expansion_ratio: Expansion from ``r/2`` inside SRA; ``4`` gives
            the update specification's ``r/2 -> 2r -> r/2`` path.
        adapter_gamma_init: Initial per-output-channel external scale.
        adapter_inner_scale_init: Initial per-operated-channel internal scale.
        cross_attention_heads_list: Attention-head count for Hiera stages 1-4.
        cross_kv_pool_sizes: Maximum key/value grid size for stages 1-4; each
            entry may be ``None`` to retain that stage's full grid.
        cross_query_chunk_size: Maximum query tokens processed per attention
            call, limiting memory without changing numerical results.
        cross_stages: One-based Hiera stages where RGB/DSM cross attention runs.

    Input:
        RGB ``[B,3,H,W]`` and DSM ``[B,1,H,W]``, ``[B,2,H,W]``, or
        ``[B,3,H,W]``. The two-channel form supports absolute/relative DSM.

    Output:
        A dictionary containing RGB/DSM FPN lists. For a 1024 x 1024 input,
        each list contains ``[B,256,256,256]``, ``[B,256,128,128]``,
        ``[B,256,64,64]``, and ``[B,256,32,32]``. ``rgb_last`` and
        ``dsm_last`` select the final ``[B,256,H/32,W/32]`` tensors.
    """

    def __init__(
        self,
        model_config: str = DEFAULT_MODEL_CONFIG,
        checkpoint: Optional[str] = None,
        adapter_reduction: float = 0.25,
        sca_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        sra_expansion_ratio: float = 4.0,
        sra_global_scale_init: float = 0.05,
        adapter_gamma_init: float = 1.0,
        adapter_inner_scale_init: float = 0.05,
        cross_modal_scale_init: float = 0.05,
        cross_attention_heads_list: Tuple[int, ...] = (1, 2, 4, 8),
        cross_kv_pool_sizes: Tuple[Optional[int], ...] = (8, 8, 16, 32),
        cross_query_chunk_size: Optional[int] = 4096,
        cross_stages: Tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT


        if model_config != DEFAULT_MODEL_CONFIG:
            raise ValueError(
                f"Only '{DEFAULT_MODEL_CONFIG}' is supported, got '{model_config}'"
            )
        self.image_encoder = build_hiera_small_image_encoder()
        self._load_image_encoder(checkpoint_path)
        self.stage_end_indices = tuple(self.trunk.stage_ends)
        self.stage_ends = set(self.stage_end_indices)
        invalid_cross_stages = sorted(set(cross_stages) - set(range(1, len(self.stage_end_indices) + 1)))
        if invalid_cross_stages:
            raise ValueError(
                f"cross_stages contains invalid stage numbers: {invalid_cross_stages}"
            )
        self.cross_stages = frozenset(cross_stages)

        self._freeze_sam2()
        self.dsm_to_in = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1),
        )
        # Kept separate so old checkpoints using the one-channel adapter remain
        # loadable; this branch is used only by the absolute+relative ablation.
        self.dsm_to_in_2 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1),
        )

        block_channels = [block.dim_out for block in self.trunk.blocks]

        def build_sca(channels: int) -> SpatialContextAdapter:
            return SpatialContextAdapter(
                channels,
                reduction=adapter_reduction,
                kernel_sizes=sca_kernel_sizes,
                gamma_init=adapter_gamma_init,
                inner_scale_init=adapter_inner_scale_init,
            )

        def build_sra(channels: int) -> SemanticResponseAdapter:
            return SemanticResponseAdapter(
                channels,
                reduction=adapter_reduction,
                expansion_ratio=sra_expansion_ratio,
                gamma_init=adapter_gamma_init,
                inner_scale_init=adapter_inner_scale_init,
                global_scale_init=sra_global_scale_init,
            )

        # Each frozen Hiera block receives independent RGB/DSM adapters at its
        # attention and MLP insertion points. They live outside the native SAM2
        # image encoder so official image-encoder weights still load strictly.
        self.rgb_sca_adapters = nn.ModuleList(
            [build_sca(channels) for channels in block_channels]
        )
        self.rgb_sra_adapters = nn.ModuleList(
            [build_sra(channels) for channels in block_channels]
        )
        self.dsm_sca_adapters = nn.ModuleList(
            [build_sca(channels) for channels in block_channels]
        )
        self.dsm_sra_adapters = nn.ModuleList(
            [build_sra(channels) for channels in block_channels]
        )
        stage_channels = [block_channels[index] for index in self.stage_end_indices]
        if len(cross_attention_heads_list) != len(stage_channels):
            raise ValueError(
                "cross_attention_heads_list must contain one value per SAM2 stage "
                f"({len(stage_channels)} values required)"
            )
        if len(cross_kv_pool_sizes) != len(stage_channels):
            raise ValueError(
                "cross_kv_pool_sizes must contain one value per SAM2 stage "
                f"({len(stage_channels)} values required)"
            )
        self.cross_query_list = nn.ModuleList(
            [
                CrossModalQuery(
                    channels,
                    num_heads=heads,
                    kv_pool_size=pool_size,
                    query_chunk_size=cross_query_chunk_size,
                    residual_scale_init=cross_modal_scale_init,
                )
                for channels, heads, pool_size in zip(
                    stage_channels, cross_attention_heads_list, cross_kv_pool_sizes
                )
            ]
        )

    def _load_image_encoder(self, checkpoint: Path) -> None:
        """Strictly load the ``image_encoder.*`` subset of a SAM2 checkpoint."""
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint was not found at '{checkpoint}'")
        try:
            checkpoint_data = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint_data = torch.load(checkpoint, map_location="cpu")
        state_dict = checkpoint_data.get("model", checkpoint_data)
        prefix = "image_encoder."
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise RuntimeError(
                f"Checkpoint '{checkpoint}' contains no parameters under '{prefix}'"
            )
        self.image_encoder.load_state_dict(encoder_state, strict=True)

    @property
    def output_channels(self) -> int:
        """Return the shared FPN channel count, currently 256."""
        return self.neck.d_model

    @property
    def trunk(self) -> nn.Module:
        """Expose the backbone trunk without registering a second module path."""
        return self.image_encoder.trunk

    @property
    def neck(self) -> nn.Module:
        """Expose the FPN neck without registering a second module path."""
        return self.image_encoder.neck

    def _freeze_sam2(self) -> None:
        """Freeze the pretrained trunk/neck while leaving adapters trainable."""
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    def train(self, mode: bool = True) -> "SAM2Encoder":
        """Set adapter mode but keep the frozen SAM2 backbone in evaluation mode."""
        super().train(mode)
        # Frozen SAM2 must not enable stochastic depth when the task heads train.
        self.image_encoder.eval()
        return self

    def _prepare_dsm(self, dsm: torch.Tensor) -> torch.Tensor:
        """Convert DSM ``[B,H,W]``/``[B,1,H,W]`` to ``[B,3,H,W]``."""
        if dsm.ndim == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.ndim != 4:
            raise ValueError(f"DSM must have shape [B, 1, H, W], got {tuple(dsm.shape)}")
        if dsm.shape[1] == 1:
            return self.dsm_to_in(dsm)
        if dsm.shape[1] == 2:
            return self.dsm_to_in_2(dsm)
        if dsm.shape[1] == 3:
            return dsm
        raise ValueError(f"DSM must contain 1, 2, or 3 channels, got {dsm.shape[1]}")

    def _patch_embed(self, image: torch.Tensor) -> torch.Tensor:
        """Map ``[B,3,H,W]`` to Hiera tokens ``[B,H/4,W/4,96]``."""
        x = self.trunk.patch_embed(image)
        return x + self.trunk._get_pos_embed(x.shape[1:3])

    def forward(
        self, rgb: torch.Tensor, dsm: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """Return four-level RGB and DSM feature pyramids with 256 channels."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
        dsm = self._prepare_dsm(dsm)
        if rgb.shape[0] != dsm.shape[0] or rgb.shape[-2:] != dsm.shape[-2:]:
            raise ValueError("RGB and DSM must have matching batch and spatial dimensions")

        rgb_x = self._patch_embed(rgb)
        dsm_x = self._patch_embed(dsm)
        rgb_stages: List[torch.Tensor] = []
        dsm_stages: List[torch.Tensor] = []
        stage_index = 0

        for index, block in enumerate(self.trunk.blocks):
            rgb_x = block(
                rgb_x,
                sca_adapter=self.rgb_sca_adapters[index],
                sra_adapter=self.rgb_sra_adapters[index],
            )
            dsm_x = block(
                dsm_x,
                sca_adapter=self.dsm_sca_adapters[index],
                sra_adapter=self.dsm_sra_adapters[index],
            )

            if index in self.stage_ends:
                if stage_index + 1 in self.cross_stages:
                    rgb_x, dsm_x = self.cross_query_list[stage_index](rgb_x, dsm_x)
                rgb_stages.append(rgb_x.permute(0, 3, 1, 2).contiguous())
                dsm_stages.append(dsm_x.permute(0, 3, 1, 2).contiguous())
                stage_index += 1

        rgb_fpn, _ = self.neck(rgb_stages)
        dsm_fpn, _ = self.neck(dsm_stages)
        return {
            "rgb_fpn": rgb_fpn,
            "dsm_fpn": dsm_fpn,
            "rgb_last": rgb_fpn[-1],
            "dsm_last": dsm_fpn[-1],
        }


# Backward-compatible name used by the original project draft.
SAM2_Encoder = SAM2Encoder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SAM2Encoder(cross_stages=(3,4))
    encoder = encoder.to(device)
    rgb_input = torch.randn(1, 3, 1024, 1024)
    dsm_input = torch.randn(1, 1, 1024, 1024)
    rgb_input = rgb_input.to(device)
    dsm_input = dsm_input.to(device)
    output = encoder(rgb_input, dsm_input)
    print([tuple(feature.shape) for feature in output["rgb_fpn"]])
