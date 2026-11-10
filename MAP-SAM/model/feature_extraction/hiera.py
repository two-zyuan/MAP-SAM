# Adapted from Meta SAM2. See model/SAM2/LICENSE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from ..common.nn_layers import DropPath, MLP


# Large fused addmm calls have triggered cuBLASLt internal errors on some CUDA
# driver/PyTorch combinations. Keep each projection below that failure shape.
_QKV_PROJECTION_CHUNK_TOKENS = 32768


"""
hiera_s
    embed_dim: 96
    num_heads: 1
    stages: [1, 2, 11, 2]
    global_att_blocks: [7, 10, 13]
    window_pos_embed_bkg_spatial_size: [7, 7]
"""


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """Pool a channel-last spatial feature and optionally normalize it.

    Args:
        x: Input feature ``[B,H,W,C]``.
        pool: Spatial BCHW pooling module, or ``None`` for identity.
        norm: Optional normalization applied after returning to channel-last form.

    Returns:
        ``[B,H',W',C]``. With the default stage stride 2, ``H'=H/2`` and
        ``W'=W/2``.
    """
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """Apply multi-head attention with optional pooled query tokens.

    Args:
        dim: Input channel width.
        dim_out: Query/key/value and output channel width.
        num_heads: Attention heads dividing ``dim_out``.
        q_pool: Optional query pooling module used at stage transitions.

    Input/Output:
        ``[B,H,W,dim] -> [B,H',W',dim_out]``. Without pooling ``H'=H`` and
        ``W'=W``; stage-transition pooling halves both spatial dimensions.
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Project QKV in bounded CUDA chunks without a fused bias epilogue."""
        token_count = x.numel() // x.shape[-1]
        if not x.is_cuda or token_count <= _QKV_PROJECTION_CHUNK_TOKENS:
            return self.qkv(x)

        flat_x = x.reshape(token_count, x.shape[-1])
        projected = flat_x.new_empty((token_count, self.dim_out * 3))
        transposed_weight = self.qkv.weight.t()
        for start in range(0, token_count, _QKV_PROJECTION_CHUNK_TOKENS):
            end = min(start + _QKV_PROJECTION_CHUNK_TOKENS, token_count)
            chunk = torch.matmul(flat_x[start:end], transposed_weight)
            if self.qkv.bias is not None:
                chunk = chunk + self.qkv.bias
            projected[start:end] = chunk
        return projected.reshape(*x.shape[:-1], self.dim_out * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attended channel-last features at the query resolution."""
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self._project_qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)


        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)  # (B, H * W, nheads, C) -> (B, H, W, n_heads * C) -> (B, H1, W1, n_heads * C)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1) # (B, H1, W1, n_heads * C) -> (B, H1 * W1, n_heads, C)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        """
            Attention = softmax(Q * K^T / sqrt(head_dim)) * V
            Q : (B, n_heads, H1 * W1, C)
            K : (B, n_heads, H * W, C)
            V : (B, n_heads, H * W, C)
            out : (B, n_heads, H1 * W1, C)
        """
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2) # (B, H1 * W1, n_heads, C)
        x = x.reshape(B, H, W, -1) # (B, H1, W1, dim_out)

        x = self.proj(x)

        return x

class MultiScaleBlock(nn.Module):
    """Hiera attention/MLP residual block with optional stage downsampling.

    Args:
        dim: Input channel width.
        dim_out: Output channel width.
        num_heads: Multi-scale attention head count.
        mlp_ratio: Hidden width multiplier for the residual MLP.
        drop_path: Per-sample residual-branch drop probability.
        norm_layer: Normalization module or a ``torch.nn`` class name.
        q_stride: Optional spatial query-pooling stride at a stage transition.
        act_layer: MLP activation class.
        window_size: Local-attention window side; zero selects global attention.

    Input/Output:
        ``[B,H,W,dim] -> [B,H',W',dim_out]``. ``q_stride=(2,2)`` produces
        ``H'=H/2`` and ``W'=W/2``; otherwise spatial size is unchanged.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(
        self,
        x: torch.Tensor,
        sca_adapter: Optional[nn.Module] = None,
        sra_adapter: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Apply attention/SCA and MLP/SRA residual updates.

        SCA runs after window unpartition and the attention residual, so its
        depth-wise convolution sees the complete ``[B,H',W',dim_out]`` grid.
        SRA runs after the MLP residual with the same spatial/channel shape.
        """
        shortcut = x
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size) # [B * num_windows, ws, ws, dim]

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        # if q_pool [B, H / 2, W / 2, dim_out] else (B, H, W, dim_out)
        x = shortcut + self.drop_path(x)
        if sca_adapter is not None:
            x = sca_adapter(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if sra_adapter is not None:
            x = sra_adapter(x)
        return x


class Hiera(nn.Module):
    """Hierarchical image transformer used by the SAM2 Hiera-S backbone.

    Args:
        embed_dim: Patch-embedding channel width of stage 1.
        num_heads: Attention-head count of stage 1.
        drop_path_rate: Maximum stochastic-depth probability across all blocks.
        q_pool: Number of stage transitions that apply query pooling.
        q_stride: Spatial stride used by query pooling.
        stages: Number of transformer blocks in each stage.
        dim_mul: Channel multiplier at every stage transition.
        head_mul: Attention-head multiplier at every stage transition.
        window_pos_embed_bkg_spatial_size: Learned background position-grid size.
        window_spec: Local-attention window side for every stage.
        global_att_blocks: Zero-based block indices using global attention.
        return_interm_layers: Return every stage end rather than only the last.

    Input/Output:
        Image ``[B,3,H,W]`` -> BCHW stage list. Hiera-S at 1024 x 1024 returns
        ``[B,96,256,256]``, ``[B,192,128,128]``, ``[B,384,64,64]``, and
        ``[B,768,32,32]``.

    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers: bool = True,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride

        # For stages (1, 2, 11, 2), stage ends are blocks (0, 2, 13, 15).
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])

        # Query pooling occurs in the first block after selected stage ends.
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]

        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:  # stage_ends = [0, 2, 13, 15]
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )


    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """Resize and combine learned positions as ``[1,H,W,C]``."""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return BCHW outputs at each configured Hiera stage end."""
        x = self.patch_embed(x)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name: str) -> int:
        """Map a parameter name to a layer-wise learning-rate-decay index."""
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        """Return the total number of transformer blocks."""
        return len(self.blocks)
