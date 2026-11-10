"""Trainable AiRS adapters and RGB/DSM cross-modal attention blocks.

Shape symbols: ``B`` is batch size, ``C``/``d`` is the Hiera embedding width,
``H``/``W`` are feature-grid dimensions, and ``r`` is the adapter bottleneck.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitBottleneckAdapter(nn.Module):
    """Shared down-project, split, operator, and up-project adapter structure.

    Input/output shape is ``[B,H,W,d]``. The down projection produces
    ``[B,H,W,r]`` and splits it into two ``r/2`` branches. Only one branch is
    processed; the other is an information-preserving bypass. The adapter owns
    its external residual, so callers should use ``x = adapter(x)`` directly.
    """

    def __init__(
        self,
        embed_dim: int,
        reduction: float = 0.25,
        gamma_init: float = 1.0,
        inner_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        bottleneck_dim = max(16, int(embed_dim * reduction))
        if bottleneck_dim % 2 != 0:
            bottleneck_dim += 1
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.branch_dim = bottleneck_dim // 2
        self.down_projection = nn.Linear(embed_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_projection = nn.Linear(bottleneck_dim, embed_dim)
        self.gamma = nn.Parameter(torch.full((embed_dim,), gamma_init))
        self.inner_scale = nn.Parameter(
            torch.full((self.branch_dim,), inner_scale_init)
        )

        # Start as an exact identity while preserving the complete trainable path.
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def _operator(self, branch: torch.Tensor) -> torch.Tensor:
        """Process ``[B,H,W,r/2]`` without changing its shape."""
        raise NotImplementedError

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Return ``feature + gamma * Adapter(feature)`` with unchanged shape."""
        if feature.ndim != 4 or feature.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected [B,H,W,{self.embed_dim}], got {tuple(feature.shape)}"
            )
        projected = self.down_projection(feature)
        bypass, operated = projected.chunk(2, dim=-1)
        operated = operated + self.inner_scale * self._operator(operated)
        merged = torch.cat((operated, bypass), dim=-1)
        update = self.up_projection(self.activation(merged))
        return feature + self.gamma * update


class SpatialContextAdapter(SplitBottleneckAdapter):
    """Multi-scale SCA with gated 3x3, 5x5, and 7x7 depth-wise branches.

    The operated branch is converted from ``[B,H,W,r/2]`` to
    ``[B,r/2,H,W]``, processed by three parallel depth-wise convolutions, and
    converted back. The internal residual is handled by the base class.
    """

    def __init__(
        self,
        embed_dim: int,
        reduction: float = 0.25,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        gamma_init: float = 1.0,
        inner_scale_init: float = 0.05,
    ) -> None:
        if tuple(kernel_sizes) != (3, 5, 7):
            raise ValueError("SCA kernel_sizes must be exactly (3, 5, 7)")
        super().__init__(embed_dim, reduction, gamma_init, inner_scale_init)
        self.depthwise_convs = nn.ModuleList(
            nn.Conv2d(
                self.branch_dim,
                self.branch_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=self.branch_dim,
            )
            for kernel_size in kernel_sizes
        )
        # Uniform per-channel mixing keeps the adapter identity-initialized.
        self.scale_logits = nn.Parameter(torch.zeros(len(kernel_sizes), self.branch_dim))

    def mean_scale_weights(self) -> torch.Tensor:
        """Return mean 3x3/5x5/7x7 gate weights for logging."""
        return torch.softmax(self.scale_logits, dim=0).detach().mean(dim=1)

    def _operator(self, branch: torch.Tensor) -> torch.Tensor:
        branch = branch.permute(0, 3, 1, 2).contiguous()
        weights = torch.softmax(self.scale_logits, dim=0)
        branch = sum(
            weight.view(1, -1, 1, 1) * conv(branch)
            for weight, conv in zip(weights, self.depthwise_convs)
        )
        return branch.permute(0, 2, 3, 1).contiguous()


class SemanticResponseAdapter(SplitBottleneckAdapter):
    """SRA with local token MLP and pooled global semantic MLP.

    For bottleneck width ``r``, the operated ``r/2`` branch is normalized,
    expanded to ``2r`` (four times the branch width), activated, and projected
    back to ``r/2`` before its internal residual is added.
    """

    def __init__(
        self,
        embed_dim: int,
        reduction: float = 0.25,
        expansion_ratio: float = 4.0,
        gamma_init: float = 1.0,
        inner_scale_init: float = 0.05,
        global_scale_init: float = 0.05,
    ) -> None:
        super().__init__(embed_dim, reduction, gamma_init, inner_scale_init)
        expanded_dim = max(
            self.branch_dim, int(round(self.branch_dim * expansion_ratio))
        )
        self.operator_norm = nn.LayerNorm(self.branch_dim)
        self.operator_up = nn.Linear(self.branch_dim, expanded_dim)
        self.operator_activation = nn.GELU()
        self.operator_down = nn.Linear(expanded_dim, self.branch_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_norm = nn.LayerNorm(self.branch_dim)
        self.global_up = nn.Linear(self.branch_dim, expanded_dim)
        self.global_activation = nn.GELU()
        self.global_down = nn.Linear(expanded_dim, self.branch_dim)
        self.global_scale = nn.Parameter(torch.tensor(float(global_scale_init)))

    def _operator(self, branch: torch.Tensor) -> torch.Tensor:
        local = self.operator_norm(branch)
        local = self.operator_activation(self.operator_up(local))
        local = self.operator_down(local)

        pooled = self.global_pool(branch.permute(0, 3, 1, 2))
        pooled = pooled.permute(0, 2, 3, 1).contiguous()
        pooled = self.global_norm(pooled)
        pooled = self.global_activation(self.global_up(pooled))
        global_response = self.global_down(pooled)
        global_response = global_response.expand(
            -1, branch.shape[1], branch.shape[2], -1
        )
        return local + self.global_scale * global_response


class CrossModalQuery(nn.Module):
    """Memory-bounded bidirectional cross attention between RGB and DSM.

    Args:
        channels: Shared RGB/DSM feature width ``C``.
        num_heads: Attention heads; ``channels`` must be divisible by it.
        mlp_ratio: Expansion ratio of each modality-specific residual FFN.
        dropout: Attention and FFN dropout probability.
        kv_pool_size: Optional maximum key/value height and width. ``None``
            preserves the full feature grid.
        query_chunk_size: Maximum query tokens per attention call. ``None``
            processes all ``H*W`` queries together.

    Input/Output:
        RGB and DSM ``[B,H,W,C]`` -> two fused tensors of the same shape.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        kv_pool_size: Optional[int] = None,
        query_chunk_size: Optional[int] = 4096,
        residual_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        if kv_pool_size is not None and kv_pool_size <= 0:
            raise ValueError("kv_pool_size must be positive or None")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None")

        self.kv_pool_size = kv_pool_size
        self.query_chunk_size = query_chunk_size
        self.rgb_norm = nn.LayerNorm(channels)
        self.dsm_norm = nn.LayerNorm(channels)
        self.rgb_queries_dsm = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.dsm_queries_rgb = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        hidden_channels = int(channels * mlp_ratio)
        self.rgb_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
        )
        self.dsm_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
        )
        self.rgb_gate = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.dsm_gate = nn.Parameter(torch.tensor(float(residual_scale_init)))
        nn.init.zeros_(self.rgb_ffn[-1].weight)
        nn.init.zeros_(self.rgb_ffn[-1].bias)
        nn.init.zeros_(self.dsm_ffn[-1].weight)
        nn.init.zeros_(self.dsm_ffn[-1].bias)

    def _memory_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        """Flatten or pool ``[B,H,W,C]`` into ``[B,N_memory,C]`` tokens."""
        batch_size, height, width, channels = feature.shape
        if self.kv_pool_size is None or (
            height <= self.kv_pool_size and width <= self.kv_pool_size
        ):
            return feature.reshape(batch_size, height * width, channels)

        pooled_height = min(height, self.kv_pool_size)
        pooled_width = min(width, self.kv_pool_size)
        feature = F.adaptive_avg_pool2d(
            feature.permute(0, 3, 1, 2),
            output_size=(pooled_height, pooled_width),
        )
        return feature.permute(0, 2, 3, 1).reshape(
            batch_size, pooled_height * pooled_width, channels
        )

    def _cross_attend(
        self,
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Attend ``[B,Nq,C]`` queries to ``[B,Nm,C]`` memory in chunks."""
        if self.query_chunk_size is None or query.shape[1] <= self.query_chunk_size:
            return attention(query, memory, memory, need_weights=False)[0]

        chunks = []
        for start in range(0, query.shape[1], self.query_chunk_size):
            query_chunk = query[:, start : start + self.query_chunk_size]
            chunks.append(attention(query_chunk, memory, memory, need_weights=False)[0])
        return torch.cat(chunks, dim=1)

    def forward(
        self, rgb: torch.Tensor, dsm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse matching RGB/DSM grids without changing their dimensions."""
        batch_size, height, width, channels = rgb.shape
        rgb_tokens = rgb.reshape(batch_size, height * width, channels)
        dsm_tokens = dsm.reshape(batch_size, height * width, channels)

        rgb_norm = self.rgb_norm(rgb_tokens)
        dsm_norm = self.dsm_norm(dsm_tokens)
        rgb_memory = self.dsm_norm(self._memory_tokens(dsm))
        dsm_memory = self.rgb_norm(self._memory_tokens(rgb))
        rgb_cross = self._cross_attend(self.rgb_queries_dsm, rgb_norm, rgb_memory)
        dsm_cross = self._cross_attend(self.dsm_queries_rgb, dsm_norm, dsm_memory)

        rgb_tokens = rgb_tokens + torch.tanh(self.rgb_gate) * rgb_cross
        dsm_tokens = dsm_tokens + torch.tanh(self.dsm_gate) * dsm_cross
        rgb_tokens = rgb_tokens + self.rgb_ffn(rgb_tokens)
        dsm_tokens = dsm_tokens + self.dsm_ffn(dsm_tokens)
        return (
            rgb_tokens.reshape(batch_size, height, width, channels),
            dsm_tokens.reshape(batch_size, height, width, channels),
        )
