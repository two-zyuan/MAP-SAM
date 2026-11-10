"""Small neural-network layers shared by the local SAM2 components.

Adapted from Meta SAM2. See ``model/SAM2/LICENSE``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop complete residual branches independently for each batch sample.

    Args:
        drop_prob: Probability of dropping a sample's branch.
        scale_by_keep: Divide retained branches by ``1 - drop_prob``.

    Input/Output:
        ``x`` has any shape beginning with ``B``; the shape is unchanged.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-sample branch dropping and return the unchanged shape."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    """Apply a configurable stack of fully connected layers to the last axis.

    Args:
        input_dim: Size of the input's final feature dimension.
        hidden_dim: Feature width of every non-final layer.
        output_dim: Size of the returned final feature dimension.
        num_layers: Total number of linear layers; must be at least one.
        activation: Activation class used between linear layers.
        sigmoid_output: Apply sigmoid to the final output when true.

    Input/Output:
        ``[..., input_dim] -> [..., output_dim]``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: type[nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(
                [input_dim] + hidden,
                hidden + [output_dim],
            )
        )
        self.num_layers = num_layers
        self.activation = activation()
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the final axis from ``input_dim`` to ``output_dim``."""
        for index, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if index < self.num_layers - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class LayerNorm2d(nn.Module):
    """Normalize a BCHW feature map across channels at every pixel.

    Args:
        num_channels: Channel count ``C`` of the input and output.
        eps: Positive numerical-stability constant.

    Input/Output:
        ``[B, C, H, W] -> [B, C, H, W]``.
    """

    def __init__(self, num_channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``[B,C,H,W]`` over ``C`` and preserve all dimensions."""
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
