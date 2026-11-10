
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# See third_party/SegFormer-LICENSE.txt at the repository root.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class DropPath(nn.Module):
    """Drop complete transformer residual branches per sample.

    Args:
        drop_prob: Branch drop probability.

    Input/Output:
        Any batch-first tensor retains its input shape.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Apply stochastic depth without changing dimensions."""
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Initialize ``tensor`` from a normal distribution clipped to ``[a,b]``."""

    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        """Perform the in-place initialization without autograd tracking."""
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a, max=b)

    _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DWConv(nn.Module):
    """Inject local spatial context into flattened MiT tokens.

    Args:
        dim: Token channel width ``C``.

    Input/Output:
        Tokens ``[B,H*W,C]`` plus ``H,W`` -> ``[B,H*W,C]``.
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """Reshape tokens to BCHW, depth-wise convolve, and flatten again."""
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MiT feed-forward network with a depth-wise spatial convolution.

    Args:
        in_features: Input token width.
        hidden_features: Hidden token width; defaults to ``in_features``.
        out_features: Output token width; defaults to ``in_features``.
        act_layer: Activation class.
        drop: Dropout probability after activation and output projection.

    Input/Output:
        ``[B,H*W,in_features] -> [B,H*W,out_features]``.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """Apply linear, spatial depth-wise, activation, and output projections."""
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """MiT self-attention with optional spatial key/value reduction.

    Args:
        dim: Token channel width ``C``.
        num_heads: Attention heads dividing ``C``.
        qkv_bias: Add bias to query/key/value projections.
        qk_scale: Optional scale overriding ``head_dim**-0.5``.
        attn_drop: Dropout probability on attention weights.
        proj_drop: Dropout probability after output projection.
        sr_ratio: Spatial reduction stride for keys and values.

    Input/Output:
        Tokens ``[B,H*W,C]`` plus ``H,W`` -> ``[B,H*W,C]``.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        """Attend full-resolution queries to optionally reduced key/value tokens."""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Pre-normalized MiT attention and spatial-MLP residual block.

    Args:
        dim: Input/output token width.
        num_heads: Attention-head count.
        mlp_ratio: MLP hidden-width multiplier.
        qkv_bias: Add bias to attention projections.
        qk_scale: Optional attention scale.
        drop: Feature dropout probability.
        attn_drop: Attention-weight dropout probability.
        drop_path: Residual stochastic-depth probability.
        act_layer: MLP activation class.
        norm_layer: Token normalization class.
        sr_ratio: Key/value spatial reduction stride.

    Input/Output:
        ``[B,H*W,dim] -> [B,H*W,dim]``.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        """Apply attention and MLP residual updates without shape changes."""
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Convert a BCHW feature map into overlapping flattened patch tokens.

    Args:
        img_size: Nominal input side used as architecture metadata.
        patch_size: Square convolution kernel side.
        stride: Spatial downsampling stride.
        in_chans: Input channel count.
        embed_dim: Returned token width.

    Input/Output:
        ``[B,in_chans,H,W] -> ([B,H'*W',embed_dim], H', W')``.
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Project, flatten, and normalize an input feature grid."""
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """Four-stage SegFormer MiT encoder used inside prompt diffusion.

    Args:
        img_size: Nominal square input side.
        in_chans: Input image/prompt channels.
        embed_dims: Channel width for stages 1-4.
        num_heads: Attention heads for stages 1-4.
        mlp_ratios: MLP expansion ratio for stages 1-4.
        qkv_bias: Add bias to attention projections.
        qk_scale: Optional attention scale.
        drop_rate: Feature dropout probability.
        attn_drop_rate: Attention-weight dropout probability.
        drop_path_rate: Maximum stochastic-depth probability.
        norm_layer: Token normalization class.
        depths: Transformer-block count for stages 1-4.
        sr_ratios: Key/value spatial reduction for stages 1-4.

    Input/Output:
        ``[B,in_chans,H,W]`` -> four BCHW tensors at strides 4/8/16/32 with
        channels from ``embed_dims``. MiT-B0 uses 32/64/160/256 channels.
    """

    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x):
        """Return the four-stage BCHW feature list."""
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        """Alias of ``forward_features`` for ``[B,C,H,W]`` input."""
        return self.forward_features(x)


# MiT family constructors retained from the original prompt-diffusion encoder.
def mit_b0(pretrained=False, **kwargs):
    """Build MiT-B0; ``pretrained`` is retained for API compatibility only."""
    model = MixVisionTransformer(
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


def mit_b1(pretrained=False, **kwargs):
    """Build MiT-B1; ``pretrained`` is retained for API compatibility only."""
    model = MixVisionTransformer(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


def mit_b2(pretrained=False, **kwargs):
    """Build MiT-B2; ``pretrained`` is retained for API compatibility only."""
    model = MixVisionTransformer(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


def mit_b3(pretrained=False, **kwargs):
    """Build MiT-B3; ``pretrained`` is retained for API compatibility only."""
    model = MixVisionTransformer(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


# Standalone feature-shape smoke test.
if __name__ == "__main__":
    print("Testing mit_b0...")
    model = mit_b0()
    x = torch.randn(2, 3, 512, 512)

    outputs = model(x)

    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Stage {i + 1}: {out.shape}")

    # Expected outputs:
    # Stage 1: torch.Size([2, 32, 128, 128])
    # Stage 2: torch.Size([2, 64, 64, 64])
    # Stage 3: torch.Size([2, 160, 32, 32])
    # Stage 4: torch.Size([2, 256, 16, 16])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
