"""Original GeoSAM prompt diffusion model."""

import math
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .segformer import mit_b0
except ImportError:
    from segformer import mit_b0


class VID(nn.Module):
    """Variational information-distillation penalty for predicted noise.

    Args:
        out_channels: Noise/prompt channel count ``K``.
        init_var: Initial learned observation variance.
        eps: Positive numerical-stability floor.

    Input/Output:
        Student and target maps ``[B,K,H,W]`` -> scalar negative log likelihood.
    """

    def __init__(self, out_channels: int, init_var: float, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            np.log(np.exp(init_var - eps) - 1.0) * torch.ones(out_channels)
        )

    def forward(
        self,
        fm_s: torch.Tensor,
        fm_t: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Measure predicted noise against target noise, optionally masked.

        Args:
            fm_s: Predicted noise ``[B,K,H,W]``.
            fm_t: Target noise ``[B,K,H,W]``.
            valid_mask: Optional valid-pixel mask ``[B,1,H,W]``.
        """
        pred_var = torch.log(1.0 + torch.exp(self.alpha)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
            torch.log(pred_var) + (fm_s - fm_t) ** 2 / pred_var
        )
        if valid_mask is not None:
            neg_log_prob = neg_log_prob * valid_mask.to(neg_log_prob.dtype)
            denominator = valid_mask.to(neg_log_prob.dtype).expand_as(neg_log_prob).sum()
            return neg_log_prob.sum() / denominator.clamp_min(1.0)
        return torch.mean(neg_log_prob)


class DecoderBlock(nn.Module):
    """Upsample a diffusion feature map by two and change its channel width.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.

    Input/Output:
        ``[B,in_channels,H,W] -> [B,out_channels,2H,2W]``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ConvTranspose2d(
                in_channels // 4,
                in_channels // 4,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel reduction, transposed convolution, and projection."""
        return self.decoder(x)


class SinusoidalPosEmb(nn.Module):
    """Encode scalar diffusion timesteps as sinusoidal feature vectors.

    Args:
        dim: Returned embedding width, which must be even.

    Input/Output:
        Timesteps ``[B] -> [B,dim]``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return sine/cosine embeddings for a batch of timesteps."""
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """Apply one-condition feature-wise affine modulation after convolution.

    Args:
        in_ch: Input map channels.
        out_ch: Convolved/output channels.
        cond_dim: Conditioning-vector width.

    Input/Output:
        Map ``[B,in_ch,H,W]`` and condition ``[B,cond_dim]`` ->
        ``[B,out_ch,H,W]``.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Convolve and apply learned per-channel scale and bias."""
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return F.relu((1.0 + gamma) * self.conv(x) + beta)


class DualModalFiLMBlock(nn.Module):
    """Fuse independently predicted RGB/DSM FiLM parameters per channel.

    Args:
        in_ch: Input feature-map channels.
        out_ch: Output feature-map channels.
        cond_dim: Width of each RGB/DSM condition vector.
        rgb_init_weight: Initial RGB gate share in ``(0,1)``.

    Input/Output:
        Map ``[B,in_ch,H,W]`` plus RGB/DSM vectors ``[B,cond_dim]`` ->
        ``[B,out_ch,H,W]``.
    """

    def __init__(self, in_ch, out_ch, cond_dim, rgb_init_weight=0.9):
        super().__init__()
        if not 0.0 < rgb_init_weight < 1.0:
            raise ValueError("rgb_init_weight must be between 0 and 1")
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.rgb_film = nn.Linear(cond_dim, out_ch * 2)
        self.dsm_film = nn.Linear(cond_dim, out_ch * 2)
        self.modal_gate = nn.Linear(cond_dim * 2, out_ch)
        self.last_rgb_weight = None

        # Start from the unmodulated convolution and introduce both conditions gradually.
        nn.init.zeros_(self.rgb_film.weight)
        nn.init.zeros_(self.rgb_film.bias)
        nn.init.zeros_(self.dsm_film.weight)
        nn.init.zeros_(self.dsm_film.bias)
        nn.init.zeros_(self.modal_gate.weight)
        gate_bias = np.log(rgb_init_weight / (1.0 - rgb_init_weight))
        nn.init.constant_(self.modal_gate.bias, gate_bias)

    def forward(self, x, rgb_cond, dsm_cond):
        """Gate RGB/DSM affine parameters and modulate the convolved feature."""
        rgb_gamma, rgb_beta = self.rgb_film(rgb_cond).chunk(2, dim=1)
        dsm_gamma, dsm_beta = self.dsm_film(dsm_cond).chunk(2, dim=1)
        rgb_weight = torch.sigmoid(
            self.modal_gate(torch.cat((rgb_cond, dsm_cond), dim=1))
        )
        self.last_rgb_weight = rgb_weight.detach().mean()
        dsm_weight = 1.0 - rgb_weight

        gamma = rgb_weight * rgb_gamma + dsm_weight * dsm_gamma
        beta = rgb_weight * rgb_beta + dsm_weight * dsm_beta
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return F.relu((1.0 + gamma) * self.conv(x) + beta)

    def mean_rgb_weight(self):
        """Return the most recent mean RGB gate or its initialized value."""
        if self.last_rgb_weight is not None:
            return self.last_rgb_weight
        return torch.sigmoid(self.modal_gate.bias).mean()


class SpatialConditionBlock(nn.Module):
    """Inject aligned RGB/DSM FPN maps into one diffusion encoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rgb_init_weight: float = 0.9,
        scale_init: float = 0.05,
        condition_mode: str = "rgb_dsm",
    ) -> None:
        super().__init__()
        if condition_mode not in {"rgb", "rgb_dsm"}:
            raise ValueError("condition_mode must be 'rgb' or 'rgb_dsm'")
        if not 0.0 < rgb_init_weight < 1.0:
            raise ValueError("rgb_init_weight must be between 0 and 1")
        self.rgb_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dsm_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.modal_gate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.last_rgb_weight = None
        self.condition_mode = condition_mode

        nn.init.zeros_(self.modal_gate.weight)
        gate_bias = math.log(rgb_init_weight / (1.0 - rgb_init_weight))
        nn.init.constant_(self.modal_gate.bias, gate_bias)

    def forward(
        self,
        feature: torch.Tensor,
        rgb_spatial: torch.Tensor,
        dsm_spatial: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target_size = feature.shape[-2:]
        rgb_spatial = F.interpolate(
            rgb_spatial, size=target_size, mode="bilinear", align_corners=False
        )
        rgb_update = self.rgb_projection(rgb_spatial)
        if self.condition_mode == "rgb":
            self.last_rgb_weight = rgb_update.new_tensor(1.0)
            return feature + self.residual_scale * rgb_update
        if dsm_spatial is None:
            raise ValueError("DSM spatial condition is required in rgb_dsm mode")
        dsm_spatial = F.interpolate(
            dsm_spatial, size=target_size, mode="bilinear", align_corners=False
        )
        dsm_update = self.dsm_projection(dsm_spatial)
        rgb_weight = torch.sigmoid(
            self.modal_gate(torch.cat((rgb_update, dsm_update), dim=1))
        )
        self.last_rgb_weight = rgb_weight.detach().mean()
        update = rgb_weight * rgb_update + (1.0 - rgb_weight) * dsm_update
        return feature + self.residual_scale * update

    def mean_rgb_weight(self) -> torch.Tensor:
        if self.last_rgb_weight is not None:
            return self.last_rgb_weight
        if self.condition_mode == "rgb":
            return self.modal_gate.bias.new_tensor(1.0)
        return torch.sigmoid(self.modal_gate.bias).mean()


class PromptDiffusionNet(nn.Module):
    """Predict diffusion noise for six-channel semantic mask prompts.

    Args:
        in_channels: Noised prompt channels ``K``; configured as six.
        cond_dim: Global RGB/DSM and timestep embedding width, currently 256.
        condition_mode: ``"rgb"`` or ``"rgb_dsm"`` FiLM conditioning.
        rgb_init_weight: Initial RGB share for dual-modal gates.
        use_spatial_condition: Inject four aligned RGB/DSM FPN pairs.

    Input/Output:
        Noised prompt ``x_t [B,K,Hp,Wp]``, timestep ``[B]``, RGB vector
        ``[B,256]``, and optional DSM vector ``[B,256]`` -> predicted noise
        ``[B,6,Hp,Wp]``. Four spatial pairs condition FB1-FB4 at their current
        resolutions. Internally MiT-B0 produces channels 32/64/160/256 at
        spatial strides 4/8/16/32 before symmetric upsampling.
    """

    def __init__(
        self,
        in_channels=1,
        cond_dim=256,
        condition_mode="rgb",
        rgb_init_weight=0.9,
        use_spatial_condition=True,
        spatial_condition_channels=256,
        spatial_scale_init=0.05,
    ):
        super().__init__()
        if condition_mode not in {"rgb", "rgb_dsm"}:
            raise ValueError("condition_mode must be 'rgb' or 'rgb_dsm'")
        self.condition_mode = condition_mode
        self.use_spatial_condition = bool(use_spatial_condition)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.encoder = mit_b0()
        self.channels = [32, 64, 160, 256]
        if condition_mode == "rgb":
            film_block = lambda in_ch, out_ch: FiLMBlock(in_ch, out_ch, cond_dim)
        else:
            film_block = lambda in_ch, out_ch: DualModalFiLMBlock(
                in_ch, out_ch, cond_dim, rgb_init_weight=rgb_init_weight
            )
        self.FB1 = film_block(in_channels, 3)
        self.FB2 = film_block(self.channels[0], self.channels[0])
        self.FB3 = film_block(self.channels[1], self.channels[1])
        self.FB4 = film_block(self.channels[2], self.channels[2])
        if self.use_spatial_condition:
            self.spatial_condition_blocks = nn.ModuleList(
                SpatialConditionBlock(
                    spatial_condition_channels,
                    out_channels,
                    rgb_init_weight=rgb_init_weight,
                    scale_init=spatial_scale_init,
                    condition_mode=condition_mode,
                )
                for out_channels in (3, self.channels[0], self.channels[1], self.channels[2])
            )

        self.decoder1 = DecoderBlock(self.channels[3], self.channels[2])
        self.decoder2 = DecoderBlock(self.channels[2], self.channels[1])
        self.decoder3 = DecoderBlock(self.channels[1], self.channels[0])
        self.decoder4 = nn.Sequential(
            DecoderBlock(self.channels[0], self.channels[0]),
            DecoderBlock(self.channels[0], 6),
        )
        self.vid_loss = VID(out_channels=in_channels, init_var=5)

    def _inject_spatial_condition(
        self,
        index: int,
        feature: torch.Tensor,
        rgb_spatial: Optional[Sequence[torch.Tensor]],
        dsm_spatial: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        if not self.use_spatial_condition:
            return feature
        if rgb_spatial is None:
            raise ValueError("RGB spatial conditions are required")
        if len(rgb_spatial) != 4:
            raise ValueError("RGB spatial conditions must contain four feature maps")
        if self.condition_mode == "rgb_dsm" and dsm_spatial is None:
            raise ValueError("DSM spatial conditions are required in rgb_dsm mode")
        if dsm_spatial is not None and len(dsm_spatial) != 4:
            raise ValueError("DSM spatial conditions must contain four feature maps")
        return self.spatial_condition_blocks[index](
            feature,
            rgb_spatial[index],
            dsm_spatial[index] if dsm_spatial is not None else None,
        )

    def forward(
        self,
        x_t,
        t,
        rgb_cond_feat,
        dsm_cond_feat=None,
        rgb_spatial=None,
        dsm_spatial=None,
    ):
        """Predict noise with RGB or jointly gated RGB/DSM conditioning."""
        t_emb = self.time_embed(t)
        rgb_cond = rgb_cond_feat + t_emb
        if self.condition_mode == "rgb_dsm":
            if dsm_cond_feat is None:
                raise ValueError("dsm_cond_feat is required in rgb_dsm condition mode")
            dsm_cond = dsm_cond_feat + t_emb
            apply_film = lambda block, feature: block(feature, rgb_cond, dsm_cond)
        else:
            apply_film = lambda block, feature: block(feature, rgb_cond)

        h1 = apply_film(self.FB1, x_t)
        h1 = self._inject_spatial_condition(0, h1, rgb_spatial, dsm_spatial)
        x, H, W = self.encoder.patch_embed1(h1)
        for blk in self.encoder.block1:
            x = blk(x, H, W)
        f1 = self.encoder.norm1(x)
        f1 = f1.reshape(f1.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        h2 = apply_film(self.FB2, f1)
        h2 = self._inject_spatial_condition(1, h2, rgb_spatial, dsm_spatial)
        x, H, W = self.encoder.patch_embed2(h2)
        for blk in self.encoder.block2:
            x = blk(x, H, W)
        f2 = self.encoder.norm2(x)
        f2 = f2.reshape(f2.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        h3 = apply_film(self.FB3, f2)
        h3 = self._inject_spatial_condition(2, h3, rgb_spatial, dsm_spatial)
        x, H, W = self.encoder.patch_embed3(h3)
        for blk in self.encoder.block3:
            x = blk(x, H, W)
        f3 = self.encoder.norm3(x)
        f3 = f3.reshape(f3.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        h4 = apply_film(self.FB4, f3)
        h4 = self._inject_spatial_condition(3, h4, rgb_spatial, dsm_spatial)
        x, H, W = self.encoder.patch_embed4(h4)
        for blk in self.encoder.block4:
            x = blk(x, H, W)
        f4 = self.encoder.norm4(x)
        f4 = f4.reshape(f4.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        d1 = self.decoder1(f4)
        d2 = self.decoder2(d1 + f3)
        d3 = self.decoder3(d2 + f2)
        return self.decoder4(d3 + f1)


def diffusion_training_step(
    prompt_diffusion_model,
    x_0,
    t,
    rgb_cond_feat,
    dsm_cond_feat=None,
    valid_mask=None,
    rgb_spatial=None,
    dsm_spatial=None,
):
    """Run one diffusion training forward and return loss plus ``x0`` estimate.

    This keeps noise sampling, denoising, and reconstruction in one network
    forward. ``x_0`` is ``[B,K,H,W]``; ``valid_mask`` is ``[B,1,H,W]``.
    """
    noise = torch.randn_like(x_0)
    betas = torch.linspace(1e-4, 0.02, 1000, device=x_0.device, dtype=x_0.dtype)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    pred_noise = prompt_diffusion_model(
        x_t,
        t,
        rgb_cond_feat,
        dsm_cond_feat,
        rgb_spatial=rgb_spatial,
        dsm_spatial=dsm_spatial,
    )
    squared_error = (pred_noise - noise) ** 2
    if valid_mask is not None:
        mask = valid_mask.to(squared_error.dtype)
        squared_error = squared_error * mask
        denominator = mask.expand_as(squared_error).sum().clamp_min(1.0)
        mse = squared_error.sum() / denominator
    else:
        mse = squared_error.mean()
    loss = mse + prompt_diffusion_model.vid_loss(
        pred_noise, noise, valid_mask=valid_mask
    )
    pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
    return loss, pred_x0


def diffusion_loss(
    prompt_diffusion_model,
    x_0,
    t,
    rgb_cond_feat,
    dsm_cond_feat=None,
    rgb_spatial=None,
    dsm_spatial=None,
):
    """Backward-compatible scalar diffusion loss wrapper."""
    loss, _ = diffusion_training_step(
        prompt_diffusion_model,
        x_0,
        t,
        rgb_cond_feat,
        dsm_cond_feat,
        rgb_spatial=rgb_spatial,
        dsm_spatial=dsm_spatial,
    )
    return loss


class DDIMSampler:
    """Deterministic/stochastic DDIM sampler over the fixed 1000-step schedule.

    Args:
        model: Noise predictor accepting prompt, timestep, and condition vectors.
        n_steps: Number of reverse-process evaluations in ``[1,1000]``.
        eta: Additional DDIM stochasticity; zero is deterministic given ``x_T``.
    """

    def __init__(self, model, n_steps=50, eta=0.0):
        if not 1 <= n_steps <= 1000:
            raise ValueError("n_steps must be between 1 and 1000")
        self.model = model
        self.n_steps = n_steps
        self.eta = eta
        self.betas = torch.linspace(1e-4, 0.02, 1000)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample(
        self,
        shape,
        rgb_cond_feat,
        dsm_cond_feat=None,
        rgb_spatial=None,
        dsm_spatial=None,
    ):
        """Generate a prompt tensor with ``shape=(B,K,H,W)``."""
        B = shape[0]
        device = rgb_cond_feat.device
        dtype = rgb_cond_feat.dtype
        alpha_bars = self.alpha_bars.to(device=device, dtype=dtype)
        x = torch.randn(shape, device=device, dtype=dtype)
        t_schedule = torch.linspace(
            999, 0, self.n_steps, device=device, dtype=torch.long
        )

        for i in range(self.n_steps):
            t = t_schedule[i].expand(B)
            alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
            with torch.no_grad():
                eps_theta = self.model(
                    x,
                    t,
                    rgb_cond_feat,
                    dsm_cond_feat,
                    rgb_spatial=rgb_spatial,
                    dsm_spatial=dsm_spatial,
                )

            pred_x0 = (
                x - torch.sqrt(1.0 - alpha_bar) * eps_theta
            ) / torch.sqrt(alpha_bar)
            if i == self.n_steps - 1:
                x = pred_x0
            else:
                next_t = t_schedule[i + 1]
                alpha_bar_next = alpha_bars[next_t].view(1, 1, 1, 1)
                sigma = (
                    self.eta
                    * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_next)
                )
                direction = torch.sqrt(
                    (1.0 - alpha_bar_next - sigma.square()).clamp_min(0.0)
                ) * eps_theta
                x = (
                    torch.sqrt(alpha_bar_next) * pred_x0
                    + direction
                    + sigma * torch.randn_like(x)
                )
        return x


class PromptGenerator:
    """Small facade exposing prompt sampling and train-time diffusion loss.

    Args:
        diffusion_model: ``PromptDiffusionNet`` noise predictor.
        ddim_steps: Reverse-process evaluations used for prompt generation.
    """

    def __init__(self, diffusion_model: nn.Module, ddim_steps: int = 5) -> None:
        self.diffusion = diffusion_model
        self.ddim = DDIMSampler(diffusion_model, n_steps=ddim_steps)

    def generate_prompt(
        self,
        rgb_cond_feat,
        dsm_cond_feat=None,
        shape=(1, 1, 256, 256),
        rgb_spatial=None,
        dsm_spatial=None,
    ):
        """Sample prompts ``[B,K,H,W]`` from global condition vectors."""
        return self.ddim.sample(
            shape=shape,
            rgb_cond_feat=rgb_cond_feat,
            dsm_cond_feat=dsm_cond_feat,
            rgb_spatial=rgb_spatial,
            dsm_spatial=dsm_spatial,
        )

    def training_step(
        self,
        x_0,
        t,
        rgb_cond_feat,
        dsm_cond_feat=None,
        valid_mask=None,
        rgb_spatial=None,
        dsm_spatial=None,
    ):
        """Return one-forward loss and reconstructed prompt ``[B,K,H,W]``."""
        return diffusion_training_step(
            self.diffusion,
            x_0,
            t,
            rgb_cond_feat,
            dsm_cond_feat,
            valid_mask=valid_mask,
            rgb_spatial=rgb_spatial,
            dsm_spatial=dsm_spatial,
        )

    def training_loss(
        self,
        x_0,
        t,
        rgb_cond_feat,
        dsm_cond_feat=None,
        rgb_spatial=None,
        dsm_spatial=None,
    ):
        """Compute scalar diffusion loss for targets ``[B,K,H,W]``."""
        loss, _ = self.training_step(
            x_0,
            t,
            rgb_cond_feat,
            dsm_cond_feat,
            rgb_spatial=rgb_spatial,
            dsm_spatial=dsm_spatial,
        )
        return loss
