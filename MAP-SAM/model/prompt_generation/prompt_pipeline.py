"""Prompt generation, supervision, checkpoint loading, and SAM head execution."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoding.sam_heads import MaskDecoder, PromptEncoder
from .diffusion import PromptGenerator


def load_checkpoint_modules(
    checkpoint: Path,
    modules: Mapping[str, nn.Module],
) -> None:
    """Load named module subsets from a SAM2 checkpoint.

    Args:
        checkpoint: Checkpoint containing either a top-level state dictionary or
            a ``model`` state dictionary.
        modules: Mapping from checkpoint prefix to destination module. Prefixes
            are stripped before non-strict module loading.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint was not found at '{checkpoint}'")
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data.get("model", checkpoint_data)
    for prefix, module in modules.items():
        module_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not module_state:
            raise RuntimeError(
                f"Checkpoint '{checkpoint}' contains no parameters under '{prefix}'"
            )
        incompatible = module.load_state_dict(module_state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                f"Missing {prefix} checkpoint keys: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected {prefix} checkpoint keys: {incompatible.unexpected_keys}"
            )


def resize_feature(feature: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize ``[B,C,H,W]`` to ``[B,C,size_h,size_w]`` if needed."""
    if feature.shape[-2:] == size:
        return feature
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


def prepare_diffusion_target(
    labels: torch.Tensor,
    prompt_size: Tuple[int, int],
    num_classes: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create one-hot diffusion targets from dense class labels.

    Args:
        labels: ``[B,H,W]`` or ``[B,1,H,W]`` integer labels.
        prompt_size: Target spatial size ``(Hp,Wp)``.
        num_classes: One-hot channel count ``K``.
        ignore_index: Label excluded from supervision.

    Returns:
        Target ``[B,K,Hp,Wp]`` and valid mask ``[B,1,Hp,Wp]``.
    """
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B, H, W], got {labels.shape}")
    if labels.shape[-2:] != prompt_size:
        labels = F.interpolate(
            labels.unsqueeze(1).float(), size=prompt_size, mode="nearest"
        ).squeeze(1).long()
    valid_mask = (
        labels.ne(ignore_index)
        & labels.ge(0)
        & labels.lt(num_classes)
    ).unsqueeze(1)
    safe_labels = labels.masked_fill(~valid_mask.squeeze(1), 0)
    one_hot = F.one_hot(safe_labels, num_classes).permute(0, 3, 1, 2)
    return one_hot.float() * valid_mask, valid_mask


def generate_prompt(
    prompt_generator: PromptGenerator,
    rgb_condition: torch.Tensor,
    dsm_condition: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    prompt_size: Tuple[int, int],
    num_classes: int,
    ignore_index: int,
    condition_mode: str,
    is_train: bool,
    rgb_spatial: Optional[Sequence[torch.Tensor]] = None,
    dsm_spatial: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Generate downstream prompts plus optional supervised reconstruction.

    The downstream prompt is always sampled from noise with DDIM, matching
    inference. During training, a separate label-noising path returns a
    supervised ``x0`` reconstruction and diffusion loss; it is never passed to
    SAM or the task decoder.
    """
    batch_size = rgb_condition.shape[0]
    rgb_cond_feat = F.adaptive_avg_pool2d(rgb_condition, 1).flatten(1)
    dsm_cond_feat = None
    if condition_mode == "rgb_dsm":
        if dsm_condition is None:
            raise ValueError("DSM condition is required in rgb_dsm diffusion mode")
        dsm_cond_feat = F.adaptive_avg_pool2d(dsm_condition, 1).flatten(1)

    if not is_train:
        return (
            prompt_generator.generate_prompt(
                rgb_cond_feat,
                dsm_cond_feat,
                shape=(batch_size, num_classes, *prompt_size),
                rgb_spatial=rgb_spatial,
                dsm_spatial=dsm_spatial,
            ),
            None,
            None,
        )

    if labels is None:
        raise ValueError("labels are required when is_train=True")
    diffusion_model = prompt_generator.diffusion
    was_training = diffusion_model.training
    diffusion_model.eval()
    try:
        with torch.no_grad():
            downstream_prompt = prompt_generator.generate_prompt(
                rgb_cond_feat.detach(),
                dsm_cond_feat.detach() if dsm_cond_feat is not None else None,
                shape=(batch_size, num_classes, *prompt_size),
                rgb_spatial=(
                    tuple(feature.detach() for feature in rgb_spatial)
                    if rgb_spatial is not None
                    else None
                ),
                dsm_spatial=(
                    tuple(feature.detach() for feature in dsm_spatial)
                    if dsm_spatial is not None
                    else None
                ),
            )
    finally:
        diffusion_model.train(was_training)
    target, valid_mask = prepare_diffusion_target(
        labels, prompt_size, num_classes, ignore_index
    )
    timesteps = torch.randint(0, 1000, (batch_size,), device=rgb_condition.device)
    diffusion_loss, supervised_prompt = prompt_generator.training_step(
        target,
        timesteps,
        rgb_cond_feat,
        dsm_cond_feat,
        valid_mask=valid_mask,
        rgb_spatial=rgb_spatial,
        dsm_spatial=dsm_spatial,
    )
    return downstream_prompt, supervised_prompt, diffusion_loss


def prompt_segmentation_loss(
    prompt_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Compute cross-entropy plus Dice on prompt logits ``[B,K,h,w]``."""
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B, H, W], got {labels.shape}")
    if labels.shape[-2:] != prompt_logits.shape[-2:]:
        labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=prompt_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()

    valid = (
        labels.ne(ignore_index)
        & labels.ge(0)
        & labels.lt(num_classes)
    )
    if not valid.any():
        return prompt_logits.sum() * 0.0
    supervised_labels = labels.masked_fill(~valid, ignore_index)
    cross_entropy = F.cross_entropy(
        prompt_logits, supervised_labels, ignore_index=ignore_index
    )

    safe_labels = labels.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_labels, num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(prompt_logits.dtype) * valid.unsqueeze(1)
    probabilities = F.softmax(prompt_logits, dim=1) * valid.unsqueeze(1)
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    cardinality = (probabilities + one_hot).sum(dims)
    present_classes = cardinality > 0
    dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
    return cross_entropy + 1.0 - dice[present_classes].mean()


def decode_sam_features(
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    image_embedding: torch.Tensor,
    mask_prompt: torch.Tensor,
    rgb_pyramid: Sequence[torch.Tensor],
    image_embedding_size: Tuple[int, int],
) -> torch.Tensor:
    """Execute native SAM2 heads and return the useful upscaled spatial feature.

    Args:
        prompt_encoder: Native SAM2 visual prompt encoder.
        mask_decoder: Native SAM2 mask decoder with high-resolution fusion.
        image_embedding: Stride-16 RGB feature ``[B,256,He,We]``.
        mask_prompt: Diffusion-derived mask ``[B,1,Hm,Wm]``.
        rgb_pyramid: Four 256-channel features at strides 4/8/16/32.
        image_embedding_size: ``(He,We)``; ``(64,64)`` for 1024 input.

    Returns:
        Decoder feature ``[B,32,4He,4We]``. Mask, IoU, and object-score heads
        are not evaluated because downstream semantic decoding does not use them.
    """
    mask_prompt = resize_feature(mask_prompt, prompt_encoder.mask_input_size)
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=None, boxes=None, masks=mask_prompt
    )
    high_res_features = [
        mask_decoder.conv_s0(
            resize_feature(
                rgb_pyramid[0],
                (image_embedding_size[0] * 4, image_embedding_size[1] * 4),
            )
        ),
        mask_decoder.conv_s1(
            resize_feature(
                rgb_pyramid[1],
                (image_embedding_size[0] * 2, image_embedding_size[1] * 2),
            )
        ),
    ]

    return mask_decoder.forward_features(
        image_embeddings=image_embedding,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        repeat_image=False,
        high_res_features=high_res_features,
    )
