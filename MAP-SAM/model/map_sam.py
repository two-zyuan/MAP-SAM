"""Top-level MAP-SAM network assembly and forward data flow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoding.sam_heads import MaskDecoder, PromptEncoder, TwoWayTransformer
from .decoding.task_decoder import FeatureDecoder
from .feature_extraction import DEFAULT_CHECKPOINT, DEFAULT_MODEL_CONFIG, SAM2Encoder
from .losses import RGBDSMContrastiveLoss
from .prompt_generation import PromptDiffusionNet, PromptGenerator
from .prompt_generation.prompt_pipeline import (
    decode_sam_features,
    generate_prompt,
    load_checkpoint_modules,
    prompt_segmentation_loss,
)


class SamFuNet(nn.Module):
    """Multimodal semantic segmenter built around pretrained SAM2 components.

    Args:
        img_size: Required square RGB/DSM input side. Must be divisible by 128.
        num_classes: Semantic classes ``K``. The original diffusion decoder
            currently requires exactly six.
        ignore_index: Label value ignored by semantic and auxiliary losses.
        sam_model_config: SAM2 image-encoder variant, currently ``hiera_small``.
        sam_checkpoint: Optional SAM2 weight path; ``None`` uses the bundled file.
        diffusion_steps: DDIM sampling iterations used to generate mask prompts.
        diffusion_condition_mode: Condition prompts using ``rgb`` or ``rgb_dsm``.
        diffusion_rgb_init_weight: Initial RGB share of dual-modal FiLM gates.
        diffusion_use_spatial_condition: Inject aligned FPN maps into diffusion.
        diffusion_spatial_scale_init: Initial residual scale of spatial injection.
        contrastive_temperature: RGB/DSM contrastive softmax temperature.
        position_contrastive_weight: Weight of same-position positives.
        class_contrastive_weight: Weight of same-class positives.
        contrastive_max_samples: Maximum spatial tokens in contrastive loss.
        cross_stages: One-based Hiera stages running RGB/DSM cross attention.
        adapter_reduction: SCA/SRA bottleneck ratio ``r/d``.
        sca_kernel_sizes: Parallel depth-wise convolution kernels used by SCA.
        sra_expansion_ratio: SRA expansion relative to the ``r/2`` branch.
        sra_global_scale_init: Initial scale of the pooled global SRA branch.
        adapter_gamma_init: Initial external adapter scale vector value.
        adapter_inner_scale_init: Initial internal operator scale vector value.
        norm: Normalization class used by the semantic task decoder.
        act: Activation class used by the semantic task decoder.

    Input:
        RGB ``[B,3,H,W]``, DSM ``[B,1,H,W]``, and optional labels
        ``[B,1,H,W]`` or ``[B,H,W]``. The configured ``H=W=img_size``.

    Output:
        Semantic logits ``[B,K,H,W]``; full-resolution diffusion prompt
        ``[B,K,H,W]``; and, during training, scalar diffusion, contrastive,
        and prompt-segmentation losses. Evaluation returns ``None`` losses.
    """

    def __init__(
        self,
        img_size: int = 1024,
        num_classes: int = 6,
        ignore_index: int = 255,
        sam_model_config: str = DEFAULT_MODEL_CONFIG,
        sam_checkpoint: Optional[str] = None,
        diffusion_steps: int = 10,
        diffusion_condition_mode: str = "rgb_dsm",
        diffusion_rgb_init_weight: float = 0.9,
        diffusion_use_spatial_condition: bool = True,
        diffusion_spatial_scale_init: float = 0.05,
        contrastive_temperature: float = 0.07,
        position_contrastive_weight: float = 1.0,
        class_contrastive_weight: float = 1.0,
        contrastive_max_samples: int = 1024,
        cross_stages: Tuple[int, ...] = (3, 4),
        cross_modal_scale_init: float = 0.05,
        adapter_reduction: float = 0.25,
        sca_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        sra_expansion_ratio: float = 4.0,
        sra_global_scale_init: float = 0.05,
        adapter_gamma_init: float = 1.0,
        adapter_inner_scale_init: float = 0.05,
        decoder_dsm_scale_init: float = 0.05,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if img_size % 128 != 0:
            raise ValueError(
                "img_size must be divisible by 128 so the stride-4 prompt grid "
                "is compatible with all diffusion encoder/decoder stages"
            )
        if num_classes != 6:
            raise ValueError("The original prompt diffusion model requires num_classes=6")
        self.img_size = img_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.diffusion_condition_mode = diffusion_condition_mode
        self.diffusion_use_spatial_condition = diffusion_use_spatial_condition
        self.image_embedding_size = (img_size // 16, img_size // 16)
        self.checkpoint = Path(sam_checkpoint) if sam_checkpoint else Path(DEFAULT_CHECKPOINT)

        self.image_encoder = SAM2Encoder(
            model_config=sam_model_config,
            checkpoint=str(self.checkpoint),
            cross_stages=cross_stages,
            cross_modal_scale_init=cross_modal_scale_init,
            adapter_reduction=adapter_reduction,
            sca_kernel_sizes=sca_kernel_sizes,
            sra_expansion_ratio=sra_expansion_ratio,
            sra_global_scale_init=sra_global_scale_init,
            adapter_gamma_init=adapter_gamma_init,
            adapter_inner_scale_init=adapter_inner_scale_init,
        )
        # These are the unmodified native SAM2 prompt modules. Unlike the SAM2
        # image backbone in SAM2Encoder, both modules remain fully trainable.
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=self.image_embedding_size,
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
        )
        self._load_native_sam2_heads(self.checkpoint)
        self.prompt_encoder.requires_grad_(True)
        self.mask_decoder.requires_grad_(True)

        self.prompt_diffusion = PromptDiffusionNet(
            in_channels=num_classes,
            cond_dim=256,
            condition_mode=diffusion_condition_mode,
            rgb_init_weight=diffusion_rgb_init_weight,
            use_spatial_condition=diffusion_use_spatial_condition,
            spatial_condition_channels=self.image_encoder.output_channels,
            spatial_scale_init=diffusion_spatial_scale_init,
        )
        self.prompt_generator = PromptGenerator(
            self.prompt_diffusion,
            ddim_steps=diffusion_steps,
        )
        self.prompt_to_sam_mask = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.contrastive_loss = RGBDSMContrastiveLoss(
            channels=self.image_encoder.output_channels,
            temperature=contrastive_temperature,
            position_weight=position_contrastive_weight,
            class_weight=class_contrastive_weight,
            ignore_index=ignore_index,
            num_classes=num_classes,
            max_samples=contrastive_max_samples,
        )
        self.feature_decoder = FeatureDecoder(
            num_classes=num_classes,
            dsm_residual_scale_init=decoder_dsm_scale_init,
            norm=norm,
            act=act,
        )

    def _load_native_sam2_heads(self, checkpoint: Path) -> None:
        """Load SAM2 PromptEncoder and MaskDecoder weights by checkpoint prefix."""
        load_checkpoint_modules(
            checkpoint,
            {
                "sam_prompt_encoder.": self.prompt_encoder,
                "sam_mask_decoder.": self.mask_decoder,
            },
        )

    def _generate_prompt(
        self,
        rgb_condition: torch.Tensor,
        dsm_condition: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        is_train: bool,
        rgb_spatial,
        dsm_spatial,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Generate downstream and supervised prompt logits at stride four."""
        return generate_prompt(
            prompt_generator=self.prompt_generator,
            rgb_condition=rgb_condition,
            dsm_condition=dsm_condition,
            labels=labels,
            prompt_size=self.prompt_encoder.mask_input_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            condition_mode=self.diffusion_condition_mode,
            is_train=is_train,
            rgb_spatial=rgb_spatial,
            dsm_spatial=dsm_spatial,
        )

    def _prompt_segmentation_loss(
        self, prompt_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Supervise prompt logits ``[B,K,h,w]`` with semantic labels."""
        return prompt_segmentation_loss(
            prompt_logits,
            labels,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def _decode_native_sam2(
        self,
        image_embedding: torch.Tensor,
        mask_prompt: torch.Tensor,
        rgb_pyramid,
    ) -> torch.Tensor:
        """Run SAM2 heads and return only ``[B,32,H/4,W/4]`` decoder features."""
        return decode_sam_features(
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
            image_embedding=image_embedding,
            mask_prompt=mask_prompt,
            rgb_pyramid=rgb_pyramid,
            image_embedding_size=self.image_embedding_size,
        )

    def forward(
        self,
        img: torch.Tensor,
        dsm: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_train: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Execute RGB/DSM extraction, prompt decoding, and semantic decoding."""
        if img.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size} input, got {img.shape[-2:]}"
            )
        encoded = self.image_encoder(img, dsm)
        rgb_pyramid = encoded["rgb_fpn"]
        if len(rgb_pyramid) != 4:
            raise ValueError(f"Expected four RGB FPN levels, got {len(rgb_pyramid)}")

        dsm_pyramid = encoded["dsm_fpn"]
        # Feed all four FPN resolutions to the diffusion encoder so its
        # conditions retain stride-8 boundary detail as well as context.
        spatial_indices = (0, 1, 2, 3)
        rgb_spatial = tuple(rgb_pyramid[index] for index in spatial_indices)
        dsm_spatial = (
            tuple(dsm_pyramid[index] for index in spatial_indices)
            if self.diffusion_condition_mode == "rgb_dsm"
            else None
        )
        prompt_mask, supervised_prompt, diffusion_loss = self._generate_prompt(
            encoded["rgb_last"],
            encoded["dsm_last"],
            labels,
            is_train=is_train,
            rgb_spatial=rgb_spatial,
            dsm_spatial=dsm_spatial,
        )
        sam_mask_prompt = self.prompt_to_sam_mask(prompt_mask)
        sam_decoder_feature = self._decode_native_sam2(
            # SAM2 discards the /32 FPN level with scalp=1 and uses /16 here.
            image_embedding=rgb_pyramid[-2],
            mask_prompt=sam_mask_prompt,
            rgb_pyramid=rgb_pyramid,
        )

        mask = self.feature_decoder(
            rgb_pyramid, dsm_pyramid, sam_decoder_feature
        )
        if mask.shape[-2:] != img.shape[-2:]:
            mask = F.interpolate(
                mask, size=img.shape[-2:], mode="bilinear", align_corners=False
            )
        output_prompt = F.interpolate(
            prompt_mask,
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        contrastive_loss = None
        prompt_segmentation_loss = None
        if is_train:
            contrastive_loss = self.contrastive_loss(
                encoded["rgb_last"], encoded["dsm_last"], labels
            )
            prompt_segmentation_loss = self._prompt_segmentation_loss(
                supervised_prompt, labels
            )
        return (
            mask,
            output_prompt,
            diffusion_loss,
            contrastive_loss,
            prompt_segmentation_loss,
        )

Geo_SAM = SamFuNet
TriASAM = SamFuNet


