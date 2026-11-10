"""Isolated MulMoSAM ablation architectures built from audited project APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CONFIG
from model.decoding.sam_heads import MaskDecoder, PromptEncoder, TwoWayTransformer
from model.decoding.task_decoder import ClassDecoder, DecoderBlock, FeatureDecoder
from model.feature_extraction.multimodal_blocks import SemanticResponseAdapter, SpatialContextAdapter
from model.feature_extraction.multimodal_sam2_encoder import DEFAULT_CHECKPOINT, SAM2Encoder
from model.losses import RGBDSMContrastiveLoss
from model.prompt_generation import PromptDiffusionNet, PromptGenerator
from model.prompt_generation.prompt_pipeline import decode_sam_features, generate_prompt, load_checkpoint_modules, prompt_segmentation_loss

from experiments.scripts.common import ExperimentConfig, get_config


class PlainBottleneckAdapter(nn.Module):
    """Ordinary identity-initialized bottleneck adapter for BHWC Hiera tokens."""

    def __init__(self, embed_dim: int, reduction: float, gamma_init: float = 1.0) -> None:
        super().__init__()
        width = max(16, int(embed_dim * reduction))
        self.embed_dim = embed_dim
        self.down = nn.Linear(embed_dim, width)
        self.activation = nn.GELU()
        self.up = nn.Linear(width, embed_dim)
        self.gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.gamma * self.up(self.activation(self.down(feature)))


class OneWayCrossModal(nn.Module):
    """Update one modality from the other with the audited BCF attention rule."""

    def __init__(self, channels: int, heads: int, pool_size: int, scale: float) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.gate = nn.Parameter(torch.tensor(float(scale)))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _memory(self, feature: torch.Tensor) -> torch.Tensor:
        b, h, w, c = feature.shape
        if h > self.pool_size or w > self.pool_size:
            feature = F.adaptive_avg_pool2d(
                feature.permute(0, 3, 1, 2), (min(h, self.pool_size), min(w, self.pool_size))
            ).permute(0, 2, 3, 1)
        return feature.reshape(b, -1, c)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, h, w, c = query.shape
        tokens = query.reshape(b, h * w, c)
        memory_tokens = self.memory_norm(self._memory(memory))
        attended = self.attention(
            self.query_norm(tokens), memory_tokens, memory_tokens, need_weights=False,
        )[0]
        tokens = tokens + torch.tanh(self.gate) * attended
        tokens = tokens + self.ffn(tokens)
        return tokens.reshape(b, h, w, c)


class DirectionalBCF(nn.Module):
    """Instantiate only the requested BCF directions."""

    def __init__(self, channels: int, heads: int, pool_size: int, direction: str, scale: float) -> None:
        super().__init__()
        self.direction = direction
        if direction in {"auxiliary_to_primary", "bidirectional"}:
            self.auxiliary_to_primary = OneWayCrossModal(channels, heads, pool_size, scale)
        if direction in {"primary_to_auxiliary", "bidirectional"}:
            self.primary_to_auxiliary = OneWayCrossModal(channels, heads, pool_size, scale)

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        old_primary, old_auxiliary = primary, auxiliary
        if hasattr(self, "auxiliary_to_primary"):
            primary = self.auxiliary_to_primary(old_primary, old_auxiliary)
        if hasattr(self, "primary_to_auxiliary"):
            auxiliary = self.primary_to_auxiliary(old_auxiliary, old_primary)
        return primary, auxiliary


class AblationSAM2Encoder(SAM2Encoder):
    """Shared frozen SAM2 encoder with configurable adapters, modalities, and BCF."""

    def __init__(self, experiment: ExperimentConfig, checkpoint: str, **kwargs) -> None:
        super().__init__(checkpoint=checkpoint, cross_stages=(), **kwargs)
        self.experiment = experiment
        channels = [block.dim_out for block in self.trunk.blocks]
        reduction = kwargs.get("adapter_reduction", MODEL_CONFIG.adapter_reduction)
        gamma = kwargs.get("adapter_gamma_init", MODEL_CONFIG.adapter_gamma_init)
        inner = kwargs.get("adapter_inner_scale_init", MODEL_CONFIG.adapter_inner_scale_init)
        kernels = kwargs.get("sca_kernel_sizes", MODEL_CONFIG.sca_kernel_sizes)
        expansion = kwargs.get("sra_expansion_ratio", MODEL_CONFIG.sra_expansion_ratio)
        global_scale = kwargs.get("sra_global_scale_init", MODEL_CONFIG.sra_global_scale_init)

        def attention_adapter(c: int) -> nn.Module:
            if experiment.sca:
                return SpatialContextAdapter(c, reduction, kernels, gamma, inner)
            return PlainBottleneckAdapter(c, reduction, gamma)

        def mlp_adapter(c: int) -> nn.Module:
            if experiment.sra:
                return SemanticResponseAdapter(c, reduction, expansion, gamma, inner, global_scale)
            return PlainBottleneckAdapter(c, reduction, gamma)

        self.rgb_sca_adapters = nn.ModuleList(attention_adapter(c) for c in channels)
        self.rgb_sra_adapters = nn.ModuleList(mlp_adapter(c) for c in channels)
        self.dsm_sca_adapters = nn.ModuleList(attention_adapter(c) for c in channels)
        self.dsm_sra_adapters = nn.ModuleList(mlp_adapter(c) for c in channels)
        if experiment.modalities == "primary":
            self.dsm_sca_adapters = nn.ModuleList()
            self.dsm_sra_adapters = nn.ModuleList()
            del self.dsm_to_in
            del self.dsm_to_in_2
        elif experiment.modalities == "auxiliary":
            self.rgb_sca_adapters = nn.ModuleList()
            self.rgb_sra_adapters = nn.ModuleList()
        stage_channels = [channels[index] for index in self.stage_end_indices]
        heads = (1, 2, 4, 8)
        pools = (8, 8, 16, 32)
        if experiment.cross_direction == "none":
            self.cross_stages = frozenset()
            self.cross_query_list = nn.ModuleList()
        else:
            self.cross_stages = frozenset((1, 2, 3, 4))
            self.cross_query_list = nn.ModuleList(
                DirectionalBCF(c, h, p, experiment.cross_direction, MODEL_CONFIG.cross_modal_scale_init)
                for c, h, p in zip(stage_channels, heads, pools)
            )

    def _run_stream(self, image: torch.Tensor, sca: nn.ModuleList, sra: nn.ModuleList) -> List[torch.Tensor]:
        x = self._patch_embed(image)
        stages = []
        for index, block in enumerate(self.trunk.blocks):
            x = block(x, sca_adapter=sca[index], sra_adapter=sra[index])
            if index in self.stage_ends:
                stages.append(x.permute(0, 3, 1, 2).contiguous())
        return self.neck(stages)[0]

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> Dict[str, object]:
        mode = self.experiment.modalities
        if mode == "primary":
            rgb_fpn = self._run_stream(rgb, self.rgb_sca_adapters, self.rgb_sra_adapters)
            return {"rgb_fpn": rgb_fpn, "dsm_fpn": None, "rgb_last": rgb_fpn[-1], "dsm_last": None}
        prepared_dsm = self._prepare_dsm(dsm)
        if mode == "auxiliary":
            dsm_fpn = self._run_stream(prepared_dsm, self.dsm_sca_adapters, self.dsm_sra_adapters)
            return {"rgb_fpn": None, "dsm_fpn": dsm_fpn, "rgb_last": None, "dsm_last": dsm_fpn[-1]}

        rgb_x = self._patch_embed(rgb)
        dsm_x = self._patch_embed(prepared_dsm)
        rgb_stages: List[torch.Tensor] = []
        dsm_stages: List[torch.Tensor] = []
        stage = 0
        for index, block in enumerate(self.trunk.blocks):
            rgb_x = block(rgb_x, sca_adapter=self.rgb_sca_adapters[index], sra_adapter=self.rgb_sra_adapters[index])
            dsm_x = block(dsm_x, sca_adapter=self.dsm_sca_adapters[index], sra_adapter=self.dsm_sra_adapters[index])
            if index in self.stage_ends:
                if stage + 1 in self.cross_stages:
                    rgb_x, dsm_x = self.cross_query_list[stage](rgb_x, dsm_x)
                rgb_stages.append(rgb_x.permute(0, 3, 1, 2).contiguous())
                dsm_stages.append(dsm_x.permute(0, 3, 1, 2).contiguous())
                stage += 1
        rgb_fpn = self.neck(rgb_stages)[0]
        dsm_fpn = self.neck(dsm_stages)[0]
        return {"rgb_fpn": rgb_fpn, "dsm_fpn": dsm_fpn, "rgb_last": rgb_fpn[-1], "dsm_last": dsm_fpn[-1]}


class DeterministicPrompt(nn.Module):
    def __init__(self, channels: int, num_classes: int, dual: bool) -> None:
        super().__init__()
        inputs = channels * (2 if dual else 1)
        self.head = nn.Sequential(nn.Conv2d(inputs, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, num_classes, 1))

    def forward(self, features: Sequence[torch.Tensor], size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.head(torch.cat(tuple(features), dim=1)), size=size, mode="bilinear", align_corners=False)


class SimpleLateFusionHead(nn.Module):
    def __init__(self, num_classes: int, dual: bool) -> None:
        super().__init__()
        inputs = 256 * (2 if dual else 1) + 32
        self.head = nn.Sequential(
            nn.Conv2d(inputs, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, pyramids: Sequence[Sequence[torch.Tensor]], sam: torch.Tensor) -> torch.Tensor:
        features = [pyramid[0] for pyramid in pyramids]
        return self.head(torch.cat((*features, sam), dim=1))


class SingleStreamFRN(nn.Module):
    """FRN top-down refinement without inapplicable cross-modal residual fusion."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) for _ in range(4))
        self.refine = nn.ModuleList(nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) for _ in range(4))
        self.output_decoder = nn.Sequential(DecoderBlock(96, 32), DecoderBlock(32, 32), nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.class_decoder = ClassDecoder(32, num_classes, 32)

    def forward(self, features: Sequence[torch.Tensor], sam: torch.Tensor) -> torch.Tensor:
        pyramid = [module(feature) for module, feature in zip(self.lateral, features)]
        top = self.refine[3](pyramid[3])
        for index in (2, 1, 0):
            top = F.interpolate(top, size=pyramid[index].shape[-2:], mode="bilinear", align_corners=False)
            top = self.refine[index](pyramid[index] + top)
        return self.class_decoder(self.output_decoder(torch.cat((top, sam), dim=1)))


class AblationMulMoSAM(nn.Module):
    """Configurable model preserving the original five-output forward interface."""

    def __init__(self, experiment: ExperimentConfig | str, img_size: int = 1024, sam_checkpoint: Path | None = None) -> None:
        super().__init__()
        if isinstance(experiment, str):
            experiment = get_config(experiment)
        self.experiment = experiment
        self.img_size = img_size
        self.num_classes = MODEL_CONFIG.num_classes
        self.ignore_index = MODEL_CONFIG.ignore_index
        self.image_embedding_size = (img_size // 16, img_size // 16)
        checkpoint = str(sam_checkpoint if sam_checkpoint is not None else DEFAULT_CHECKPOINT)
        encoder_kwargs = dict(
            adapter_reduction=MODEL_CONFIG.adapter_reduction,
            sca_kernel_sizes=MODEL_CONFIG.sca_kernel_sizes,
            sra_expansion_ratio=MODEL_CONFIG.sra_expansion_ratio,
            sra_global_scale_init=MODEL_CONFIG.sra_global_scale_init,
            adapter_gamma_init=MODEL_CONFIG.adapter_gamma_init,
            adapter_inner_scale_init=MODEL_CONFIG.adapter_inner_scale_init,
        )
        self.image_encoder = AblationSAM2Encoder(experiment, checkpoint, **encoder_kwargs)
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=self.image_embedding_size,
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True, iou_prediction_use_sigmoid=True,
            pred_obj_scores=True, pred_obj_scores_mlp=True, use_multimask_token_for_obj_ptr=True,
        )
        load_checkpoint_modules(Path(checkpoint), {"sam_prompt_encoder.": self.prompt_encoder, "sam_mask_decoder.": self.mask_decoder})
        dual = experiment.modalities == "both"
        self.diffusion_condition_mode = "rgb_dsm" if dual else "rgb"
        if experiment.prompt == "diffusion":
            self.prompt_diffusion = PromptDiffusionNet(
                in_channels=self.num_classes, cond_dim=256, condition_mode=self.diffusion_condition_mode,
                rgb_init_weight=MODEL_CONFIG.diffusion_rgb_init_weight,
                use_spatial_condition=MODEL_CONFIG.diffusion_use_spatial_condition,
                spatial_condition_channels=256, spatial_scale_init=MODEL_CONFIG.diffusion_spatial_scale_init,
            )
            self.prompt_generator = PromptGenerator(self.prompt_diffusion, ddim_steps=MODEL_CONFIG.diffusion_steps)
        else:
            self.deterministic_prompt = DeterministicPrompt(256, self.num_classes, dual)
        self.prompt_to_sam_mask = nn.Conv2d(self.num_classes, 1, 1)
        weights = {"position": (1.0, 0.0), "class": (0.0, 1.0), "joint": (1.0, 1.0)}
        if experiment.contrastive != "none":
            position, semantic = weights[experiment.contrastive]
            self.contrastive_loss = RGBDSMContrastiveLoss(
                256, temperature=MODEL_CONFIG.contrastive_temperature,
                position_weight=position, class_weight=semantic,
                ignore_index=self.ignore_index, num_classes=self.num_classes,
                max_samples=MODEL_CONFIG.contrastive_max_samples,
            )
        if experiment.decoder == "simple":
            self.feature_decoder = SimpleLateFusionHead(self.num_classes, dual)
        elif dual:
            self.feature_decoder = FeatureDecoder(self.num_classes, MODEL_CONFIG.decoder_dsm_scale_init)
        else:
            self.feature_decoder = SingleStreamFRN(self.num_classes)

    def _active(self, encoded: Dict[str, object]) -> Tuple[List[Sequence[torch.Tensor]], Sequence[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        rgb, dsm = encoded["rgb_fpn"], encoded["dsm_fpn"]
        if self.experiment.modalities == "primary":
            return [rgb], rgb, encoded["rgb_last"], None
        if self.experiment.modalities == "auxiliary":
            return [dsm], dsm, encoded["dsm_last"], None
        return [rgb, dsm], rgb, encoded["rgb_last"], encoded["dsm_last"]

    def forward(self, img: torch.Tensor, dsm: torch.Tensor, labels: Optional[torch.Tensor] = None, is_train: bool = True):
        if img.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"Expected {self.img_size}x{self.img_size}, got {img.shape[-2:]}")
        encoded = self.image_encoder(img, dsm)
        pyramids, sam_pyramid, condition, second_condition = self._active(encoded)
        prompt_size = self.prompt_encoder.mask_input_size
        if self.experiment.prompt == "diffusion":
            prompt, supervised, diffusion_loss = generate_prompt(
                self.prompt_generator, condition, second_condition, labels, prompt_size,
                self.num_classes, self.ignore_index, self.diffusion_condition_mode, is_train,
                rgb_spatial=tuple(sam_pyramid),
                dsm_spatial=tuple(encoded["dsm_fpn"]) if second_condition is not None else None,
            )
        else:
            last_features = [encoded["rgb_last"]] if second_condition is None else [encoded["rgb_last"], encoded["dsm_last"]]
            prompt = self.deterministic_prompt(last_features, prompt_size)
            supervised = prompt
            diffusion_loss = prompt.sum() * 0.0 if is_train else None
        sam = decode_sam_features(
            self.prompt_encoder, self.mask_decoder, sam_pyramid[-2], self.prompt_to_sam_mask(prompt),
            sam_pyramid, self.image_embedding_size,
        )
        if self.experiment.decoder == "simple":
            logits = self.feature_decoder(pyramids, sam)
        elif self.experiment.modalities == "both":
            logits = self.feature_decoder(encoded["rgb_fpn"], encoded["dsm_fpn"], sam)
        else:
            logits = self.feature_decoder(sam_pyramid, sam)
        logits = F.interpolate(logits, size=img.shape[-2:], mode="bilinear", align_corners=False)
        output_prompt = F.interpolate(prompt, size=img.shape[-2:], mode="bilinear", align_corners=False)
        contrastive = None
        prompt_loss = None
        if is_train:
            if labels is None:
                raise ValueError("labels are required for training")
            contrastive = self.contrastive_loss(encoded["rgb_last"], encoded["dsm_last"], labels) if hasattr(self, "contrastive_loss") else logits.sum() * 0.0
            prompt_loss = prompt_segmentation_loss(supervised, labels, self.num_classes, self.ignore_index)
        return logits, output_prompt, diffusion_loss, contrastive, prompt_loss


def build_ablation_model(name: str, img_size: int = 1024, sam_checkpoint: Path | None = None) -> AblationMulMoSAM:
    config = get_config(name)
    if not config.train:
        raise ValueError(f"{name} is a Full alias, not a trainable ablation model")
    return AblationMulMoSAM(config, img_size=img_size, sam_checkpoint=sam_checkpoint)
