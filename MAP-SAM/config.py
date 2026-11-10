"""Fixed experiment configuration shared by training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Tuple


# Version 10 adds four-scale DSM residual fusion in the semantic decoder and
# records the 8-4 non-zero multimodal residual initialization policy.
CHECKPOINT_FORMAT_VERSION = 10
# Version 3 keeps foreground-only macro metrics but defines OA over all valid pixels.
METRIC_POLICY_VERSION = 3


@dataclass(frozen=True)
class ModelConfig:
    """Architecture values that must stay consistent with a checkpoint.

    Tensor-shape symbols used throughout the model documentation are ``B``
    (batch size), ``K`` (semantic classes), and ``H``/``W`` (input height and
    width). The default input is 1024 x 1024 and ``K=6``.
    """

    # Square RGB/DSM crop size. The diffusion pyramid requires a multiple of 128.
    img_size: int = 1024
    # Number of semantic-logit channels and diffusion prompt channels.
    num_classes: int = 6
    # Label value excluded from segmentation and auxiliary supervision.
    ignore_index: int = 255
    # Number of deterministic DDIM denoising iterations used at evaluation.
    diffusion_steps: int = 10
    # Diffusion conditioning source: "rgb" or jointly gated "rgb_dsm".
    diffusion_condition_mode: str = "rgb_dsm"
    # Initial RGB share of each dual-modal FiLM gate; DSM receives 1 - value.
    diffusion_rgb_init_weight: float = 0.9
    # Inject aligned FPN maps into all four prompt-diffusion encoder stages.
    diffusion_use_spatial_condition: bool = True
    # Initial residual scale for each spatial-condition injection block.
    diffusion_spatial_scale_init: float = 0.05
    # Softmax temperature used by the RGB/DSM contrastive objective.
    contrastive_temperature: float = 0.07
    # Relative weight of same-position RGB/DSM positives inside that objective.
    position_contrastive_weight: float = 1.0
    # Relative weight of same-semantic-class positives inside that objective.
    class_contrastive_weight: float = 1.0
    # Maximum number of spatial tokens sampled for contrastive loss per batch.
    contrastive_max_samples: int = 1024
    # One-based Hiera stages where bidirectional RGB/DSM attention is applied.
    cross_stages: Tuple[int, ...] = (1, 2, 3, 4)
    # Initial bidirectional RGB/DSM cross-attention residual gate.
    cross_modal_scale_init: float = 0.05
    # SCA/SRA bottleneck ratio ``r/d`` for every frozen Hiera block.
    # Smaller adapter capacity reduces overfitting on the small Vaihingen split.
    adapter_reduction: float = 0.125
    # Parallel depth-wise kernels used by the gated spatial SCA branch.
    sca_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    # SRA expansion relative to its ``r/2`` operated branch; 4 gives width 2r.
    sra_expansion_ratio: float = 4.0
    # Initial scale of the pooled global MLP branch in SRA.
    sra_global_scale_init: float = 0.05
    # Initial external per-output-channel adapter scale ``gamma``.
    adapter_gamma_init: float = 1.0
    # Initial internal SCA/SRA residual scale.
    adapter_inner_scale_init: float = 0.05
    # Initial residual scale for direct DSM fusion at all four decoder levels.
    decoder_dsm_scale_init: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "ModelConfig":
        if not values:
            return cls()
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        if "cross_stages" in kwargs:
            kwargs["cross_stages"] = tuple(kwargs["cross_stages"])
        if "sca_kernel_sizes" in kwargs:
            kwargs["sca_kernel_sizes"] = tuple(kwargs["sca_kernel_sizes"])
        return cls(**kwargs)


@dataclass(frozen=True)
class TrainingConfig:
    """Stable training policy; CLI flags are reserved for run-time choices."""

    # Experiment/checkpoint name used by logging and command-line defaults.
    net_type: str = "MAP-SAM"
    # Seed for initialization, data order, augmentation, and training noise.
    random_seed: int = 42
    # Seed reset before every validation/test pass for fixed diffusion noise.
    evaluation_noise_seed: int = 42
    # Number of linear learning-rate warm-up epochs.
    warm_epochs: int = 5
    # Validation metric whose best value selects the saved checkpoint.
    reference_metric: str = "mIoU"
    # Whether macro metrics include the final clutter/background class.
    include_background: bool = False
    # Learning-rate multiplier for pretrained SAM2 prompt and mask heads.
    pretrained_lr_scale: float = 0.1
    # AdamW decoupled weight-decay coefficient.
    weight_decay: float = 0.005
    # Initial and final weights of diffusion denoising supervision.
    diffusion_weight: float = 0.1
    diffusion_weight_final: float = 0.01
    # Initial and final weights of semantic supervision on diffusion prompts.
    prompt_segmentation_weight: float = 0.025
    prompt_segmentation_weight_final: float = 0.002
    # Initial and final weights of RGB/DSM contrastive supervision.
    contrastive_weight: float = 0.01
    contrastive_weight_final: float = 0.001
    # Inclusive epoch interval over which auxiliary weights decay linearly.
    auxiliary_decay_start_epoch: int = 20
    auxiliary_decay_end_epoch: int = 40
    # Number of physical mini-batches accumulated before one optimizer step.
    accumulation_steps: int = 1
    # Maximum global L2 gradient norm; non-positive values would disable it.
    grad_clip_norm: float = 1.0
    # Early stopping monitors the configured validation reference metric.
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.05
    # Multiplicative CE weight for valid label-boundary pixels.
    boundary_pixel_weight: float = 3.0
    # Stable class weighting policy computed from train-pixel frequencies.
    class_weight_mode: str = "sqrt_inverse"
    # DSM normalization ablation: ``per_image`` is the historical baseline;
    # ``global`` uses train-split mean/std; ``ndsm`` uses relative height;
    # ``absolute_relative`` returns two DSM channels (absolute + relative).
    dsm_normalization: str = "absolute_relative"
    # Clamp normalized DSM values to this symmetric range.
    dsm_clip: float = 5.0
    # Synchronized train augmentation: ``geometric`` avoids aspect-ratio
    # distortion; ``legacy`` retains RandomResizedCrop; ``none`` disables it.
    augmentation_policy: str = "geometric"
    # Fraction of the shortest side sampled by the geometric square crop.
    augmentation_crop_scale: Tuple[float, float] = (0.75, 1.0)
    # Probabilities for synchronized horizontal/vertical flips and 90-degree rotation.
    augmentation_hflip_prob: float = 0.5
    augmentation_vflip_prob: float = 0.5
    augmentation_rot90_prob: float = 0.5
    # RGB-only light photometric jitter; DSM and labels are never color-jittered.
    augmentation_color_jitter_prob: float = 0.2
    augmentation_color_jitter_strength: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "TrainingConfig":
        if not values:
            return cls()
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        if "augmentation_crop_scale" in kwargs:
            kwargs["augmentation_crop_scale"] = tuple(kwargs["augmentation_crop_scale"])
        return cls(**kwargs)


MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
