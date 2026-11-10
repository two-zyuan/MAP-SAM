"""RGB/DSM feature extraction with the frozen SAM2 Hiera backbone."""

from .multimodal_sam2_encoder import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
    SAM2Encoder,
    SAM2_Encoder,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MODEL_CONFIG",
    "SAM2Encoder",
    "SAM2_Encoder",
]
