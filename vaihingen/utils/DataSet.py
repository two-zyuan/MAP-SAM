import logging
import os
from typing import Optional, Tuple

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PotsdamDataset(Dataset):
    RGB_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    RGB_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    def __init__(
        self,
        root,
        query,
        if_transform=True,
        image_size=1024,
        dsm_normalization="per_image",
        dsm_stats: Optional[Tuple[float, float]] = None,
        dsm_clip=5.0,
        augmentation_policy="geometric",
        crop_scale=(0.75, 1.0),
        hflip_prob=0.5,
        vflip_prob=0.5,
        rot90_prob=0.5,
        color_jitter_prob=0.2,
        color_jitter_strength=0.1,
    ):
        """Create a synchronized RGB/DSM/label dataset.

        Args:
            dsm_normalization: ``per_image``, ``global``, ``ndsm``, or
                ``absolute_relative``. The last returns two DSM channels:
                global absolute height and relative height.
            dsm_stats: Train-split ``(mean, std)`` used by global/nDSM modes.
            augmentation_policy: ``geometric``, ``legacy``, or ``none``.
            crop_scale: Fraction range for the geometric square crop.
            hflip_prob/vflip_prob/rot90_prob: Synchronized augmentation
                probabilities applied to RGB, DSM, and labels.
            color_jitter_prob/color_jitter_strength: RGB-only light brightness,
                contrast, and saturation augmentation.
        """
        self.root = root
        self.query = query
        self.if_transform = if_transform
        self.image_size = image_size
        valid_norm = {"per_image", "global", "ndsm", "absolute_relative"}
        if dsm_normalization not in valid_norm:
            raise ValueError(f"Unsupported DSM normalization: {dsm_normalization}")
        valid_aug = {"geometric", "legacy", "none"}
        if augmentation_policy not in valid_aug:
            raise ValueError(f"Unsupported augmentation policy: {augmentation_policy}")
        if dsm_normalization in {"global", "ndsm", "absolute_relative"} and dsm_stats is None:
            raise ValueError("dsm_stats=(mean, std) is required for this normalization")
        self.dsm_normalization = dsm_normalization
        self.dsm_stats = dsm_stats
        self.dsm_clip = float(dsm_clip)
        self.augmentation_policy = augmentation_policy
        self.crop_scale = tuple(crop_scale)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rot90_prob = float(rot90_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=0.0,
        )
        label_dir = os.path.join(root, query, "gray_label")
        self.ids = sorted(os.path.splitext(file)[0] for file in os.listdir(label_dir))
        logging.info("Creating dataset with %d examples", len(self.ids))

    @staticmethod
    def compute_dsm_stats(root, query="train") -> Tuple[float, float]:
        """Compute finite train-split DSM mean/std without loading all images."""
        dsm_dir = os.path.join(root, query, "dsm")
        count = 0
        total = 0.0
        total_sq = 0.0
        for filename in sorted(os.listdir(dsm_dir)):
            values = np.asarray(tiff.imread(os.path.join(dsm_dir, filename)), dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            count += int(finite.size)
            total += float(finite.sum())
            total_sq += float(np.square(finite).sum())
        if count == 0:
            return 0.0, 1.0
        mean = total / count
        variance = max(total_sq / count - mean * mean, 1.0e-12)
        return float(mean), float(np.sqrt(variance))

    @staticmethod
    def compute_class_weights(
        root,
        query="train",
        num_classes=6,
        mode="sqrt_inverse",
    ) -> Tuple[float, ...]:
        """Compute stable inverse-frequency weights from train labels."""
        if mode not in {"none", "sqrt_inverse"}:
            raise ValueError(f"Unsupported class-weight mode: {mode}")
        if mode == "none":
            return tuple(1.0 for _ in range(num_classes))
        label_dir = os.path.join(root, query, "gray_label")
        counts = np.zeros(num_classes, dtype=np.float64)
        for filename in sorted(os.listdir(label_dir)):
            labels = np.asarray(
                tiff.imread(os.path.join(label_dir, filename)), dtype=np.int64
            ).reshape(-1)
            valid = labels[(labels >= 0) & (labels < num_classes)]
            if valid.size:
                counts += np.bincount(valid, minlength=num_classes)
        if counts.sum() <= 0:
            return tuple(1.0 for _ in range(num_classes))
        frequencies = counts / counts.sum()
        weights = 1.0 / np.sqrt(np.maximum(frequencies, 1.0e-12))
        weights /= weights.mean()
        return tuple(float(value) for value in weights)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(image):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        return torch.from_numpy(image.transpose((2, 0, 1)))

    def _aligned_transform(self, image, dsm, label):
        output_size = (self.image_size, self.image_size)
        if self.if_transform and self.augmentation_policy != "none":
            if self.augmentation_policy == "legacy":
                crop = transforms.RandomResizedCrop.get_params(
                    image, scale=self.crop_scale, ratio=(0.75, 1.3333333333333333)
                )
            elif self.augmentation_policy == "geometric":
                height, width = image.shape[-2:]
                side_min = min(height, width)
                min_scale, max_scale = self.crop_scale
                side = int(round(side_min * float(torch.empty(()).uniform_(min_scale, max_scale))))
                side = max(1, min(side, side_min))
                top = int(torch.randint(0, height - side + 1, ()).item())
                left = int(torch.randint(0, width - side + 1, ()).item())
                crop = (top, left, side, side)
            else:
                crop = None
            if crop is not None:
                image = TF.resized_crop(image, *crop, output_size, InterpolationMode.BILINEAR)
                dsm = TF.resized_crop(dsm, *crop, output_size, InterpolationMode.BILINEAR)
                label = TF.resized_crop(label, *crop, output_size, InterpolationMode.NEAREST)
            if torch.rand(()) < self.hflip_prob:
                image, dsm, label = TF.hflip(image), TF.hflip(dsm), TF.hflip(label)
            if torch.rand(()) < self.vflip_prob:
                image, dsm, label = TF.vflip(image), TF.vflip(dsm), TF.vflip(label)
            if torch.rand(()) < self.rot90_prob:
                turns = int(torch.randint(1, 4, ()).item())
                image = torch.rot90(image, turns, dims=(-2, -1))
                dsm = torch.rot90(dsm, turns, dims=(-2, -1))
                label = torch.rot90(label, turns, dims=(-2, -1))
            if torch.rand(()) < self.color_jitter_prob:
                # ColorJitter expects floating-point images in [0, 1]. Keep the
                # dataset's historical [0, 255] range until RGB normalization.
                image = self.color_jitter(image / 255.0) * 255.0
        elif image.shape[-2:] != output_size:
            image = TF.resize(image, output_size, InterpolationMode.BILINEAR)
            dsm = TF.resize(dsm, output_size, InterpolationMode.BILINEAR)
            label = TF.resize(label, output_size, InterpolationMode.NEAREST)
        return image, dsm, label

    @classmethod
    def _normalize_rgb(cls, image):
        image = image / 255.0
        return (image - cls.RGB_MEAN) / cls.RGB_STD

    def _normalize_dsm(self, dsm):
        finite = torch.isfinite(dsm)
        if not finite.any():
            return torch.zeros((2 if self.dsm_normalization == "absolute_relative" else 1, *dsm.shape[-2:]), dtype=dsm.dtype)
        values = dsm[finite]
        image_mean = values.mean()
        image_std = values.std(unbiased=False).clamp_min(1.0e-6)
        dsm = torch.nan_to_num(dsm, nan=float(image_mean), posinf=float(image_mean), neginf=float(image_mean))
        if self.dsm_normalization == "per_image":
            return ((dsm - image_mean) / image_std).clamp(-self.dsm_clip, self.dsm_clip)

        global_mean, global_std = self.dsm_stats
        global_mean = dsm.new_tensor(global_mean)
        global_std = dsm.new_tensor(max(float(global_std), 1.0e-6))
        absolute = ((dsm - global_mean) / global_std).clamp(-self.dsm_clip, self.dsm_clip)
        baseline = torch.quantile(values, 0.02)
        relative = ((dsm - baseline) / global_std).clamp(-self.dsm_clip, self.dsm_clip)
        if self.dsm_normalization == "global":
            return absolute
        if self.dsm_normalization == "ndsm":
            return relative
        return torch.cat((absolute, relative), dim=0)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        base = os.path.join(self.root, self.query)
        image = self.preprocess(
            tiff.imread(os.path.join(base, "img", sample_id + ".tif"))
        ).float()
        dsm = self.preprocess(
            tiff.imread(os.path.join(base, "dsm", sample_id + ".tif"))
        ).float()
        label = self.preprocess(
            tiff.imread(os.path.join(base, "gray_label", sample_id + ".tif"))
        ).float()
        image, dsm, label = self._aligned_transform(image, dsm, label)
        return {
            "id": sample_id,
            "img": self._normalize_rgb(image),
            "dsm": self._normalize_dsm(dsm),
            "mask": label.long(),
        }
