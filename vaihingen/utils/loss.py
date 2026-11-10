import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, weight=None, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, pred, ground_truth):
        num_classes = pred.shape[1]
        labels = ground_truth.long()
        valid = (labels >= 0) & (labels < num_classes)
        if self.ignore_index is not None:
            valid &= labels.ne(self.ignore_index)
        safe_labels = labels.masked_fill(~valid, 0)

        probabilities = F.softmax(pred, dim=1) * valid.unsqueeze(1)
        one_hot = F.one_hot(safe_labels, num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        cardinality = (probabilities + one_hot).sum(dims)
        valid_classes = cardinality > 0
        if not valid_classes.any():
            return pred.sum() * 0.0

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        if self.weight is not None:
            weight = torch.as_tensor(self.weight, device=pred.device, dtype=pred.dtype)
            return 1.0 - (dice[valid_classes] * weight[valid_classes]).sum() / weight[
                valid_classes
            ].sum().clamp_min(1.0e-6)
        return 1.0 - dice[valid_classes].mean()


class ClsLoss(nn.Module):
    def __init__(
        self,
        ignore_index=255,
        weight=None,
        dice_weight=1.0,
        cross_entropy_weight=1.0,
        smooth=1.0,
        boundary_weight=1.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.boundary_weight = float(boundary_weight)
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.dice_loss = DiceLoss(ignore_index, weight, smooth)
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index
        )

    def forward(self, pred, true):
        labels = true.long()
        invalid = ((labels < 0) | (labels >= pred.shape[1])) & labels.ne(
            self.ignore_index
        )
        labels = labels.masked_fill(invalid, self.ignore_index)
        if not labels.ne(self.ignore_index).any():
            return pred.sum() * 0.0
        ce_weight = self.cross_entropy_loss.weight
        if ce_weight is not None:
            ce_weight = ce_weight.to(device=pred.device, dtype=pred.dtype)
        loss = (
            self.dice_weight * self.dice_loss(pred, labels)
            + self.cross_entropy_weight
            * F.cross_entropy(
                pred,
                labels,
                weight=ce_weight,
                ignore_index=self.ignore_index,
            )
        )
        if self.boundary_weight <= 1.0:
            return loss
        valid = labels.ne(self.ignore_index)
        padded = F.pad(labels.unsqueeze(1).float(), (1, 1, 1, 1), mode="replicate")
        center = padded[:, :, 1:-1, 1:-1]
        boundary = torch.zeros_like(center, dtype=torch.bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = padded[:, :, 1 + dy : 1 + dy + labels.shape[-2], 1 + dx : 1 + dx + labels.shape[-1]]
            boundary |= neighbor.ne(center)
        boundary = boundary[:, 0] & valid
        if not boundary.any():
            return loss
        pixel_loss = F.cross_entropy(
            pred,
            labels,
            weight=ce_weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        weights = torch.ones_like(pixel_loss)
        weights[boundary] = self.boundary_weight
        weighted_ce = (pixel_loss * weights).sum() / weights[valid].sum().clamp_min(1.0)
        return self.dice_weight * self.dice_loss(pred, labels) + self.cross_entropy_weight * weighted_ce
