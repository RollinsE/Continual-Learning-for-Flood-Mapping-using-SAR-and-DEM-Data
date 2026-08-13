from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as func

from floods.losses.functional import lovasz_hinge


def _binary_logits_and_targets(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int):
    """Return finite float32 binary logits and targets over valid pixels only."""
    if preds.ndim == 4 and preds.shape[1] == 1:
        preds = preds[:, 0]
    mask = targets != ignore_index
    if mask.sum() == 0:
        return preds.reshape(-1)[:0].float(), targets.reshape(-1)[:0].float()
    preds = torch.nan_to_num(preds.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    preds = torch.clamp(preds, min=-30.0, max=30.0)
    targets = (targets > 0).float()
    return preds[mask], targets[mask]


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, reduction: str = "mean", ignore_index: int = 255, weight: torch.Tensor = None, pos_weight: torch.Tensor = None, **kwargs: dict):
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        # In this binary segmentation pipeline a scalar ``weight`` coming from
        # class-weight/pos-weight calculation is better interpreted as
        # BCEWithLogitsLoss(pos_weight=...), not as per-pixel weights.
        self.pos_weight = pos_weight if pos_weight is not None else weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds, targets = _binary_logits_and_targets(preds, targets, self.ignore_index)
        if targets.numel() == 0:
            return preds.sum() * 0.0
        pos_weight = self.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device=preds.device, dtype=preds.dtype)
        return func.binary_cross_entropy_with_logits(preds, targets, reduction=self.reduction, pos_weight=pos_weight)


class CombinedLoss(nn.Module):
    """Simply combines two losses into a single one, with weights.
    """

    def __init__(self,
                 criterion_a: Callable,
                 criterion_b: Callable,
                 weight_a: float = 1.0,
                 weight_b: float = 1.0,
                 **kwargs: dict):
        super().__init__()
        self.criterion_a = criterion_a(**kwargs)
        self.criterion_b = criterion_b(**kwargs)
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_a = self.criterion_a(preds, targets)
        loss_b = self.criterion_b(preds, targets)
        return self.weight_a * loss_a + self.weight_b * loss_b


class FocalLoss(nn.Module):
    """Simple implementation of focal loss.
    The focal loss can be seen as a generalization of the cross entropy, where more effort is put on
    hard examples, thanks to its gamma parameter.
    """

    def __init__(self,
                 reduction: str = "mean",
                 ignore_index: int = 255,
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 weight: torch.Tensor = None,
                 **kwargs: dict):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = torch.mean if reduction == "mean" else torch.sum
        self.pos_weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds, targets = _binary_logits_and_targets(preds, targets, self.ignore_index)
        if targets.numel() == 0:
            return preds.sum() * 0.0
        pos_weight = self.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device=preds.device, dtype=preds.dtype)
        ce_loss = func.binary_cross_entropy_with_logits(preds, targets, reduction='none', pos_weight=pos_weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt).clamp_min(0.0)**self.gamma * ce_loss
        return self.reduction(focal_loss)


class FocalTverskyLoss(nn.Module):
    """Custom implementation of a generalized Dice loss (called Tversky loss) with focal components.
    """

    def __init__(self,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None,
                 **kwargs: dict):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        # normalize weights so that they sum to 1
        if isinstance(weight, torch.Tensor):
            weight /= weight.sum()
        self.weight = weight if weight is not None else 1.0

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds, targets = _binary_logits_and_targets(preds, targets, self.ignore_index)
        if targets.numel() == 0:
            return preds.sum() * 0.0

        # Reductions are deliberately float32 even when AMP is enabled. Summing
        # millions of fp16 pixels can overflow and produce NaN gradients.
        probs = torch.sigmoid(preds.float())
        targets = targets.float()

        tp = (targets * probs).sum(dtype=torch.float32)
        fp = (probs * (1.0 - targets)).sum(dtype=torch.float32)
        fn = ((1.0 - probs) * targets).sum(dtype=torch.float32)

        eps = torch.finfo(torch.float32).eps
        index = self.weight * ((tp + eps) / (tp + self.alpha * fp + self.beta * fn + eps))
        index = torch.clamp(index, min=0.0, max=1.0)
        return torch.pow(1.0 - index.mean(), self.gamma)


class BCETverskyLoss(nn.Module):
    """Stable binary segmentation loss: BCEWithLogits + Tversky.

    BCE stabilises probability calibration and penalises false positives in a
    way pure Tversky often does not. Tversky keeps the loss sensitive to sparse
    foreground masks. ``weight`` is treated as a scalar positive-class weight
    for BCE when provided.
    """

    def __init__(self,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 gamma: float = 1.0,
                 bce_weight: float = 0.5,
                 tversky_weight: float = 0.5,
                 reduction: str = "mean",
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None,
                 **kwargs: dict):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.bce = BCEWithLogitsLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)
        self.tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, ignore_index=ignore_index)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(preds, targets) + self.tversky_weight * self.tversky(preds, targets)


class FocalTverskyComboLoss(nn.Module):
    """Focal BCE + Tversky for sparse flood masks.

    This is useful when many examples are easy background pixels but the model
    still needs a Tversky-style overlap term for flood extent.
    """

    def __init__(self,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 gamma: float = 1.0,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 1.0,
                 focal_weight: float = 0.5,
                 tversky_weight: float = 0.5,
                 reduction: str = "mean",
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None,
                 **kwargs: dict):
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.tversky_weight = float(tversky_weight)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction, ignore_index=ignore_index, weight=weight)
        self.tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, ignore_index=ignore_index)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(preds, targets) + self.tversky_weight * self.tversky(preds, targets)


class LovaszSoftmax(nn.Module):

    def __init__(self, classes='present', per_image=True, ignore_index=255, weight=None):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # Lovasz-Hinge does not currently apply the optional class-weight argument.
        loss = lovasz_hinge(preds, targets, ignore=self.ignore_index)
        return loss
