"""Loss functions used by the segmentation experiments."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


_bce = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets, smooth=1.0):
    """Dice loss matching the custom Dice implementation in the notebook."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()

    dice = (2 * intersection + smooth) / (
        probs.sum() + targets.sum() + smooth
    )
    return 1 - dice


def bce_dice_loss(logits, targets):
    """BCE + custom Dice loss used for the non-improved models."""
    return _bce(logits, targets) + dice_loss(logits, targets)


_smp_dice = smp.losses.DiceLoss(mode="binary")
_smp_focal = smp.losses.FocalLoss(mode="binary")


def combined_dice_bce_focal_loss(logits, targets):
    """
    Combined loss used for the improved ResNet34 experiment:
    Dice + 0.5*BCE + 0.5*Focal.
    """
    return (
        _smp_dice(logits, targets)
        + 0.5 * _bce(logits, targets)
        + 0.5 * _smp_focal(logits, targets)
    )
