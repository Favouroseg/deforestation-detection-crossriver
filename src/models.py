"""Model definitions used in the deforestation segmentation experiments."""

import torch
import segmentation_models_pytorch as smp


def build_vanilla_unet(device=None):
    """
    Build the model labelled 'Vanilla U-Net' in the experiment.

    Note: the original notebook implements this as an SMP U-Net with
    a ResNet18 encoder and no pretrained weights.
    """
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=4,
        classes=1,
    )
    return model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def build_resnet34_unet(device=None, pretrained=False):
    """Build the ResNet34 U-Net used in the experiments."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=4,
        classes=1,
    )
    return model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def load_checkpoint(model, checkpoint_path, device=None):
    """Load either a raw state_dict or the notebook's checkpoint dictionary."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model.to(device), checkpoint
