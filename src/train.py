"""Reusable training loop for binary deforestation segmentation."""

import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch and return mean training loss."""
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validation_iou(model, loader, device, threshold=0.5):
    """Calculate pixel-level IoU on a validation loader."""
    model.eval()
    intersection = 0.0
    union = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        outputs = model(images)
        preds = torch.sigmoid(outputs) >= threshold
        targets = masks >= 0.5

        intersection += (preds & targets).float().sum().item()
        union += (preds | targets).float().sum().item()

    return intersection / (union + 1e-8)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs=40,
    checkpoint_path=None,
):
    """
    Train a model using validation IoU for ReduceLROnPlateau scheduling
    and best-checkpoint selection.
    """
    best_iou = -1.0
    history = {"train_loss": [], "val_iou": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_iou = validation_iou(model, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_iou)

        history["train_loss"].append(train_loss)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if checkpoint_path and val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_iou": val_iou,
                },
                checkpoint_path,
            )

    return history, best_iou
