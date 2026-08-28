"""Evaluation and threshold-selection utilities."""

import torch


@torch.no_grad()
def evaluate_binary_segmentation(model, loader, device, threshold=0.5):
    """Return IoU, precision, recall, F1, accuracy and confusion counts."""
    model.eval()

    tp = fp = fn = tn = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        predictions = torch.sigmoid(outputs) >= threshold
        targets = masks >= 0.5

        tp += (predictions & targets).sum().item()
        fp += (predictions & ~targets).sum().item()
        fn += (~predictions & targets).sum().item()
        tn += (~predictions & ~targets).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "threshold": threshold,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def select_threshold_from_validation(model, val_loader, device, thresholds):
    """Select the threshold with the best validation F1, as in the notebook."""
    results = [
        evaluate_binary_segmentation(
            model, val_loader, device, threshold=t
        )
        for t in thresholds
    ]
    return max(results, key=lambda x: x["F1"]), results
