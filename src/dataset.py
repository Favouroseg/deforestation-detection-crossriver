"""Dataset and reproducible train/validation/test split utilities."""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeforestationDataset(Dataset):
    """Load paired Sentinel-2 image patches and binary deforestation masks."""

    def __init__(self, images_dir: str, masks_dir: str):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patches = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image = torch.from_numpy(
            np.load(os.path.join(self.images_dir, f"patch_{idx}.npy"))
        ).float()

        mask = torch.from_numpy(
            np.load(os.path.join(self.masks_dir, f"patch_{idx}.npy"))
        ).float().unsqueeze(0)

        # Sentinel-2 values were scaled by 10,000 before model input.
        image = image / 10000.0

        return image, mask


def create_dataloaders(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 4,
    seed: int = SEED,
):
    """Create the fixed 70/15/15 split used in the experiments."""
    set_seed(seed)

    dataset = DeforestationDataset(images_dir, masks_dir)

    total = len(dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return dataset, train_loader, val_loader, test_loader
