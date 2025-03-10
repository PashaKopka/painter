import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from model.config import RANDOM_SEED, BATCH_SIZE, NUM_WORKERS


class ImageDataset(Dataset):
    def __init__(
        self,
        x_images_paths: str,
        y_images_paths: str,
        transformations: Optional[callable] = None,
    ):
        self.root_x = x_images_paths
        self.root_y = y_images_paths

        self.transform = transformations

        self.x_images = os.listdir(x_images_paths)
        self.y_images = os.listdir(y_images_paths)

        self.x_length = len(self.x_images)
        self.y_length = len(self.y_images)

        # amount of images from both domains can be different
        # we set dataset length to the maximum amount of images
        self.length_dataset = max(self.x_length, self.y_length)

    def __getitem__(self, idx):
        # we should use modulo to avoid index out of range for
        # domain where length < length_dataset
        x_image = self.x_images[idx % self.x_length]
        y_image = self.y_images[idx % self.y_length]

        x_image = Image.open(os.path.join(self.root_x, x_image))
        y_image = Image.open(os.path.join(self.root_y, y_image))

        x_image = x_image.convert("RGB")
        y_image = y_image.convert("RGB")

        if self.transform:
            x_image = self.transform(x_image)
            y_image = self.transform(y_image)

        return x_image, y_image

    def __len__(self):
        return self.length_dataset


def get_loaders(
    x_path: str,
    y_path: str,
    train_transform: callable,
    val_transform: callable,
    batch_size: int = BATCH_SIZE,
    train_split: float = 0.8,
    num_workers: int = NUM_WORKERS,
):
    full_dataset = ImageDataset(x_path, y_path, transformations=None)

    # Compute split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Ensure deterministic split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    # Create separate train/val datasets with different transforms
    train_dataset = Subset(
        ImageDataset(x_path, y_path, transformations=train_transform), train_indices
    )
    val_dataset = Subset(
        ImageDataset(x_path, y_path, transformations=val_transform), val_indices
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
