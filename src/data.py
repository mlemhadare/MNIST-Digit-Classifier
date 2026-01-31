"""
data.py
Data loading, preprocessing, and EDA utilities for MNIST classification.
"""
import os
from typing import Tuple, Any
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_data_loaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download MNIST, split into train/val/test, and return DataLoaders.

    Args:
        data_dir (str): Directory to store/download MNIST data.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, Val, Test DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    total_train = len(full_train)
    val_size = int(val_split * total_train)
    train_size = total_train - val_size
    train, val = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def plot_sample_images(data_loader: DataLoader, n: int = 10) -> None:
    """
    Display a grid of n random sample images with their labels.

    Args:
        data_loader (DataLoader): DataLoader to sample from.
        n (int): Number of images to display.
    """
    images, labels = next(iter(data_loader))
    plt.figure(figsize=(12, 2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i][0], cmap="gray")
        plt.title(str(labels[i].item()))
        plt.axis("off")
    plt.tight_layout()
    plt.show()
