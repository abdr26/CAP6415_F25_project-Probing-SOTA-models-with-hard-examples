"""
data.py
--------
This module handles dataset loading and preprocessing for the project:
"Probing SOTA Models with Hard Examples".

It uses the CIFAR-10 dataset available through torchvision and prepares it
for SOTA vision models such as ResNet-50 and Vision Transformer (ViT).

Before running this module, ensure the following dependencies are installed:
    pip install torch torchvision

Functions:
-----------
cifar10_loaders(batch_size, num_workers, img_size, data_root)
    -> Returns train and test DataLoaders along with the class labels list.

Usage:
-------
    from data import cifar10_loaders
    train_loader, test_loader, classes = cifar10_loaders()
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets

# Normalization constants from ImageNet (used by pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cifar10_loaders(batch_size=64, num_workers=2, img_size=224, data_root="./data"):
    """
    Creates PyTorch DataLoaders for the CIFAR-10 dataset.
    """

    # Define transformations: resize, convert to tensor, normalize.
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Automatically downloads CIFAR-10.
    train_set = dsets.CIFAR10(root=data_root, train=True,
                              download=True, transform=train_tfms)
    test_set = dsets.CIFAR10(root=data_root, train=False,
                             download=True, transform=test_tfms)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_set.classes


# Example test run
if __name__ == "__main__":
    train_loader, test_loader, classes = cifar10_loaders()
    print(f"✅ CIFAR-10 dataset loaded successfully with classes: {classes}")
