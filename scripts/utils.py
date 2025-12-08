"""
utils.py
--------
Utility functions used across training and evaluation scripts.
"""

import torch

def load_checkpoint(model, ckpt_path):
    """Load a saved checkpoint (.pth file)."""
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

def compute_accuracy(true_labels, preds):
    """Compute accuracy from flattened lists."""
    true_labels = list(true_labels)
    preds = list(preds)
    correct = sum(t == p for t, p in zip(true_labels, preds))
    return correct / len(true_labels)
