"""
evaluate.py
-----------
Evaluates the fine-tuned model on the CIFAR-10 test set.

Outputs:
    - test accuracy
    - per-class accuracy (printed via classification report)
    - confusion matrix (CSV)
    - prediction CSV (index, true label, predicted label, confidence)

Execution:
    Google Colab (recommended)

    # 1) Clone repository and cd into project root
    # 2) Make sure the checkpoint exists:
    #       models/resnet50_finetuned.pth
    # 3) Run:
    #       !python scripts/evaluate.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models

from data import cifar10_loaders
from utils import load_checkpoint, compute_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint is stored under models/
CHECKPOINT_PATH = "models/resnet50_finetuned.pth"


def main():
    print(f"Using device: {DEVICE}")
    _, test_loader, class_names = cifar10_loaders(batch_size=64)

    # Make sure results folder exists
    os.makedirs("results", exist_ok=True)

    # Load model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    load_checkpoint(model, CHECKPOINT_PATH)

    # Evaluate
    all_preds = []
    all_labels = []
    all_conf = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            probs = nn.Softmax(dim=1)(outputs)
            conf, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())

    acc = compute_accuracy(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save files into results/
    # 1) Confusion matrix
    cm_path = os.path.join("results", "confusion_matrix_finetuned.csv")
    pd.DataFrame(cm).to_csv(cm_path, index=False)

    # 2) Predictions with index column (for Grad-CAM / hard examples)
    num_samples = len(all_labels)
    preds_df = pd.DataFrame({
        "index": list(range(num_samples)),  # position in CIFAR-10 test set
        "true": all_labels,
        "pred": all_preds,
        "conf": all_conf,
    })
    preds_path = os.path.join("results", "test_predictions_finetuned.csv")
    preds_df.to_csv(preds_path, index=False)

    print("\nSaved evaluation outputs in 'results/' folder:")
    print(f"  - {cm_path}")
    print(f"  - {preds_path}")


if __name__ == "__main__":
    main()
