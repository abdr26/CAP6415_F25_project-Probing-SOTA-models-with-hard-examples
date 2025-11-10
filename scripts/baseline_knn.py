"""
baseline_knn.py
---------------
This script implements the baseline experiment for:
"Probing SOTA Models with Hard Examples".

The goal is to extract feature representations from a pretrained
ResNet-50 model (trained on ImageNet) and train a simple k-Nearest
Neighbors (kNN) classifier on these features using the CIFAR-10 dataset.

This baseline provides a non-trainable reference for model performance
before fine-tuning or probing SOTA models.

--------------------------------------------------------------
Dependencies (install before running):
    pip install torch torchvision numpy pandas scikit-learn tqdm pillow
--------------------------------------------------------------

Command Line Execution:
    1. Activate your virtual environment:
        .\.venv\Scripts\Activate.ps1
    2. Run the baseline script:
        python .\scripts\baseline_knn.py

Expected Outputs (saved in /results):
    - confusion_matrix.csv
    - classification_report.txt
    - test_predictions_baseline.csv
    - class_names.csv
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data import cifar10_loaders


def extract_features(model, loader, device):
    """
    Extracts deep feature embeddings from an image dataset using a pretrained CNN.
    """
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting features"):
            x = x.to(device)
            f = model(x).cpu().numpy()  # Extract 2048-D features
            features.append(f)
            labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main():
    """
    Main routine for baseline experiment:
    - Loads CIFAR-10 data
    - Loads pretrained ResNet-50
    - Extracts feature representations
    - Trains and evaluates kNN classifier
    - Saves evaluation results
    """

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # 1️ Load dataset
    train_loader, test_loader, class_names = cifar10_loaders(batch_size=128)

    # 2️ Load pretrained ResNet-50 and remove classification layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Use ResNet-50 up to the average pooling layer (feature extractor)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor = nn.Sequential(feature_extractor, nn.Flatten()).to(device)

    # 3️ Extract features for train and test sets
    X_train, y_train = extract_features(feature_extractor, train_loader, device)
    X_test, y_test = extract_features(feature_extractor, test_loader, device)

    # 4️ Train k-Nearest Neighbor classifier
    print("\nTraining k-Nearest Neighbors (kNN) classifier...")
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    clf.fit(X_train, y_train)

    # 5️ Evaluate performance
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[Baseline] ResNet50-features + kNN → CIFAR-10 Accuracy: {accuracy*100:.2f}%")

    # 6️ Save confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

    np.savetxt("results/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    # 7️ Save per-sample predictions with confidence
    proba = clf.predict_proba(X_test)
    conf = proba.max(axis=1)

    pd.DataFrame({
        "true": y_test,
        "pred": y_pred,
        "conf": conf
    }).to_csv("results/test_predictions_baseline.csv", index=False)

    pd.Series(class_names).to_csv("results/class_names.csv",
                                  index_label="idx", header=["class"])

    print("\n All results saved in the 'results/' directory.")


# Entry point for script execution
if __name__ == "__main__":
    main()
