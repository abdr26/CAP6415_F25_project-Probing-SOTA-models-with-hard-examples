"""
gradcam.py
----------
Generates Grad-CAM heatmaps for hard examples (misclassified and low-confidence).

Execution (Google Colab):

    # 1) Clone repository and cd into project root
    # 2) Ensure the following files exist:
    #       models/resnet50_finetuned.pth
    #       results/misclassified.csv
    #       results/low_confidence.csv
    # 3) Run:
    #       !python scripts/gradcam.py

This script will create:
    gradcam_outputs/misclassified/*.png
    gradcam_outputs/lowconf/*.png
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torchvision import models, transforms, datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "models/resnet50_finetuned.pth"   # fine-tuned model checkpoint
BASE_OUTPUT_DIR = "gradcam_outputs"            # root folder for all Grad-CAM images

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# CIFAR-10 normalization (must match training/eval)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])


def load_model():
    """Load the fine-tuned ResNet-50 model."""
    print(f"Loading model from: {MODEL_PATH}")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def generate_gradcam_for_csv(csv_file, tag, model, testset):
    """
    Generate Grad-CAM visualizations for all samples listed in csv_file.

    Args:
        csv_file: path to CSV file (e.g., 'results/misclassified.csv')
        tag:      string to distinguish outputs ('misclassified' or 'lowconf')
        model:    loaded ResNet-50 model
        testset:  CIFAR-10 test dataset
    """
    print(f"\n=== Processing {csv_file} ({tag}) ===")

    if not os.path.exists(csv_file):
        print(f"[WARN] CSV file not found: {csv_file}. Skipping.")
        return

    df = pd.read_csv(csv_file)

    if "index" not in df.columns:
        raise ValueError(f"CSV file {csv_file} does NOT contain an 'index' column. "
                         "Ensure evaluate.py writes an 'index' column.")

    # Ensure index column is integer
    df["index"] = df["index"].astype(int)

    out_dir = os.path.join(BASE_OUTPUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Total samples in {csv_file}: {len(df)}")
    print(f"Saving Grad-CAM images to: {out_dir}")

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for _, row in df.iterrows():
        idx = int(row["index"])

        # Safety check to avoid IndexError
        if idx < 0 or idx >= len(testset):
            print(f"[WARN] Skipping invalid index {idx} (out of range).")
            continue

        img, _ = testset[idx]
        input_tensor = img.unsqueeze(0).to(DEVICE)

        # Use predicted class from CSV as target
        pred_class = int(row["pred"])
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)]
        )[0]

        # Convert to [0,1] RGB for overlay
        rgb = img.permute(1, 2, 0).cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        out_path = os.path.join(out_dir, f"gradcam_{tag}_{idx}.png")
        cv2.imwrite(out_path, vis)

        print("Saved:", out_path)

    print(f"Finished Grad-CAM for {csv_file}.\n")


def main():
    print("Using device:", DEVICE)

    # Load CIFAR-10 test set (same root as in data.py / training)
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Load fine-tuned model once and reuse
    model = load_model()

    # 1) Misclassified examples
    generate_gradcam_for_csv(
        "results/misclassified.csv",
        tag="misclassified",
        model=model,
        testset=testset,
    )

    # 2) Low-confidence examples
    generate_gradcam_for_csv(
        "results/low_confidence.csv",
        tag="lowconf",
        model=model,
        testset=testset,
    )


if __name__ == "__main__":
    main()
