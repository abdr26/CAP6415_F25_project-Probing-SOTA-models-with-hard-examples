"""
train.py
--------
Fine-tune a pretrained SOTA vision model (ResNet-50) on CIFAR-10.

This script:
  • Loads the CIFAR-10 dataset via data.py  
  • Fine-tunes a pretrained ResNet-50 on top of ImageNet weights  
  • Tracks loss & accuracy  
  • Saves best model checkpoint  
  • Logs training progress to console and CSV

--------------------------------------------------------------
Dependencies:
    pip install torch torchvision tqdm pandas matplotlib
--------------------------------------------------------------

Execution:
    Google Colab
    # Train the model on T4 GPU
    cloned repository
    !python scripts/train.py
--------------------------------------------------------------
"""

import os, time, torch, torch.nn as nn, torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torchvision import models
from data import cifar10_loaders

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-4
CHECKPOINT_PATH = "models/resnet50_finetuned.pth"
LOG_CSV = "results/train_log.csv"

# Training & Evaluation Helpers
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """Runs one training epoch and returns average loss & accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def evaluate(model, loader, criterion):
    """Evaluates model on validation/test set."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return loss_sum / total, correct / total

# Main Training Loop
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1 Data
    train_loader, test_loader, class_names = cifar10_loaders(batch_size=BATCH_SIZE)

    # 2️ Model setup (ResNet-50)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)     # 10 CIFAR-10 classes
    model = model.to(DEVICE)

    # 3️ Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4️ Training loop
    best_acc = 0.0
    log_rows = []

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        elapsed = time.time() - start

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Time: {elapsed/60:.2f} min")

        log_rows.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_min": round(elapsed/60, 2)
        })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f" Saved best model → {CHECKPOINT_PATH}")

    # 5️ Save training log
    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)
    print(f" Training log saved → {LOG_CSV}")

#  Entry Point
if __name__ == "__main__":
    main()
