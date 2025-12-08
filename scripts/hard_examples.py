"""
hard_examples.py
----------------
Extracts:
- Misclassified samples
- Low-confidence predictions
from results/test_predictions_finetuned.csv

Outputs:
    results/misclassified.csv
    results/low_confidence.csv

Execution (Google Colab or PowerShell):

    # After running evaluate.py
    # CSV must exist in: results/test_predictions_finetuned.csv

    !python scripts/hard_examples.py          # Colab
    python .\scripts\hard_examples.py         # PowerShell
"""

import os
import pandas as pd

# Input file path
CSV_PATH = "results/test_predictions_finetuned.csv"

# Low-confidence threshold
CONF_THRESHOLD = 0.40

def main():
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    # Load predictions CSV
    df = pd.read_csv(CSV_PATH)

    # 1. Misclassified samples
    misclassified = df[df["true"] != df["pred"]]
    misclassified_path = "results/misclassified.csv"
    misclassified.to_csv(misclassified_path, index=False)
    print(f"Saved {len(misclassified)} misclassified samples → {misclassified_path}")

    # 2. Low-confidence predictions
    low_conf = df[df["conf"] < CONF_THRESHOLD]
    low_conf_path = "results/low_confidence.csv"
    low_conf.to_csv(low_conf_path, index=False)
    print(f"Saved {len(low_conf)} low-confidence samples (<{CONF_THRESHOLD}) → {low_conf_path}")


if __name__ == "__main__":
    main()
