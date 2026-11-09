"""
Model Evaluation Script for ASL Fingerspelling CNN.
Generates confusion matrix plots, classification reports, and metric visual assets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import CLASSES

def plot_evaluation(metrics_json_path="../models/evaluation_metrics.json", output_dir="../models/plots"):
    """
    Plots training curves and confusion matrix from evaluation metrics JSON.
    """
    if not os.path.exists(metrics_json_path):
        print(f"Error: metrics file {metrics_json_path} not found. Run train.py first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)

    if metrics.get("status") != "trained":
        print("Model has not completed training yet.")
        return

    epochs = range(1, metrics["epochs_trained"] + 1)

    # 1. Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, metrics["train_accuracy"], 'b-', label='Training Acc', lw=2)
    ax1.plot(epochs, metrics["val_accuracy"], 'g--', label='Validation Acc', lw=2)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs, metrics["train_loss"], 'r-', label='Training Loss', lw=2)
    ax2.plot(epochs, metrics["val_loss"], 'm--', label='Validation Loss', lw=2)
    ax2.set_title("Model Loss Across Epochs", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[Saved] Training curve plot: {curve_path}")

    # 2. Confusion Matrix Heatmap
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Purples",
            xticklabels=CLASSES,
            yticklabels=CLASSES
        )
        plt.title("ASL Fingerspelling CNN Confusion Matrix (A-Z)", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[Saved] Confusion matrix plot: {cm_path}")

if __name__ == '__main__':
    plot_evaluation()
