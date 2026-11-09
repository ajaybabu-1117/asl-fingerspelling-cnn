"""
Train pipeline for ASL Fingerspelling CNN.
Loads landmark dataset, applies data augmentation, trains CNN,
evaluates on validation set, and exports model for TensorFlow.js browser inference.
"""

import os
import json
import glob
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import landmarks_to_image, augment_landmarks, CLASSES, CLASS_TO_IDX
from model import build_asl_cnn

def load_data_from_directory(dataset_dir="dataset", samples_per_class=200):
    """
    Loads landmark arrays or pre-saved numpy landmark files.
    If fewer samples exist, generates augmented landmark variations.
    """
    X_imgs = []
    y_labels = []

    for label in CLASSES:
        class_idx = CLASS_TO_IDX[label]
        class_dir = os.path.join(dataset_dir, label)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))

        collected_pts = []
        for f in npy_files:
            try:
                pts = np.load(f)
                if pts.shape[0] == 21:
                    collected_pts.append(pts)
            except Exception:
                continue

        if len(collected_pts) == 0:
            print(f"[Notice] No recorded samples for letter '{label}' in {class_dir}")
            continue

        for pts in collected_pts:
            X_imgs.append(landmarks_to_image(pts))
            y_labels.append(class_idx)

        # If data augmentation is needed to reach target samples
        if len(collected_pts) < samples_per_class:
            deficit = samples_per_class - len(collected_pts)
            for i in range(deficit):
                base_pts = collected_pts[i % len(collected_pts)]
                aug_pts = augment_landmarks(base_pts)
                X_imgs.append(landmarks_to_image(aug_pts))
                y_labels.append(class_idx)

    if len(X_imgs) == 0:
        return None, None

    return np.array(X_imgs, dtype=np.float32), np.array(y_labels, dtype=np.int32)

def export_tfjs_model(keras_model, output_dir="../frontend/public/model"):
    """Exports trained Keras model to TensorFlow.js web model format."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(keras_model, output_dir)
        print(f"[SUCCESS] TensorFlow.js model exported to: {output_dir}")
    except ImportError:
        print("[WARNING] tensorflowjs package not installed. Saved native Keras model only.")

def run_training(dataset_dir="dataset", epochs=25, batch_size=32, output_dir="../models"):
    os.makedirs(output_dir, exist_ok=True)
    web_model_dir = os.path.join(output_dir, "web_model")
    frontend_model_dir = "../frontend/public/model"

    print("=" * 60)
    print(" ASL Fingerspelling CNN Training Pipeline")
    print("=" * 60)

    X, y = load_data_from_directory(dataset_dir)

    if X is None or len(X) == 0:
        print("\n[INFO] No local dataset found in 'dataset/' directory.")
        print("To train with your own hand gestures:")
        print("1. Run: python collect_data.py --letter A --samples 100 (repeat for B-Z)")
        print("2. Or download the ASL Alphabet dataset from Kaggle as detailed in dataset/README.md")
        return False

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Total samples: {len(X)} | Train: {len(X_train)} | Validation: {len(X_val)}")

    model = build_asl_cnn(input_shape=(64, 64, 1), num_classes=len(CLASSES))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    y_pred = np.argmax(model.predict(X_val), axis=1)
    report = classification_report(y_val, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_val, y_pred).tolist()

    metrics = {
        "status": "trained",
        "accuracy": float(val_acc),
        "loss": float(val_loss),
        "epochs_trained": len(history.history['loss']),
        "train_loss": [float(v) for v in history.history['loss']],
        "val_loss": [float(v) for v in history.history['val_loss']],
        "train_accuracy": [float(v) for v in history.history['accuracy']],
        "val_accuracy": [float(v) for v in history.history['val_accuracy']],
        "confusion_matrix": cm,
        "classes": CLASSES,
        "summary": {
            "macro_avg_f1": float(report["macro avg"]["f1-score"]),
            "macro_avg_precision": float(report["macro avg"]["precision"]),
            "macro_avg_recall": float(report["macro avg"]["recall"])
        }
    }

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save Keras model
    keras_path = os.path.join(output_dir, "asl_landmark_cnn.keras")
    model.save(keras_path)
    print(f"[SUCCESS] Saved Keras model: {keras_path}")

    # Export to TFJS
    export_tfjs_model(model, web_model_dir)
    export_tfjs_model(model, frontend_model_dir)

    # Copy metrics to frontend public folder for the dashboard
    frontend_metrics_path = "../frontend/public/evaluation_metrics.json"
    with open(frontend_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_training(args.dataset, args.epochs, args.batch_size)
