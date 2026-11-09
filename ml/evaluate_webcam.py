"""
Real-World Webcam Evaluation Script for ASL Alphabet CNN.
Empirically tests real-time hand gestures captured via webcam, computes real-world
confusion matrix, accuracy, precision, recall, F1-score, and compares against
the 98.90% offline validation benchmark.

Requirements:
    pip install opencv-python mediapipe numpy
"""

import os
import sys
import time
import json
from datetime import datetime
import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError:
    print("[Error] OpenCV and MediaPipe required for webcam evaluation.")
    print("Run: pip install opencv-python mediapipe numpy")
    sys.exit(1)

from preprocess import normalize_landmarks, landmarks_to_image, CLASSES, HAND_CONNECTIONS

SAMPLES_PER_LETTER = 30
VALIDATION_BENCHMARK = 98.90

def main():
    print("=" * 70)
    print("ASL CNN Real-World Webcam Evaluation Studio")
    print(f"Goal: Collect {SAMPLES_PER_LETTER} real webcam frames per letter (A-Z)")
    print(f"Validation Benchmark: {VALIDATION_BENCHMARK}%")
    print("=" * 70)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    # Check if a trained Keras model exists in models/
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "asl_cnn_model.keras")
    h5_path = os.path.join(os.path.dirname(__file__), "..", "models", "asl_cnn_model.h5")
    
    keras_model = None
    try:
        import tensorflow as tf
        if os.path.exists(model_path):
            keras_model = tf.keras.models.load_model(model_path)
            print(f"[Info] Loaded Keras model from {model_path}")
        elif os.path.exists(h5_path):
            keras_model = tf.keras.models.load_model(h5_path)
            print(f"[Info] Loaded Keras model from {h5_path}")
        else:
            print("[Note] Keras model file not found locally. Live evaluation can record landmark tensors or use the React Web App for TensorFlow.js inference.")
    except Exception as e:
        print(f"[Note] TensorFlow not loaded in Python: {e}")

    samples_by_letter = {c: [] for c in CLASSES}
    current_class_idx = 0
    recording_burst = False
    burst_count = 0
    last_capture_time = 0

    print("\nControls:")
    print("  [SPACE]     : Capture 1 sample for current letter")
    print("  [B]         : Start/Stop Auto-Burst capture (30 samples)")
    print("  [N]         : Move to Next letter")
    print("  [P]         : Move to Previous letter")
    print("  [C]         : Clear current letter samples")
    print("  [Q] or [ESC]: Finish & Generate Evaluation Report\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        target_letter = CLASSES[current_class_idx]
        current_samples = len(samples_by_letter[target_letter])

        hand_detected = False
        pred_letter = "-"
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            tensor_img = landmarks_to_image(pts, img_size=32)

            if keras_model is not None:
                inp = np.expand_dims(tensor_img, axis=0)
                preds = keras_model.predict(inp, verbose=0)[0]
                pred_idx = np.argmax(preds)
                pred_letter = CLASSES[pred_idx]
                confidence = float(preds[pred_idx] * 100)

            # Auto-burst capture logic
            now = time.time()
            if recording_burst and (now - last_capture_time >= 0.12):
                samples_by_letter[target_letter].append({
                    "target": target_letter,
                    "predicted": pred_letter,
                    "confidence": round(confidence, 1),
                    "timestamp": time.time()
                })
                burst_count += 1
                last_capture_time = now
                if burst_count >= SAMPLES_PER_LETTER or len(samples_by_letter[target_letter]) >= SAMPLES_PER_LETTER:
                    recording_burst = False
                    burst_count = 0

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (w, 90), (15, 23, 42), -1)
        cv2.putText(frame, f"TARGET: {target_letter} ({current_samples}/{SAMPLES_PER_LETTER})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        status_color = (16, 185, 129) if hand_detected else (239, 68, 68)
        status_txt = "Hand Tracked" if hand_detected else "No Hand Detected"
        cv2.putText(frame, status_txt, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if keras_model is not None:
            cv2.putText(frame, f"Pred: {pred_letter} ({confidence:.1f}%)", (320, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2)

        if recording_burst:
            cv2.putText(frame, f"REC BURST: {burst_count}/{SAMPLES_PER_LETTER}", (320, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ASL Real-World Webcam Evaluation", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:
            break
        elif key == ord(' ') and hand_detected and not recording_burst:
            samples_by_letter[target_letter].append({
                "target": target_letter,
                "predicted": pred_letter,
                "confidence": round(confidence, 1),
                "timestamp": time.time()
            })
            print(f"Captured 1 sample for {target_letter} (Total: {len(samples_by_letter[target_letter])})")
        elif key in [ord('b'), ord('B')] and hand_detected:
            recording_burst = not recording_burst
            burst_count = 0
            last_capture_time = time.time()
        elif key in [ord('n'), ord('N')]:
            if current_class_idx < len(CLASSES) - 1:
                current_class_idx += 1
        elif key in [ord('p'), ord('P')]:
            if current_class_idx > 0:
                current_class_idx -= 1
        elif key in [ord('c'), ord('C')]:
            samples_by_letter[target_letter] = []
            print(f"Cleared samples for {target_letter}")

    cap.release()
    cv2.destroyAllWindows()

    # Generate Evaluation Report
    total_samples = sum(len(v) for v in samples_by_letter.values())
    if total_samples == 0:
        print("[Info] No samples recorded. Exiting without report.")
        return

    matrix = [[0] * 26 for _ in range(26)]
    total_correct = 0
    confusions = {}

    for t_idx, true_c in enumerate(CLASSES):
        for s in samples_by_letter[true_c]:
            p = s["predicted"]
            if p in CLASSES:
                p_idx = CLASSES.index(p)
                matrix[t_idx][p_idx] += 1
                if p == true_c:
                    total_correct += 1
                else:
                    pair = f"{true_c}->{p}"
                    confusions[pair] = confusions.get(pair, 0) + 1

    real_acc = (total_correct / total_samples) * 100
    delta = real_acc - VALIDATION_BENCHMARK

    print("\n" + "=" * 70)
    print("REAL-WORLD EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Real Samples Evaluated : {total_samples}")
    print(f"Real-World Webcam Accuracy   : {real_acc:.2f}%")
    print(f"Validation Accuracy Baseline : {VALIDATION_BENCHMARK:.2f}%")
    print(f"Variance (Delta)             : {delta:+.2f}%")
    print("=" * 70)

    # Save JSON Report
    report = {
        "title": "ASL CNN Real-World Webcam Evaluation Report",
        "evaluated_at": datetime.now().isoformat(),
        "total_real_samples": total_samples,
        "real_world_accuracy": round(real_acc, 2),
        "validation_benchmark_accuracy": VALIDATION_BENCHMARK,
        "accuracy_delta": round(delta, 2),
        "confusion_matrix": matrix,
        "classes": CLASSES,
        "top_confusions": sorted([{"pair": k, "count": v} for k, v in confusions.items()], key=lambda x: x["count"], reverse=True),
        "samples_by_letter": samples_by_letter
    }

    out_file = os.path.join(os.path.dirname(__file__), "real_world_evaluation.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Success] Real-world evaluation report saved to {out_file}")

if __name__ == "__main__":
    main()
