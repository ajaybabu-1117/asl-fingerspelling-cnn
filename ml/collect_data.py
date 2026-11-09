"""
Real-time ASL Data Collector using MediaPipe & OpenCV.
Enables collecting labeled hand-landmark images directly from the webcam for all letters A-Z.
"""

import os
import time
import argparse
import numpy as np
import cv2

try:
    import mediapipe as mp
    from preprocess import landmarks_to_image, CLASSES
except ImportError:
    print("Warning: mediapipe/opencv dependencies missing. Install using: pip install -r requirements.txt")

def collect_samples(label, num_samples=100, output_dir="dataset"):
    """
    Collects hand landmark images for a specific ASL character.
    """
    assert label.upper() in CLASSES, f"Label {label} must be one of A-Z"
    label = label.upper()
    save_path = os.path.join(output_dir, label)
    os.makedirs(save_path, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"\n==========================================")
    print(f" Collecting data for ASL Letter: [{label}]")
    print(f" Target samples: {num_samples}")
    print(f" Press [SPACE] to start recording.")
    print(f" Press [Q] to abort.")
    print(f"==========================================\n")

    recording = False
    count = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened() and count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            status_text = f"Letter: {label} | Collected: {count}/{num_samples}"
            if not recording:
                status_text += " [PAUSED - Press SPACE]"
            else:
                status_text += " [RECORDING...]"

            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if recording:
                        pts = []
                        for lm in hand_landmarks.landmark:
                            pts.append([lm.x, lm.y, lm.z])
                        pts = np.array(pts)

                        # Rasterize to 64x64 CNN input
                        lm_img = landmarks_to_image(pts, img_size=64)
                        # Save both numpy array and visualization png
                        sample_id = int(time.time() * 1000)
                        np_path = os.path.join(save_path, f"{label}_{sample_id}.npy")
                        np.save(np_path, pts)
                        
                        img_path = os.path.join(save_path, f"{label}_{sample_id}.png")
                        cv2.imwrite(img_path, (lm_img * 255).astype(np.uint8))
                        
                        count += 1
                        time.sleep(0.04)  # ~25 fps sampling

            cv2.imshow("ASL Hand Landmark Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                recording = not recording
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Saved {count} samples for '{label}' in {save_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect ASL Hand Landmark Data")
    parser.add_argument("--letter", type=str, default="A", help="ASL letter to collect (A-Z)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to capture")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()

    collect_samples(args.letter, args.samples, args.output)
