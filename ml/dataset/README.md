# ASL Alphabet Dataset Setup Guide

This project recognizes American Sign Language (ASL) alphabet fingerspelling (A–Z) using **MediaPipe Hand Landmarks** converted into normalized **Hand-Landmark Skeleton Images** classified by a **Deep Convolutional Neural Network (CNN)**.

---

## 1. Dataset Options

### Option A: Authentic Webcam Hand Collector (Recommended for custom setups)
Collect real landmark samples from your own webcam with our collector tool:
```bash
# Capture 100 samples for letter 'A'
python collect_data.py --letter A --samples 100

# Capture for all 26 letters (A through Z)
python collect_data.py --letter B --samples 100
...
```
- Press **[SPACE]** to start/pause recording.
- Press **[Q]** to finish.
- Data will be saved in `dataset/<LETTER>/` as both 21-landmark `.npy` files and 64x64 landmark `.png` images.

---

### Option B: Kaggle ASL Alphabet Dataset
You can use the popular Kaggle ASL Alphabet dataset:
- **Kaggle URL**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Contains 87,000 images (3,000 per class) covering 29 classes (A–Z, nothing, space, del).

To download using Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d dataset/
```

Then extract landmarks from raw images into landmark images:
```bash
python -m pip install -r requirements.txt
python train.py --dataset dataset
```

---

## 2. Landmark Rasterization & Preprocessing
Each hand detection by MediaPipe produces 21 3D keypoints:
1. **Wrist translation**: Wrist landmark $(x_0, y_0)$ is translated to origin $(0, 0)$.
2. **Scale normalization**: All coordinates are scaled by hand span $\max \|p_i - p_0\|$.
3. **Rasterization**: Skeletal bone connections and joints are rendered onto a $64 \times 64 \times 1$ image map using antialiased lines.
4. **CNN Input**: Shape `(64, 64, 1)` with pixel intensities in range `[0.0, 1.0]`.

This ensures lighting, skin tone, and background clutter are completely decoupled from gesture geometry!
