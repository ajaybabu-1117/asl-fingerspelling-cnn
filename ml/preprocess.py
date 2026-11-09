"""
Preprocess module for ASL Fingerspelling Recognition.
Handles landmark extraction, spatial normalization, and rasterization
of 21 MediaPipe hand landmarks into 64x64 landmark skeleton images for CNN classification.
"""

import numpy as np
import cv2

# ASL Alphabet classes: A to Z (26 classes)
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# MediaPipe standard 21-hand landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

def normalize_landmarks(landmarks):
    """
    Normalizes 21 landmarks relative to wrist and scales to bounding box.
    Automatically applies handedness/mirroring invariance: ensures thumb points left.
    landmarks: shape (21, 2) or (21, 3)
    Returns: normalized array of shape (21, 2)
    """
    pts = np.array(landmarks)[:, :2].copy()
    
    # Check handedness: if thumb MCP (landmark 2) has x > pinky MCP (landmark 17), flip x
    if pts[2, 0] > pts[17, 0]:
        pts[:, 0] = -pts[:, 0]
    
    # Translate wrist (landmark 0) to origin
    wrist = pts[0].copy()
    pts -= wrist
    
    # Scale invariance: divide by maximum span
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist
    
    return pts

def landmarks_to_image(landmarks, img_size=32):
    """
    Renders normalized 21 hand landmarks onto a 2D grayscale image (img_size x img_size)
    with bones and joints drawn with antialiased lines.
    Output shape: (img_size, img_size, 1), values in [0.0, 1.0]
    """
    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    
    pts = normalize_landmarks(landmarks)
    
    # Fit into image bounds with margin (margin = 3 pixels)
    margin = 3.0
    usable_size = img_size - 2 * margin
    
    # Find bounding box of normalized points
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    range_xy = np.maximum(max_xy - min_xy, 1e-5)
    scale = usable_size / np.max(range_xy)
    
    pixel_pts = (pts - min_xy) * scale + margin
    pixel_pts = np.clip(pixel_pts, 0, img_size - 1).astype(np.int32)
    
    # Draw bone connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = tuple(pixel_pts[start_idx])
        pt2 = tuple(pixel_pts[end_idx])
        cv2.line(canvas, pt1, pt2, color=0.6, thickness=1, lineType=cv2.LINE_AA)
    
    # Draw landmark joints (dots)
    for idx, pt in enumerate(pixel_pts):
        radius = 2 if idx in [4, 8, 12, 16, 20] else 1  # Fingertips
        cv2.circle(canvas, tuple(pt), radius=radius, color=1.0, thickness=-1, lineType=cv2.LINE_AA)
        
    return np.expand_dims(canvas, axis=-1)

def augment_landmarks(landmarks, angle_range=15, scale_range=(0.85, 1.15), noise_std=0.015):
    """
    Applies spatial data augmentation directly to hand landmarks:
    - Random 2D in-plane rotation
    - Random scaling
    - Random jitter noise
    """
    pts = landmarks.copy()
    
    # Random rotation
    angle_deg = np.random.uniform(-angle_range, angle_range)
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    pts[:, :2] = np.dot(pts[:, :2], rot_matrix)
    
    # Random scale
    scale = np.random.uniform(*scale_range)
    pts[:, :2] *= scale
    
    # Gaussian jitter noise
    pts[:, :2] += np.random.normal(0, noise_std, size=pts[:, :2].shape)
    
    return pts
