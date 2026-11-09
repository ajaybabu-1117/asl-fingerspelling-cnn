/**
 * MediaPipe Hand Landmarker Service
 * Initializes vision WASM and executes real-time hand detection on webcam video frames.
 */

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

let landmarkerInstance = null;

// MediaPipe 21 Hand Landmark skeletal connections
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index finger
  [9, 10], [10, 11], [11, 12],          // Middle finger
  [13, 14], [14, 15], [15, 16],         // Ring finger
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky finger
  [5, 9], [9, 13], [13, 17]              // Knuckle palm connections
];

export async function getHandLandmarker() {
  if (landmarkerInstance) return landmarkerInstance;

  const wasmFileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  const modelAssetPath = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

  try {
    landmarkerInstance = await HandLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath,
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.65,
      minHandPresenceConfidence: 0.65,
      minTrackingConfidence: 0.65
    });
  } catch (err) {
    console.warn('GPU acceleration unavailable for MediaPipe, falling back to CPU:', err);
    landmarkerInstance = await HandLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath,
        delegate: 'CPU'
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.65,
      minHandPresenceConfidence: 0.65,
      minTrackingConfidence: 0.65
    });
  }

  return landmarkerInstance;
}

/**
 * Draws hand landmarks and glowing skeletal connections onto an overlay 2D canvas.
 */
export function drawHandLandmarks(ctx, landmarks, width, height) {
  if (!landmarks || landmarks.length === 0) return;

  const points = landmarks.map(lm => ({
    x: lm.x * width,
    y: lm.y * height
  }));

  // Draw skeletal connections
  ctx.save();
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = 'rgba(139, 92, 246, 0.85)';
  ctx.shadowColor = '#8b5cf6';
  ctx.shadowBlur = 10;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = points[startIdx];
    const p2 = points[endIdx];
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
  ctx.restore();

  // Draw landmark points
  ctx.save();
  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    const isTip = [4, 8, 12, 16, 20].includes(i);
    const radius = isTip ? 6.5 : 4.5;

    // Outer glow ring
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius + 2, 0, 2 * Math.PI);
    ctx.fillStyle = isTip ? 'rgba(6, 182, 212, 0.4)' : 'rgba(139, 92, 246, 0.3)';
    ctx.fill();

    // Core point
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = isTip ? '#06b6d4' : '#c084fc';
    ctx.shadowColor = isTip ? '#06b6d4' : '#c084fc';
    ctx.shadowBlur = 8;
    ctx.fill();

    // Inner bright center
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }
  ctx.restore();
}
