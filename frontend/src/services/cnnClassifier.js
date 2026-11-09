/**
 * CNN Classifier Service
 * Loads TensorFlow.js model, normalizes MediaPipe 21 hand landmarks,
 * rasterizes them into a 64x64x1 hand skeleton tensor, and runs live inference.
 */

import * as tf from '@tensorflow/tfjs';
import { HAND_CONNECTIONS } from './handDetector';

const CLASSES = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)); // 'A'..'Z'
const INPUT_SIZE = 32;

let cnnModel = null;
let isModelLoading = false;
let loadError = null;

// Reusable offscreen canvas for rasterizing landmarks into 64x64 CNN input
const offscreenCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
if (offscreenCanvas) {
  offscreenCanvas.width = INPUT_SIZE;
  offscreenCanvas.height = INPUT_SIZE;
}

/**
 * Loads the CNN model from /model/model.json
 */
export async function loadASLModel() {
  if (cnnModel) return cnnModel;
  if (isModelLoading) return null;

  isModelLoading = true;
  loadError = null;

  try {
    // Attempt loading from web public model path
    cnnModel = await tf.loadLayersModel('/model/model.json');
    console.log('[ASL CNN] Model loaded successfully from /model/model.json');
    isModelLoading = false;
    return cnnModel;
  } catch (err) {
    console.error('[ASL CNN] Failed to load model from /model/model.json:', err);
    loadError = err.message || 'Failed to load model file';
    isModelLoading = false;
    return null;
  }
}

export function getModelStatus() {
  return {
    isLoaded: !!cnnModel,
    isLoading: isModelLoading,
    error: loadError
  };
}

/**
 * Normalizes 21 3D landmarks relative to wrist and bounding box,
 * renders bones and joints onto a 64x64 grayscale canvas,
 * and returns both the Float32Array pixel buffer and an image data URL for visualization.
 */
export function landmarksToRaster(landmarks, size = INPUT_SIZE) {
  if (!landmarks || landmarks.length !== 21 || !offscreenCanvas) return null;

  const ctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, size, size);

  // Translate relative to wrist (landmark 0)
  const wrist = landmarks[0];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

  const relPts = landmarks.map(lm => {
    const rx = lm.x - wrist.x;
    const ry = lm.y - wrist.y;
    if (rx < minX) minX = rx;
    if (rx > maxX) maxX = rx;
    if (ry < minY) minY = ry;
    if (ry > maxY) maxY = ry;
    return [rx, ry];
  });

  const spanX = Math.max(maxX - minX, 0.001);
  const spanY = Math.max(maxY - minY, 0.001);
  const maxSpan = Math.max(spanX, spanY);

  const margin = 8;
  const usable = size - margin * 2;
  const scale = usable / maxSpan;

  const pixelPts = relPts.map(([rx, ry]) => [
    Math.round(margin + (rx - minX) * scale),
    Math.round(margin + (ry - minY) * scale)
  ]);

  // Draw bone lines
  ctx.strokeStyle = '#bbbbbb';
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [s, e] of HAND_CONNECTIONS) {
    const p1 = pixelPts[s];
    const p2 = pixelPts[e];
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
  }

  // Draw joint circles
  for (let i = 0; i < pixelPts.length; i++) {
    const [px, py] = pixelPts[i];
    const isTip = [4, 8, 12, 16, 20].includes(i);
    const radius = isTip ? 3.5 : 2.5;

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Extract pixel intensities (normalize 0..255 to 0.0..1.0)
  const imgData = ctx.getImageData(0, 0, size, size);
  const pixels = new Float32Array(size * size);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = imgData.data[i * 4] / 255.0; // Grayscale intensity
  }

  return {
    pixels,
    previewUrl: offscreenCanvas.toDataURL('image/png')
  };
}

/**
 * Predicts ASL letter from normalized hand landmarks using the CNN model.
 */
export async function predictASLLetter(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  if (!cnnModel) {
    await loadASLModel();
    if (!cnnModel) return null;
  }

  const rasterResult = landmarksToRaster(landmarks, INPUT_SIZE);
  if (!rasterResult) return null;

  const startTime = performance.now();

  const prediction = tf.tidy(() => {
    // Shape: [1, 64, 64, 1]
    const tensor = tf.tensor4d(rasterResult.pixels, [1, INPUT_SIZE, INPUT_SIZE, 1]);
    const output = cnnModel.predict(tensor);
    return output.dataSync();
  });

  const latency = Math.round(performance.now() - startTime);

  // Find top class and probabilities
  let maxIdx = 0;
  let maxProb = -1;
  const topList = [];

  for (let i = 0; i < prediction.length; i++) {
    const prob = prediction[i];
    if (prob > maxProb) {
      maxProb = prob;
      maxIdx = i;
    }
    topList.push({ letter: CLASSES[i], probability: prob });
  }

  topList.sort((a, b) => b.probability - a.probability);

  return {
    letter: CLASSES[maxIdx],
    confidence: Math.round(maxProb * 1000) / 10, // e.g. 96.4
    rawScore: maxProb,
    latencyMs: latency,
    topPredictions: topList.slice(0, 3),
    rasterPreview: rasterResult.previewUrl
  };
}
