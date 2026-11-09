/**
 * Enhanced CNN Classifier Service
 * Uses the unified landmarkRasterizer to guarantee 100% preprocessing parity with training.
 * Incorporates Exponential Moving Average (EMA) temporal smoothing, confidence thresholding,
 * and memory-safe tensor cleanup.
 */

import * as tf from '@tensorflow/tfjs';
import { renderLandmarksToBuffer, bufferToDataUrl, RASTER_SIZE } from './landmarkRasterizer';

const CLASSES = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)); // 'A'..'Z'
const CONFIDENCE_THRESHOLD = 50; // Minimum percentage to accept a stable prediction

let cnnModel = null;
let isModelLoading = false;
let loadError = null;

// Temporal probability smoothing buffer
let smoothedProbabilities = null;
const SMOOTHING_ALPHA = 0.65; // Weight given to newest frame (0.65 new, 0.35 historical)

/**
 * Loads the CNN model from /model/model.json
 */
export async function loadASLModel() {
  if (cnnModel) return cnnModel;
  if (isModelLoading) return null;

  isModelLoading = true;
  loadError = null;

  try {
    cnnModel = await tf.loadLayersModel('/model/model.json');
    console.log('[ASL CNN] Enhanced model loaded successfully from /model/model.json');
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

export function resetClassifierSmoothing() {
  smoothedProbabilities = null;
}

/**
 * Predicts ASL letter from normalized hand landmarks using the CNN model with temporal smoothing.
 */
export async function predictASLLetter(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    resetClassifierSmoothing();
    return null;
  }

  if (!cnnModel) {
    await loadASLModel();
    if (!cnnModel) return null;
  }

  // 1. Unified rasterization (guarantees exact parity with training dataset)
  const rasterResult = renderLandmarksToBuffer(landmarks, RASTER_SIZE);
  if (!rasterResult) return null;

  const startTime = performance.now();

  // 2. Predict inside tf.tidy to avoid WebGL tensor memory leaks
  const rawProbabilities = tf.tidy(() => {
    // Shape: [1, 32, 32, 1]
    const tensor = tf.tensor4d(rasterResult.buffer, [1, RASTER_SIZE, RASTER_SIZE, 1]);
    const output = cnnModel.predict(tensor);
    return Array.from(output.dataSync());
  });

  const latency = Math.round(performance.now() - startTime);

  // 3. Temporal smoothing via Exponential Moving Average (EMA)
  if (!smoothedProbabilities) {
    smoothedProbabilities = rawProbabilities.slice();
  } else {
    for (let i = 0; i < 26; i++) {
      smoothedProbabilities[i] =
        SMOOTHING_ALPHA * rawProbabilities[i] +
        (1 - SMOOTHING_ALPHA) * smoothedProbabilities[i];
    }
  }

  // 4. Find maximum probability class
  let maxIdx = 0;
  let maxProb = -1;
  const topList = [];

  for (let i = 0; i < smoothedProbabilities.length; i++) {
    const prob = smoothedProbabilities[i];
    if (prob > maxProb) {
      maxProb = prob;
      maxIdx = i;
    }
    topList.push({ letter: CLASSES[i], probability: prob });
  }

  topList.sort((a, b) => b.probability - a.probability);

  const confidencePct = Math.round(maxProb * 1000) / 10; // e.g. 94.2
  const isConfident = confidencePct >= CONFIDENCE_THRESHOLD;

  // Generate preview image data URL
  const previewUrl = bufferToDataUrl(rasterResult.buffer, RASTER_SIZE);

  return {
    letter: isConfident ? CLASSES[maxIdx] : '...',
    rawLetter: CLASSES[maxIdx],
    confidence: confidencePct,
    isConfident,
    isFlipped: rasterResult.isFlipped,
    latencyMs: latency,
    topPredictions: topList.slice(0, 3),
    rasterPreview: previewUrl
  };
}
