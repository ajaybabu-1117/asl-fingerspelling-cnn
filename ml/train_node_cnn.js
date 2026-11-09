/**
 * Node.js CNN Training Script for ASL Alphabet Landmark Recognition
 * Generates canonical ASL landmark poses, synthesizes augmented training/validation sets,
 * trains a genuine Convolutional Neural Network (CNN) with TensorFlow.js,
 * evaluates classification performance (Accuracy, Precision, Recall, F1, Confusion Matrix),
 * and exports model.json + weights.bin for browser inference.
 */

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

const CLASSES = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)); // 'A'..'Z'
const IMG_SIZE = 64;

// Standard MediaPipe hand connections
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [9, 10], [10, 11], [11, 12],          // Middle
  [13, 14], [14, 15], [15, 16],         // Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17]              // Knuckles
];

/**
 * Creates canonical 21-landmark coordinates [x, y] for a given ASL letter.
 * Baseline coordinates are normalized with wrist near [0.5, 0.88].
 */
function getCanonicalLandmarks(letter) {
  // Base wrist & knuckle MCP positions
  const wrist = [0.5, 0.88];
  const mcpThumb = [0.38, 0.72];
  const mcpIndex = [0.38, 0.58];
  const mcpMid = [0.48, 0.55];
  const mcpRing = [0.58, 0.57];
  const mcpPinky = [0.66, 0.62];

  // Helper to make finger joints
  // state: 'up', 'folded', 'hook', 'cross', 'horizontal', 'down'
  function makeFinger(mcp, state, angleOffset = 0, lengthScale = 1.0) {
    const [mx, my] = mcp;
    if (state === 'up') {
      const dx = Math.sin(angleOffset);
      const dy = -Math.cos(angleOffset);
      return [
        [mx + dx * 0.10 * lengthScale, my + dy * 0.10 * lengthScale],
        [mx + dx * 0.22 * lengthScale, my + dy * 0.22 * lengthScale],
        [mx + dx * 0.35 * lengthScale, my + dy * 0.35 * lengthScale]
      ];
    } else if (state === 'folded') {
      return [
        [mx - 0.02, my + 0.05],
        [mx - 0.01, my + 0.11],
        [mx + 0.01, my + 0.14]
      ];
    } else if (state === 'curled') {
      return [
        [mx - 0.02, my - 0.08],
        [mx + 0.02, my - 0.02],
        [mx + 0.03, my + 0.06]
      ];
    } else if (state === 'hook') {
      return [
        [mx, my - 0.12],
        [mx + 0.05, my - 0.06],
        [mx + 0.04, my + 0.01]
      ];
    } else if (state === 'horizontal') {
      return [
        [mx - 0.10, my - 0.02],
        [mx - 0.22, my - 0.03],
        [mx - 0.34, my - 0.04]
      ];
    } else if (state === 'down') {
      return [
        [mx - 0.02, my + 0.12],
        [mx - 0.04, my + 0.24],
        [mx - 0.06, my + 0.35]
      ];
    }
    return [[mx, my + 0.05], [mx, my + 0.10], [mx, my + 0.14]];
  }

  // Define finger states based on ASL letter
  let thumbState = 'folded';
  let indexState = 'folded';
  let midState = 'folded';
  let ringState = 'folded';
  let pinkyState = 'folded';
  let indexAngle = 0, midAngle = 0, ringAngle = 0, pinkyAngle = 0;

  switch (letter) {
    case 'A': // Fist with thumb alongside index
      thumbState = 'up_side';
      break;
    case 'B': // 4 fingers extended up, thumb folded across palm
      indexState = 'up'; midState = 'up'; ringState = 'up'; pinkyState = 'up';
      thumbState = 'across_palm';
      break;
    case 'C': // Curled C shape
      indexState = 'curled'; midState = 'curled'; ringState = 'curled'; pinkyState = 'curled';
      thumbState = 'curled_c';
      break;
    case 'D': // Index up, others touch thumb in circle
      indexState = 'up';
      midState = 'curled'; ringState = 'curled'; pinkyState = 'curled';
      thumbState = 'touch_mid';
      break;
    case 'E': // All curled tightly, thumb folded under
      indexState = 'folded'; midState = 'folded'; ringState = 'folded'; pinkyState = 'folded';
      thumbState = 'across_under';
      break;
    case 'F': // Index and thumb touch, middle/ring/pinky up
      indexState = 'curled';
      midState = 'up'; ringState = 'up'; pinkyState = 'up';
      midAngle = -0.05; ringAngle = 0.05; pinkyAngle = 0.15;
      thumbState = 'touch_index';
      break;
    case 'G': // Index pointing horizontal, thumb parallel
      indexState = 'horizontal';
      thumbState = 'horizontal_thumb';
      break;
    case 'H': // Index and middle horizontal together
      indexState = 'horizontal'; midState = 'horizontal';
      thumbState = 'horizontal_thumb';
      break;
    case 'I': // Pinky extended up, others folded
      pinkyState = 'up';
      thumbState = 'across_palm';
      break;
    case 'J': // Pinky up with curve
      pinkyState = 'up';
      pinkyAngle = -0.2;
      thumbState = 'across_palm';
      break;
    case 'K': // Index up, middle angled forward, thumb between
      indexState = 'up';
      midState = 'up'; midAngle = 0.25;
      thumbState = 'between';
      break;
    case 'L': // Index up, thumb perpendicular out (L shape)
      indexState = 'up';
      thumbState = 'out_l';
      break;
    case 'M': // 3 fingers folded over thumb
      indexState = 'folded'; midState = 'folded'; ringState = 'folded';
      thumbState = 'tucked_m';
      break;
    case 'N': // 2 fingers folded over thumb
      indexState = 'folded'; midState = 'folded';
      thumbState = 'tucked_n';
      break;
    case 'O': // All fingers touch thumb tip forming O
      indexState = 'curled'; midState = 'curled'; ringState = 'curled'; pinkyState = 'curled';
      thumbState = 'touch_o';
      break;
    case 'P': // Pointing downwards (like K)
      indexState = 'down'; midState = 'down';
      thumbState = 'down_thumb';
      break;
    case 'Q': // Pointing downwards (like G)
      indexState = 'down';
      thumbState = 'down_thumb';
      break;
    case 'R': // Index and middle crossed up
      indexState = 'up'; indexAngle = 0.08;
      midState = 'up'; midAngle = -0.08;
      thumbState = 'across_palm';
      break;
    case 'S': // Fist with thumb across front
      thumbState = 'front_fist';
      break;
    case 'T': // Thumb between index and middle
      indexState = 'folded';
      thumbState = 'tucked_t';
      break;
    case 'U': // Index and middle up together touching
      indexState = 'up'; midState = 'up';
      indexAngle = 0.02; midAngle = -0.02;
      thumbState = 'across_palm';
      break;
    case 'V': // Index and middle spread in V (peace)
      indexState = 'up'; midState = 'up';
      indexAngle = -0.18; midAngle = 0.18;
      thumbState = 'across_palm';
      break;
    case 'W': // Index, middle, ring spread in W
      indexState = 'up'; midState = 'up'; ringState = 'up';
      indexAngle = -0.22; midAngle = 0.0; ringAngle = 0.22;
      thumbState = 'across_palm';
      break;
    case 'X': // Index hooked
      indexState = 'hook';
      thumbState = 'across_palm';
      break;
    case 'Y': // Thumb and pinky extended (shaka)
      pinkyState = 'up'; pinkyAngle = 0.35;
      thumbState = 'out_y';
      break;
    case 'Z': // Index pointing up/diagonal
      indexState = 'up'; indexAngle = -0.2;
      thumbState = 'across_palm';
      break;
  }

  // Thumb joints
  let thumbJoints;
  if (thumbState === 'out_l') {
    thumbJoints = [[0.30, 0.70], [0.22, 0.68], [0.14, 0.67]];
  } else if (thumbState === 'out_y') {
    thumbJoints = [[0.28, 0.72], [0.18, 0.74], [0.10, 0.78]];
  } else if (thumbState === 'up_side') {
    thumbJoints = [[0.32, 0.65], [0.30, 0.56], [0.29, 0.48]];
  } else if (thumbState === 'touch_index' || thumbState === 'touch_o') {
    thumbJoints = [[0.34, 0.65], [0.35, 0.55], [0.38, 0.48]];
  } else if (thumbState === 'touch_mid') {
    thumbJoints = [[0.36, 0.65], [0.40, 0.58], [0.45, 0.54]];
  } else if (thumbState === 'horizontal_thumb') {
    thumbJoints = [[0.32, 0.68], [0.24, 0.66], [0.16, 0.64]];
  } else if (thumbState === 'down_thumb') {
    thumbJoints = [[0.34, 0.76], [0.32, 0.84], [0.30, 0.92]];
  } else if (thumbState === 'front_fist') {
    thumbJoints = [[0.40, 0.66], [0.46, 0.64], [0.52, 0.63]];
  } else { // folded / across palm
    thumbJoints = [[0.40, 0.68], [0.45, 0.66], [0.48, 0.65]];
  }

  const idxJoints = makeFinger(mcpIndex, indexState, indexAngle);
  const midJoints = makeFinger(mcpMid, midState, midAngle);
  const ringJoints = makeFinger(mcpRing, ringState, ringAngle);
  const pinkyJoints = makeFinger(mcpPinky, pinkyState, pinkyAngle, 0.85);

  const landmarks = [
    wrist,
    mcpThumb, thumbJoints[0], thumbJoints[1], thumbJoints[2],
    mcpIndex, idxJoints[0], idxJoints[1], idxJoints[2],
    mcpMid, midJoints[0], midJoints[1], midJoints[2],
    mcpRing, ringJoints[0], ringJoints[1], ringJoints[2],
    mcpPinky, pinkyJoints[0], pinkyJoints[1], pinkyJoints[2]
  ];

  return landmarks;
}

/**
 * Augment landmarks with random rotation, scale, translation, and jitter.
 */
function augmentLandmarks(pts) {
  const angle = (Math.random() - 0.5) * 0.35; // +/- ~10 degrees
  const scale = 0.88 + Math.random() * 0.24;   // 0.88 to 1.12
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  // Center around palm (MCP mid point = index 9)
  const cx = pts[9][0];
  const cy = pts[9][1];

  return pts.map(([x, y]) => {
    // Relative to center
    let rx = (x - cx) * scale;
    let ry = (y - cy) * scale;
    // Rotate
    let nx = rx * cos - ry * sin;
    let ny = rx * sin + ry * cos;
    // Add small jitter
    nx += (Math.random() - 0.5) * 0.012;
    ny += (Math.random() - 0.5) * 0.012;
    return [nx + cx, ny + cy];
  });
}

/**
 * Normalizes and rasterizes 21 landmarks into a 64x64 float32 grayscale image buffer.
 */
function landmarksToImage(pts, size = IMG_SIZE) {
  const img = new Float32Array(size * size);

  // Normalize relative to wrist (point 0)
  const [wx, wy] = pts[0];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const rel = pts.map(([x, y]) => {
    const rx = x - wx;
    const ry = y - wy;
    if (rx < minX) minX = rx;
    if (rx > maxX) maxX = rx;
    if (ry < minY) minY = ry;
    if (ry > maxY) maxY = ry;
    return [rx, ry];
  });

  const spanX = Math.max(maxX - minX, 0.01);
  const spanY = Math.max(maxY - minY, 0.01);
  const maxSpan = Math.max(spanX, spanY);

  const margin = 8;
  const usable = size - margin * 2;
  const scale = usable / maxSpan;

  const pixelPts = rel.map(([rx, ry]) => {
    const px = Math.round(margin + (rx - minX) * scale);
    const py = Math.round(margin + (ry - minY) * scale);
    return [Math.max(0, Math.min(size - 1, px)), Math.max(0, Math.min(size - 1, py))];
  });

  // Helper to draw line using Bresenham algorithm
  function drawLine(p1, p2, val = 0.75, width = 2) {
    let [x0, y0] = p1;
    const [x1, y1] = p2;
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    while (true) {
      for (let ox = -Math.floor(width/2); ox <= Math.floor(width/2); ox++) {
        for (let oy = -Math.floor(width/2); oy <= Math.floor(width/2); oy++) {
          const px = x0 + ox;
          const py = y0 + oy;
          if (px >= 0 && px < size && py >= 0 && py < size) {
            img[py * size + px] = Math.max(img[py * size + px], val);
          }
        }
      }
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) { err -= dy; x0 += sx; }
      if (e2 < dx) { err += dx; y0 += sy; }
    }
  }

  // Draw connections
  for (const [s, e] of CONNECTIONS) {
    drawLine(pixelPts[s], pixelPts[e], 0.7, 2);
  }

  // Draw joints (dots)
  for (let i = 0; i < pixelPts.length; i++) {
    const [px, py] = pixelPts[i];
    const radius = [4, 8, 12, 16, 20].includes(i) ? 3 : 2; // bigger for tips
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy <= radius * radius) {
          const x = px + dx;
          const y = py + dy;
          if (x >= 0 && x < size && y >= 0 && y < size) {
            img[y * size + x] = 1.0;
          }
        }
      }
    }
  }

  return img;
}

async function trainAndSave() {
  console.log('Generating augmented ASL Alphabet landmark dataset...');
  const samplesPerClass = 75; // 75 samples per letter * 26 = 1950 samples
  const totalSamples = samplesPerClass * 26;
  const allImages = [];
  const allLabels = [];

  for (let c = 0; c < 26; c++) {
    const letter = CLASSES[c];
    const canonical = getCanonicalLandmarks(letter);

    for (let s = 0; s < samplesPerClass; s++) {
      const aug = s === 0 ? canonical : augmentLandmarks(canonical);
      const img = landmarksToImage(aug, IMG_SIZE);
      allImages.push(img);
      allLabels.push(c);
    }
  }

  // Shuffle dataset
  const indices = Array.from({ length: totalSamples }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const splitIdx = Math.floor(totalSamples * 0.8);
  const trainIndices = indices.slice(0, splitIdx);
  const valIndices = indices.slice(splitIdx);

  console.log(`Dataset size: ${totalSamples} | Train: ${trainIndices.length} | Val: ${valIndices.length}`);

  // Create Tensors
  const xTrainFlat = new Float32Array(trainIndices.length * IMG_SIZE * IMG_SIZE);
  const yTrainVals = new Int32Array(trainIndices.length);
  for (let i = 0; i < trainIndices.length; i++) {
    const idx = trainIndices[i];
    xTrainFlat.set(allImages[idx], i * IMG_SIZE * IMG_SIZE);
    yTrainVals[i] = allLabels[idx];
  }

  const xValFlat = new Float32Array(valIndices.length * IMG_SIZE * IMG_SIZE);
  const yValVals = new Int32Array(valIndices.length);
  for (let i = 0; i < valIndices.length; i++) {
    const idx = valIndices[i];
    xValFlat.set(allImages[idx], i * IMG_SIZE * IMG_SIZE);
    yValVals[i] = allLabels[idx];
  }

  const xTrain = tf.tensor4d(xTrainFlat, [trainIndices.length, IMG_SIZE, IMG_SIZE, 1]);
  const yTrain = tf.oneHot(tf.tensor1d(yTrainVals, 'int32'), 26);

  const xVal = tf.tensor4d(xValFlat, [valIndices.length, IMG_SIZE, IMG_SIZE, 1]);
  const yVal = tf.oneHot(tf.tensor1d(yValVals, 'int32'), 26);

  // Build genuine CNN model
  console.log('Constructing CNN Architecture...');
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [IMG_SIZE, IMG_SIZE, 1],
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: 26, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  model.summary();

  console.log('Training CNN on Hand-Landmark Skeleton Images...');
  const epochs = 18;
  const trainLossHistory = [];
  const valLossHistory = [];
  const trainAccHistory = [];
  const valAccHistory = [];

  await model.fit(xTrain, yTrain, {
    epochs: epochs,
    batchSize: 32,
    validationData: [xVal, yVal],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        trainLossHistory.push(Number(logs.loss.toFixed(4)));
        valLossHistory.push(Number(logs.val_loss.toFixed(4)));
        trainAccHistory.push(Number(logs.acc.toFixed(4)));
        valAccHistory.push(Number(logs.val_acc.toFixed(4)));
        console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Acc: ${(logs.acc * 100).toFixed(1)}% - Val Loss: ${logs.val_loss.toFixed(4)} - Val Acc: ${(logs.val_acc * 100).toFixed(1)}%`);
      }
    }
  });

  // Evaluate model on validation set
  console.log('Evaluating validation set...');
  const valPreds = model.predict(xVal);
  const valPredClasses = valPreds.argMax(-1).dataSync();
  const valTrueClasses = yValVals;

  // Compute confusion matrix & metrics
  const cm = Array.from({ length: 26 }, () => Array(26).fill(0));
  let correct = 0;
  for (let i = 0; i < valTrueClasses.length; i++) {
    const t = valTrueClasses[i];
    const p = valPredClasses[i];
    cm[t][p]++;
    if (t === p) correct++;
  }

  const finalValAcc = correct / valTrueClasses.length;
  console.log(`Final Evaluation Accuracy: ${(finalValAcc * 100).toFixed(2)}%`);

  // Compute per-class precision, recall, f1
  let sumP = 0, sumR = 0, sumF1 = 0;
  const perClassMetrics = {};

  for (let c = 0; c < 26; c++) {
    const tp = cm[c][c];
    let fn = 0, fp = 0;
    for (let j = 0; j < 26; j++) {
      if (j !== c) {
        fn += cm[c][j];
        fp += cm[j][c];
      }
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    sumP += prec; sumR += rec; sumF1 += f1;
    perClassMetrics[CLASSES[c]] = { precision: prec, recall: rec, f1: f1, support: tp + fn };
  }

  const metricsData = {
    status: 'trained',
    accuracy: Number(finalValAcc.toFixed(4)),
    loss: valLossHistory[valLossHistory.length - 1],
    epochs_trained: epochs,
    train_loss: trainLossHistory,
    val_loss: valLossHistory,
    train_accuracy: trainAccHistory,
    val_accuracy: valAccHistory,
    confusion_matrix: cm,
    classes: CLASSES,
    summary: {
      macro_avg_f1: Number((sumF1 / 26).toFixed(4)),
      macro_avg_precision: Number((sumP / 26).toFixed(4)),
      macro_avg_recall: Number((sumR / 26).toFixed(4))
    },
    dataset_info: {
      total_samples: totalSamples,
      train_samples: trainIndices.length,
      val_samples: valIndices.length,
      method: "MediaPipe 21-landmark normalized rasterization (64x64x1) with multi-angle spatial jitter augmentation",
      architecture: "Conv2D(32) -> Conv2D(64) -> Conv2D(128) -> Dense(128) -> Softmax(26)"
    }
  };

  // Export model to both frontend/public/model and models/web_model
  const targetDirs = [
    path.resolve('../frontend/public/model'),
    path.resolve('../models/web_model')
  ];

  for (const dir of targetDirs) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }

  // Save handler for TFJS
  const saveHandler = tf.io.withSaveHandler(async (artifacts) => {
    const weightsManifest = [{
      paths: ['./weights.bin'],
      weights: artifacts.weightSpecs
    }];

    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightsManifest: weightsManifest
    };

    const weightBuffer = Buffer.from(artifacts.weightData);

    for (const dir of targetDirs) {
      fs.writeFileSync(path.join(dir, 'model.json'), JSON.stringify(modelJson, null, 2));
      fs.writeFileSync(path.join(dir, 'weights.bin'), weightBuffer);
    }

    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        weightDataBytes: weightBuffer.length
      }
    };
  });

  await model.save(saveHandler);
  console.log('Saved model.json and weights.bin successfully!');

  // Save evaluation metrics
  fs.writeFileSync(
    path.resolve('../frontend/public/evaluation_metrics.json'),
    JSON.stringify(metricsData, null, 2)
  );
  fs.writeFileSync(
    path.resolve('../models/evaluation_metrics.json'),
    JSON.stringify(metricsData, null, 2)
  );

  console.log('Saved evaluation_metrics.json to frontend/public and models/');
  console.log('CNN Model training & export complete!');
}

trainAndSave().catch(err => {
  console.error('Error during training:', err);
  process.exit(1);
});
