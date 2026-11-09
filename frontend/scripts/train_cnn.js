/**
 * Improved ASL CNN Training Script
 * Resolves M/N/T, A/S/E, U/V, D/F distinctions, normalizes hand orientation,
 * and uses the unified landmarkRasterizer for 100% training-inference parity.
 */

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';
import { renderLandmarksToBuffer, RASTER_SIZE } from '../src/services/landmarkRasterizer.js';

const CLASSES = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)); // 'A'..'Z'

function getCanonicalLandmarks(letter) {
  const wrist = [0.5, 0.88];
  const mcpThumb = [0.38, 0.74];
  const mcpIndex = [0.38, 0.58];
  const mcpMid = [0.48, 0.54];
  const mcpRing = [0.58, 0.57];
  const mcpPinky = [0.66, 0.62];

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
        [mx - 0.01, my + 0.05],
        [mx - 0.01, my + 0.11],
        [mx + 0.01, my + 0.14]
      ];
    } else if (state === 'curled') {
      return [
        [mx - 0.02, my - 0.07],
        [mx + 0.02, my - 0.01],
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

  let thumbState = 'folded';
  let indexState = 'folded';
  let midState = 'folded';
  let ringState = 'folded';
  let pinkyState = 'folded';
  let indexAngle = 0, midAngle = 0, ringAngle = 0, pinkyAngle = 0;

  switch (letter) {
    case 'A': thumbState = 'side_up'; break;
    case 'B':
      indexState = 'up'; midState = 'up'; ringState = 'up'; pinkyState = 'up';
      thumbState = 'across_palm'; break;
    case 'C':
      indexState = 'curled'; midState = 'curled'; ringState = 'curled'; pinkyState = 'curled';
      thumbState = 'curled_c'; break;
    case 'D':
      indexState = 'up'; midState = 'curled'; ringState = 'curled'; pinkyState = 'curled';
      thumbState = 'touch_mid'; break;
    case 'E':
      indexState = 'folded'; midState = 'folded'; ringState = 'folded'; pinkyState = 'folded';
      thumbState = 'across_under'; break;
    case 'F':
      indexState = 'curled'; midState = 'up'; ringState = 'up'; pinkyState = 'up';
      midAngle = -0.06; ringAngle = 0.04; pinkyAngle = 0.14;
      thumbState = 'touch_index'; break;
    case 'G': indexState = 'horizontal'; thumbState = 'horizontal_g'; break;
    case 'H': indexState = 'horizontal'; midState = 'horizontal'; thumbState = 'horizontal_h'; break;
    case 'I': pinkyState = 'up'; thumbState = 'across_palm'; break;
    case 'J': pinkyState = 'up'; pinkyAngle = -0.22; thumbState = 'across_palm'; break;
    case 'K': indexState = 'up'; midState = 'up'; midAngle = 0.26; thumbState = 'between_k'; break;
    case 'L': indexState = 'up'; thumbState = 'out_l'; break;
    case 'M': indexState = 'folded'; midState = 'folded'; ringState = 'folded'; thumbState = 'tucked_m'; break;
    case 'N': indexState = 'folded'; midState = 'folded'; thumbState = 'tucked_n'; break;
    case 'O':
      indexState = 'curled'; midState = 'curled'; ringState = 'curled'; pinkyState = 'curled';
      thumbState = 'touch_o'; break;
    case 'P': indexState = 'down'; midState = 'down'; thumbState = 'down_p_q'; break;
    case 'Q': indexState = 'down'; thumbState = 'down_p_q'; break;
    case 'R': indexState = 'up'; indexAngle = 0.08; midState = 'up'; midAngle = -0.08; thumbState = 'across_palm'; break;
    case 'S': thumbState = 'front_fist'; break;
    case 'T': indexState = 'folded'; thumbState = 'tucked_t'; break;
    case 'U': indexState = 'up'; midState = 'up'; indexAngle = 0.01; midAngle = -0.01; thumbState = 'across_palm'; break;
    case 'V': indexState = 'up'; midState = 'up'; indexAngle = -0.16; midAngle = 0.16; thumbState = 'across_palm'; break;
    case 'W':
      indexState = 'up'; midState = 'up'; ringState = 'up';
      indexAngle = -0.20; midAngle = 0.0; ringAngle = 0.20;
      thumbState = 'across_palm'; break;
    case 'X': indexState = 'hook'; thumbState = 'across_palm'; break;
    case 'Y': pinkyState = 'up'; pinkyAngle = 0.35; thumbState = 'out_y'; break;
    case 'Z': indexState = 'up'; indexAngle = -0.20; thumbState = 'across_palm'; break;
  }

  // Explicit, mathematically distinct thumb geometry for every ASL sign
  let thumbJoints;
  if (thumbState === 'side_up') {
    thumbJoints = [[0.32, 0.65], [0.30, 0.55], [0.29, 0.46]];
  } else if (thumbState === 'front_fist') {
    thumbJoints = [[0.40, 0.65], [0.48, 0.63], [0.56, 0.63]];
  } else if (thumbState === 'across_under') {
    thumbJoints = [[0.38, 0.70], [0.44, 0.69], [0.50, 0.69]];
  } else if (thumbState === 'tucked_m') {
    thumbJoints = [[0.42, 0.66], [0.52, 0.62], [0.62, 0.60]];
  } else if (thumbState === 'tucked_n') {
    thumbJoints = [[0.40, 0.66], [0.46, 0.62], [0.53, 0.60]];
  } else if (thumbState === 'tucked_t') {
    thumbJoints = [[0.38, 0.66], [0.41, 0.58], [0.43, 0.52]];
  } else if (thumbState === 'touch_index') {
    thumbJoints = [[0.35, 0.66], [0.36, 0.55], [0.38, 0.46]];
  } else if (thumbState === 'touch_o') {
    thumbJoints = [[0.36, 0.66], [0.40, 0.56], [0.46, 0.50]];
  } else if (thumbState === 'touch_mid') {
    thumbJoints = [[0.36, 0.66], [0.42, 0.58], [0.48, 0.52]];
  } else if (thumbState === 'out_l') {
    thumbJoints = [[0.28, 0.72], [0.18, 0.72], [0.08, 0.72]];
  } else if (thumbState === 'out_y') {
    thumbJoints = [[0.26, 0.72], [0.16, 0.74], [0.08, 0.78]];
  } else if (thumbState === 'horizontal_g' || thumbState === 'horizontal_h') {
    thumbJoints = [[0.32, 0.68], [0.22, 0.66], [0.14, 0.64]];
  } else if (thumbState === 'between_k') {
    thumbJoints = [[0.36, 0.66], [0.40, 0.58], [0.43, 0.49]];
  } else if (thumbState === 'down_p_q') {
    thumbJoints = [[0.34, 0.76], [0.32, 0.86], [0.30, 0.96]];
  } else if (thumbState === 'curled_c') {
    thumbJoints = [[0.32, 0.70], [0.26, 0.60], [0.28, 0.50]];
  } else {
    thumbJoints = [[0.40, 0.68], [0.46, 0.66], [0.52, 0.65]];
  }

  const idxJoints = makeFinger(mcpIndex, indexState, indexAngle);
  const midJoints = makeFinger(mcpMid, midState, midAngle);
  const ringJoints = makeFinger(mcpRing, ringState, ringAngle);
  const pinkyJoints = makeFinger(mcpPinky, pinkyState, pinkyAngle, 0.85);

  return [
    wrist,
    mcpThumb, thumbJoints[0], thumbJoints[1], thumbJoints[2],
    mcpIndex, idxJoints[0], idxJoints[1], idxJoints[2],
    mcpMid, midJoints[0], midJoints[1], midJoints[2],
    mcpRing, ringJoints[0], ringJoints[1], ringJoints[2],
    mcpPinky, pinkyJoints[0], pinkyJoints[1], pinkyJoints[2]
  ];
}

function augmentLandmarks(pts) {
  const angle = (Math.random() - 0.5) * 0.28;
  const scale = 0.90 + Math.random() * 0.22;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const cx = pts[9][0];
  const cy = pts[9][1];

  return pts.map(([x, y]) => {
    let rx = (x - cx) * scale;
    let ry = (y - cy) * scale;
    let nx = rx * cos - ry * sin;
    let ny = rx * sin + ry * cos;
    nx += (Math.random() - 0.5) * 0.015;
    ny += (Math.random() - 0.5) * 0.015;
    return [nx + cx, ny + cy];
  });
}

async function trainAndSave() {
  console.log('=' .repeat(65));
  console.log(' ASL Alphabet Fingerspelling CNN - Improved Model Training');
  console.log('=' .repeat(65));

  const samplesPerClass = 35; // 35 * 26 = 910 balanced samples
  const totalSamples = samplesPerClass * 26;
  const allImages = [];
  const allLabels = [];

  for (let c = 0; c < 26; c++) {
    const letter = CLASSES[c];
    const canonical = getCanonicalLandmarks(letter);

    for (let s = 0; s < samplesPerClass; s++) {
      const aug = s === 0 ? canonical : augmentLandmarks(canonical);
      const rendered = renderLandmarksToBuffer(aug, RASTER_SIZE);
      allImages.push(rendered.buffer);
      allLabels.push(c);
    }
  }

  const indices = Array.from({ length: totalSamples }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const splitIdx = Math.floor(totalSamples * 0.8);
  const trainIndices = indices.slice(0, splitIdx);
  const valIndices = indices.slice(splitIdx);

  console.log(`Dataset size: ${totalSamples} | Train: ${trainIndices.length} | Val: ${valIndices.length}`);

  const xTrainFlat = new Float32Array(trainIndices.length * RASTER_SIZE * RASTER_SIZE);
  const yTrainVals = new Int32Array(trainIndices.length);
  for (let i = 0; i < trainIndices.length; i++) {
    const idx = trainIndices[i];
    xTrainFlat.set(allImages[idx], i * RASTER_SIZE * RASTER_SIZE);
    yTrainVals[i] = allLabels[idx];
  }

  const xValFlat = new Float32Array(valIndices.length * RASTER_SIZE * RASTER_SIZE);
  const yValVals = new Int32Array(valIndices.length);
  for (let i = 0; i < valIndices.length; i++) {
    const idx = valIndices[i];
    xValFlat.set(allImages[idx], i * RASTER_SIZE * RASTER_SIZE);
    yValVals[i] = allLabels[idx];
  }

  const xTrain = tf.tensor4d(xTrainFlat, [trainIndices.length, RASTER_SIZE, RASTER_SIZE, 1]);
  const yTrain = tf.oneHot(tf.tensor1d(yTrainVals, 'int32'), 26);
  const xVal = tf.tensor4d(xValFlat, [valIndices.length, RASTER_SIZE, RASTER_SIZE, 1]);
  const yVal = tf.oneHot(tf.tensor1d(yValVals, 'int32'), 26);

  console.log('Constructing Optimized CNN Architecture...');
  const model = tf.sequential();

  // Block 1
  model.add(tf.layers.conv2d({
    inputShape: [RASTER_SIZE, RASTER_SIZE, 1],
    filters: 16,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    name: 'conv1'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], name: 'pool1' }));
  model.add(tf.layers.dropout({ rate: 0.15, name: 'drop1' }));

  // Block 2
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    name: 'conv2'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], name: 'pool2' }));
  model.add(tf.layers.dropout({ rate: 0.20, name: 'drop2' }));

  // Classification Head
  model.add(tf.layers.flatten({ name: 'flatten' }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu', name: 'dense1' }));
  model.add(tf.layers.dropout({ rate: 0.25, name: 'drop_dense' }));
  model.add(tf.layers.dense({ units: 26, activation: 'softmax', name: 'output_softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.002),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  model.summary();

  const epochs = 12;
  const trainLossHistory = [];
  const valLossHistory = [];
  const trainAccHistory = [];
  const valAccHistory = [];

  console.log(`Training for ${epochs} epochs...`);
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
        console.log(`Epoch ${(epoch + 1).toString().padStart(2, ' ')}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Acc: ${(logs.acc * 100).toFixed(1)}% - Val Loss: ${logs.val_loss.toFixed(4)} - Val Acc: ${(logs.val_acc * 100).toFixed(1)}%`);
      }
    }
  });

  console.log('\nEvaluating validation set...');
  const valPreds = model.predict(xVal);
  const valPredClasses = valPreds.argMax(-1).dataSync();
  const valTrueClasses = yValVals;

  const cm = Array.from({ length: 26 }, () => Array(26).fill(0));
  let correct = 0;
  for (let i = 0; i < valTrueClasses.length; i++) {
    const t = valTrueClasses[i];
    const p = valPredClasses[i];
    cm[t][p]++;
    if (t === p) correct++;
  }

  const finalValAcc = correct / valTrueClasses.length;
  console.log(`Final Evaluation Accuracy: ${(finalValAcc * 100).toFixed(2)}% (${correct}/${valTrueClasses.length})`);

  let sumP = 0, sumR = 0, sumF1 = 0;
  const perClass = {};
  const misclassifiedPairs = [];

  for (let c = 0; c < 26; c++) {
    const tp = cm[c][c];
    let fn = 0, fp = 0;
    for (let j = 0; j < 26; j++) {
      if (j !== c) {
        fn += cm[c][j];
        fp += cm[j][c];
        if (cm[c][j] > 0) {
          misclassifiedPairs.push({ trueClass: CLASSES[c], predClass: CLASSES[j], count: cm[c][j] });
        }
      }
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    sumP += prec; sumR += rec; sumF1 += f1;
    perClass[CLASSES[c]] = {
      precision: Number(prec.toFixed(4)),
      recall: Number(rec.toFixed(4)),
      f1: Number(f1.toFixed(4)),
      support: tp + fn,
      accuracy: Number((tp / Math.max(1, tp + fn)).toFixed(4))
    };
  }

  misclassifiedPairs.sort((a, b) => b.count - a.count);

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
    per_class_metrics: perClass,
    top_misclassified_pairs: misclassifiedPairs.slice(0, 5),
    summary: {
      macro_avg_f1: Number((sumF1 / 26).toFixed(4)),
      macro_avg_precision: Number((sumP / 26).toFixed(4)),
      macro_avg_recall: Number((sumR / 26).toFixed(4))
    },
    dataset_info: {
      total_samples: totalSamples,
      train_samples: trainIndices.length,
      val_samples: valIndices.length,
      method: "Unified Antialiased 32x32x1 Landmark Skeleton Tensor with Orientation Normalization",
      architecture: "Conv2D(16) -> MaxPool -> Dropout(0.15) -> Conv2D(32) -> MaxPool -> Dropout(0.20) -> Dense(64) -> Softmax(26)"
    }
  };

  const targetDirs = [
    path.resolve('public/model'),
    path.resolve('../models/web_model')
  ];

  for (const dir of targetDirs) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }

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
  console.log('[SUCCESS] Successfully saved model.json and weights.bin!');

  fs.writeFileSync(path.resolve('public/evaluation_metrics.json'), JSON.stringify(metricsData, null, 2));
  fs.writeFileSync(path.resolve('../models/evaluation_metrics.json'), JSON.stringify(metricsData, null, 2));

  console.log('[SUCCESS] Saved evaluation_metrics.json to public/ and models/!');
}

trainAndSave().catch(err => {
  console.error('Training error:', err);
  process.exit(1);
});
