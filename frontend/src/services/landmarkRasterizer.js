/**
 * Unified Landmark Rasterizer and Normalization Engine
 * Shared between training (Node.js) and real-time live inference (Browser)
 * Guarantees 100% mathematical parity in preprocessing and tensor generation.
 */

export const RASTER_SIZE = 32; // 32x32 resolution: fast, highly accurate, low latency

// Standard 21 MediaPipe hand connections
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [9, 10], [10, 11], [11, 12],          // Middle
  [13, 14], [14, 15], [15, 16],         // Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17]              // Knuckles
];

/**
 * Normalizes 21 3D landmarks:
 * 1. Centers wrist (landmark 0) to origin.
 * 2. Standardizes hand orientation so thumb is consistently positioned on the canonical side
 *    (supporting both right and left hands, and mirrored webcams).
 * 3. Scales to unit bounding span.
 */
export function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  // Extract [x, y] coordinates
  const pts = landmarks.map(lm => [
    typeof lm.x === 'number' ? lm.x : lm[0],
    typeof lm.y === 'number' ? lm.y : lm[1]
  ]);

  const [wx, wy] = pts[0]; // Wrist

  // Step 1: Center wrist at (0, 0)
  const centered = pts.map(([x, y]) => [x - wx, y - wy]);

  // Step 2: Determine hand orientation
  // In a canonical right palm view, MCP of thumb (index 1) is to the left of pinky MCP (index 17)
  // If pinky MCP is to the left of thumb MCP, flip horizontally to canonical orientation
  const isFlipped = centered[1][0] > centered[17][0];
  const oriented = centered.map(([x, y]) => [isFlipped ? -x : x, y]);

  // Step 3: Compute bounding box and max span
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < oriented.length; i++) {
    const [x, y] = oriented[i];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const spanX = Math.max(maxX - minX, 0.001);
  const spanY = Math.max(maxY - minY, 0.001);
  const maxSpan = Math.max(spanX, spanY);

  return {
    points: oriented,
    minX, maxX, minY, maxY,
    maxSpan,
    isFlipped
  };
}

/**
 * Deterministically rasterizes normalized landmarks into a Float32Array image buffer (size x size).
 * Values in range [0.0, 1.0].
 */
export function renderLandmarksToBuffer(landmarks, size = RASTER_SIZE) {
  const norm = normalizeLandmarks(landmarks);
  if (!norm) return null;

  const { points, minX, minY, maxSpan } = norm;
  const buffer = new Float32Array(size * size);

  // Keep a 3-pixel margin on all sides
  const margin = 3;
  const usable = size - margin * 2;
  const scale = usable / maxSpan;

  // Map normalized coordinates into discrete canvas pixels
  const pixelPts = points.map(([x, y]) => [
    Math.max(0, Math.min(size - 1, Math.round(margin + (x - minX) * scale))),
    Math.max(0, Math.min(size - 1, Math.round(margin + (y - minY) * scale)))
  ]);

  // Antialiased line drawing with subpixel anti-aliasing
  function drawLine(p1, p2, intensity = 0.85, width = 1.3) {
    const [x0, y0] = p1;
    const [x1, y1] = p2;
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;
    let cx = x0;
    let cy = y0;

    const halfW = 1;

    while (true) {
      for (let oy = -halfW; oy <= halfW; oy++) {
        for (let ox = -halfW; ox <= halfW; ox++) {
          const px = cx + ox;
          const py = cy + oy;
          if (px >= 0 && px < size && py >= 0 && py < size) {
            const dist = Math.sqrt(ox * ox + oy * oy);
            const falloff = Math.max(0.3, 1.0 - dist / 1.6);
            const val = intensity * falloff;
            const idx = py * size + px;
            buffer[idx] = Math.max(buffer[idx], val);
          }
        }
      }

      if (cx === x1 && cy === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) { err -= dy; cx += sx; }
      if (e2 < dx) { err += dx; cy += sy; }
    }
  }

  // Draw 21 skeletal connections
  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    drawLine(pixelPts[startIdx], pixelPts[endIdx], 0.85, 1.2);
  }

  // Draw joint landmarks (dots)
  for (let i = 0; i < pixelPts.length; i++) {
    const [px, py] = pixelPts[i];
    const isTip = [4, 8, 12, 16, 20].includes(i);
    const radius = isTip ? 2.2 : 1.5;
    const rInt = Math.ceil(radius);

    for (let dy = -rInt; dy <= rInt; dy++) {
      for (let dx = -rInt; dx <= rInt; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= radius) {
          const x = px + dx;
          const y = py + dy;
          if (x >= 0 && x < size && y >= 0 && y < size) {
            const falloff = Math.max(0.4, 1.0 - (dist / (radius + 0.5)) * 0.4);
            const idx = y * size + x;
            buffer[idx] = Math.max(buffer[idx], isTip ? 1.0 : 0.9 * falloff);
          }
        }
      }
    }
  }

  return {
    buffer,
    pixelPts,
    isFlipped: norm.isFlipped
  };
}

/**
 * Converts a Float32Array buffer into an HTML data URL (PNG) for clean UI preview
 */
export function bufferToDataUrl(buffer, size = RASTER_SIZE) {
  if (typeof document === 'undefined' || !buffer) return null;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(size, size);
  for (let i = 0; i < buffer.length; i++) {
    const v = Math.round(buffer[i] * 255);
    const idx = i * 4;
    imgData.data[idx] = v;     // R
    imgData.data[idx + 1] = v; // G
    imgData.data[idx + 2] = v; // B
    imgData.data[idx + 3] = 255; // A
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}
