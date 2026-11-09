import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import {
  Camera, CameraOff, Play, Pause, RotateCcw, CheckCircle2,
  AlertTriangle, Download, Upload, BarChart3, ChevronRight,
  ChevronLeft, Sparkles, Layers, RefreshCw, Award, Info, Trash2
} from 'lucide-react';
import { getHandLandmarker, drawHandLandmarks } from '../services/handDetector';
import { predictASLLetter, getModelStatus } from '../services/cnnClassifier';
import { ASL_ALPHABET } from '../data/aslData';

const CLASSES = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const STORAGE_KEY = 'asl_real_world_evaluation_v1';
const VALIDATION_BENCHMARK_ACCURACY = 98.90; // Validation accuracy of trained CNN

export default function WebcamEvaluation() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameId = useRef(null);
  const streamRef = useRef(null);
  const burstTimerRef = useRef(null);

  // Camera and detection state
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStatus, setCameraStatus] = useState('Camera Off');
  const [errorMessage, setErrorMessage] = useState(null);
  const [handDetected, setHandDetected] = useState(false);
  const [livePrediction, setLivePrediction] = useState({ letter: '-', confidence: 0 });

  // Evaluation setup
  const [targetLetter, setTargetLetter] = useState('A');
  const [targetSamplesPerLetter, setTargetSamplesPerLetter] = useState(30);
  const [isRecordingBurst, setIsRecordingBurst] = useState(false);
  const [burstCount, setBurstCount] = useState(0);

  // Real webcam collected samples store: { 'A': [ { target, predicted, confidence, latencyMs, timestamp }, ... ], 'B': [...] }
  const [samplesByLetter, setSamplesByLetter] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (e) {
      console.warn('Could not parse saved evaluation session:', e);
    }
    const init = {};
    CLASSES.forEach(l => { init[l] = []; });
    return init;
  });

  const [selectedConfusionCell, setSelectedConfusionCell] = useState(null);

  // Synchronize samples with localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(samplesByLetter));
    } catch (e) {
      console.warn('LocalStorage save failed:', e);
    }
  }, [samplesByLetter]);

  // Target letter reference info
  const targetInfo = useMemo(() => {
    return ASL_ALPHABET.find(item => item.letter === targetLetter) || {
      letter: targetLetter,
      name: `Letter ${targetLetter}`,
      description: 'Perform the ASL hand posture for this letter.',
      confusions: 'Keep hand steady facing the webcam.'
    };
  }, [targetLetter]);

  // Camera controls
  const startCamera = async () => {
    setErrorMessage(null);
    setCameraStatus('Connecting');

    try {
      await getHandLandmarker();

      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setCameraActive(true);
          setCameraStatus('Live');
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setCameraStatus('Error');
      setErrorMessage(`Failed to open camera: ${err.message || 'Permission denied'}`);
      setCameraActive(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (animFrameId.current) {
      cancelAnimationFrame(animFrameId.current);
      animFrameId.current = null;
    }
    if (burstTimerRef.current) {
      clearInterval(burstTimerRef.current);
      burstTimerRef.current = null;
    }
    setIsRecordingBurst(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setCameraActive(false);
    setCameraStatus('Camera Off');
    setHandDetected(false);
    setLivePrediction({ letter: '-', confidence: 0 });
  }, []);

  // Frame processing loop
  const lastLandmarksRef = useRef(null);
  useEffect(() => {
    let isProcessing = false;

    const processLoop = async () => {
      if (!cameraActive || !videoRef.current || !canvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.readyState >= 2 && !video.paused && !video.ended) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!isProcessing) {
          isProcessing = true;
          try {
            const landmarker = await getHandLandmarker();
            const results = landmarker.detectForVideo(video, performance.now());

            if (results && results.landmarks && results.landmarks.length > 0) {
              const currentLandmarks = results.landmarks[0];
              lastLandmarksRef.current = currentLandmarks;
              setHandDetected(true);

              drawHandLandmarks(ctx, currentLandmarks, canvas.width, canvas.height);

              const pred = await predictASLLetter(currentLandmarks);
              if (pred) {
                setLivePrediction({
                  letter: pred.rawLetter || pred.letter,
                  confidence: pred.confidence,
                  latencyMs: pred.latencyMs
                });
              }
            } else {
              setHandDetected(false);
              lastLandmarksRef.current = null;
            }
          } catch (err) {
            console.error('Webcam eval error:', err);
          } finally {
            isProcessing = false;
          }
        }
      }

      animFrameId.current = requestAnimationFrame(processLoop);
    };

    if (cameraActive) {
      animFrameId.current = requestAnimationFrame(processLoop);
    }

    return () => {
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
    };
  }, [cameraActive]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Capture single sample
  const captureSingleSample = useCallback(async () => {
    if (!lastLandmarksRef.current) return false;

    const pred = await predictASLLetter(lastLandmarksRef.current);
    if (!pred) return false;

    const sample = {
      target: targetLetter,
      predicted: pred.rawLetter || pred.letter,
      confidence: pred.confidence,
      latencyMs: pred.latencyMs,
      timestamp: Date.now()
    };

    setSamplesByLetter(prev => {
      const list = prev[targetLetter] ? [...prev[targetLetter]] : [];
      return {
        ...prev,
        [targetLetter]: [...list, sample]
      };
    });

    return true;
  }, [targetLetter]);

  // Burst capture mode (records up to targetSamplesPerLetter at ~100ms interval)
  const startBurstCapture = () => {
    if (isRecordingBurst || !cameraActive) return;

    setIsRecordingBurst(true);
    setBurstCount(0);

    let count = 0;
    const currentListLen = samplesByLetter[targetLetter]?.length || 0;
    const remainingNeeded = Math.max(0, targetSamplesPerLetter - currentListLen);
    const toCapture = remainingNeeded > 0 ? remainingNeeded : targetSamplesPerLetter;

    burstTimerRef.current = setInterval(async () => {
      if (!lastLandmarksRef.current) return;

      const success = await captureSingleSample();
      if (success) {
        count++;
        setBurstCount(count);
        if (count >= toCapture) {
          clearInterval(burstTimerRef.current);
          burstTimerRef.current = null;
          setIsRecordingBurst(false);
        }
      }
    }, 120);
  };

  const stopBurstCapture = () => {
    if (burstTimerRef.current) {
      clearInterval(burstTimerRef.current);
      burstTimerRef.current = null;
    }
    setIsRecordingBurst(false);
  };

  // Keyboard shortcut: Spacebar captures sample
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && cameraActive && handDetected && !isRecordingBurst) {
        e.preventDefault();
        captureSingleSample();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [cameraActive, handDetected, isRecordingBurst, captureSingleSample]);

  // Navigate letters
  const handleNextLetter = () => {
    const idx = CLASSES.indexOf(targetLetter);
    if (idx < CLASSES.length - 1) setTargetLetter(CLASSES[idx + 1]);
  };

  const handlePrevLetter = () => {
    const idx = CLASSES.indexOf(targetLetter);
    if (idx > 0) setTargetLetter(CLASSES[idx - 1]);
  };

  const handleClearCurrentLetter = () => {
    setSamplesByLetter(prev => ({
      ...prev,
      [targetLetter]: []
    }));
  };

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to clear all real-world webcam evaluation samples?')) {
      const reset = {};
      CLASSES.forEach(l => { reset[l] = []; });
      setSamplesByLetter(reset);
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  // Real-world metrics calculation
  const metrics = useMemo(() => {
    let totalSamples = 0;
    let totalCorrect = 0;
    const matrix = Array.from({ length: 26 }, () => Array(26).fill(0));
    const perClass = {};
    const confusionPairs = {};

    CLASSES.forEach((c, i) => {
      perClass[c] = { support: 0, correct: 0, accuracy: 0, precDenom: 0, precision: 0, recall: 0, f1: 0 };
    });

    // Populate matrix & totals from samples
    CLASSES.forEach((trueClass, trueIdx) => {
      const samples = samplesByLetter[trueClass] || [];
      samples.forEach(s => {
        totalSamples++;
        const predIdx = CLASSES.indexOf(s.predicted);
        if (predIdx >= 0) {
          matrix[trueIdx][predIdx]++;
          perClass[trueClass].support++;
          perClass[s.predicted].precDenom++;

          if (s.predicted === trueClass) {
            totalCorrect++;
            perClass[trueClass].correct++;
          } else {
            const pairKey = `${trueClass}->${s.predicted}`;
            confusionPairs[pairKey] = (confusionPairs[pairKey] || 0) + 1;
          }
        }
      });
    });

    // Per-class and macro metrics
    let sumPrecision = 0;
    let sumRecall = 0;
    let sumF1 = 0;
    let classesWithSamples = 0;

    CLASSES.forEach(c => {
      const row = perClass[c];
      row.accuracy = row.support > 0 ? (row.correct / row.support) : 0;
      row.recall = row.support > 0 ? (row.correct / row.support) : 0;
      row.precision = row.precDenom > 0 ? (row.correct / row.precDenom) : 0;
      row.f1 = (row.precision + row.recall) > 0
        ? (2 * row.precision * row.recall) / (row.precision + row.recall)
        : 0;

      if (row.support > 0) {
        sumPrecision += row.precision;
        sumRecall += row.recall;
        sumF1 += row.f1;
        classesWithSamples++;
      }
    });

    const realWorldAccuracy = totalSamples > 0 ? (totalCorrect / totalSamples) * 100 : 0;
    const macroPrecision = classesWithSamples > 0 ? sumPrecision / classesWithSamples : 0;
    const macroRecall = classesWithSamples > 0 ? sumRecall / classesWithSamples : 0;
    const macroF1 = classesWithSamples > 0 ? sumF1 / classesWithSamples : 0;
    const delta = realWorldAccuracy - VALIDATION_BENCHMARK_ACCURACY;

    // Top confused pairs ranked
    const topConfusions = Object.entries(confusionPairs)
      .map(([pair, count]) => {
        const [actual, pred] = pair.split('->');
        const actualSupport = perClass[actual]?.support || 1;
        return {
          actual,
          predicted: pred,
          count,
          ratePct: Math.round((count / actualSupport) * 1000) / 10
        };
      })
      .sort((a, b) => b.count - a.count);

    return {
      totalSamples,
      totalCorrect,
      classesWithSamples,
      realWorldAccuracy: Math.round(realWorldAccuracy * 10) / 10,
      validationBenchmark: VALIDATION_BENCHMARK_ACCURACY,
      delta: Math.round(delta * 10) / 10,
      macroPrecision: Math.round(macroPrecision * 1000) / 1000,
      macroRecall: Math.round(macroRecall * 1000) / 1000,
      macroF1: Math.round(macroF1 * 1000) / 1000,
      matrix,
      perClass,
      topConfusions
    };
  }, [samplesByLetter]);

  // Export results to JSON
  const exportJsonReport = () => {
    const reportData = {
      title: 'ASL Alphabet Fingerspelling CNN - Real-World Webcam Evaluation Report',
      evaluated_at: new Date().toISOString(),
      evaluation_methodology: 'Live MediaPipe Hand Landmark Extraction + TensorFlow.js CNN Client-Side Inference',
      target_samples_per_letter: targetSamplesPerLetter,
      total_real_samples_collected: metrics.totalSamples,
      total_correct_predictions: metrics.totalCorrect,
      classes_evaluated: `${metrics.classesWithSamples} / 26`,
      metrics: {
        real_world_accuracy_pct: metrics.realWorldAccuracy,
        validation_accuracy_baseline_pct: VALIDATION_BENCHMARK_ACCURACY,
        accuracy_delta_pct: metrics.delta,
        macro_f1_score: metrics.macroF1,
        macro_precision: metrics.macroPrecision,
        macro_recall: metrics.macroRecall
      },
      most_frequently_confused_pairs: metrics.topConfusions,
      per_class_summary: metrics.perClass,
      confusion_matrix: metrics.matrix,
      classes: CLASSES,
      raw_samples_by_letter: samplesByLetter
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `asl_real_world_evaluation_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Import existing JSON evaluation session
  const fileInputRef = useRef(null);
  const handleImportJson = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target.result);
        if (data.raw_samples_by_letter) {
          setSamplesByLetter(data.raw_samples_by_letter);
        } else if (typeof data === 'object') {
          // If pure dictionary of arrays
          setSamplesByLetter(data);
        }
        alert('Evaluation session imported successfully.');
      } catch (err) {
        alert('Failed to parse evaluation JSON file: ' + err.message);
      }
    };
    reader.readAsText(file);
    e.target.value = null;
  };

  const currentLetterSamples = samplesByLetter[targetLetter] || [];
  const currentLetterCorrect = currentLetterSamples.filter(s => s.predicted === targetLetter).length;
  const currentLetterAcc = currentLetterSamples.length > 0
    ? Math.round((currentLetterCorrect / currentLetterSamples.length) * 1000) / 10
    : 0;

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Award color="#8b5cf6" size={26} /> Real-World Webcam Evaluation Studio
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginTop: '0.25rem' }}>
            Empirically evaluate the CNN on actual webcam gestures across ASL letters A&ndash;Z to measure real-world generalization vs. the 98.90% validation benchmark.
          </p>
        </div>

        {/* Global Controls & Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div className={`badge-status ${cameraActive ? 'active' : 'inactive'}`}>
            <span className={`status-dot ${cameraActive ? 'pulse' : ''}`} />
            <span>Camera: {cameraStatus}</span>
          </div>

          <button className="btn-secondary" onClick={exportJsonReport} disabled={metrics.totalSamples === 0} style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}>
            <Download size={14} /> Export Report JSON
          </button>

          <button className="btn-secondary" onClick={() => fileInputRef.current?.click()} style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}>
            <Upload size={14} /> Load Session
          </button>
          <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }} onChange={handleImportJson} />

          <button className="btn-secondary" onClick={handleClearAll} style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem', color: '#f87171' }}>
            <Trash2 size={14} /> Reset Data
          </button>
        </div>
      </div>

      {/* Step-by-Step Instructions Banner */}
      <div className="glass-card" style={{
        padding: '1.25rem 1.5rem',
        background: 'linear-gradient(135deg, rgba(30, 41, 75, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%)',
        borderLeft: '4px solid #8b5cf6'
      }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Info size={16} /> How Real-World Evaluation Works
        </div>
        <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
          1. <strong>Start your webcam</strong> and position your hand in frame with good lighting.<br />
          2. Select an alphabet letter (A&ndash;Z) and pose the hand posture shown in the guidance card.<br />
          3. Click <strong>&ldquo;Start Burst Capture (30 Frames)&rdquo;</strong> or tap the <strong>Spacebar</strong> to collect actual webcam landmark samples.<br />
          4. The app captures real MediaPipe hand skeletons, runs the live CNN inference, logs the prediction, and generates the real-world confusion matrix and accuracy metrics.
        </div>
      </div>

      {/* Letter Selector Chips Bar */}
      <div className="glass-card" style={{ padding: '1rem', overflowX: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Alphabet Progress Tracker ({metrics.classesWithSamples} of 26 Tested &bull; {metrics.totalSamples} Total Frames)
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Target: {targetSamplesPerLetter} samples/letter
          </span>
        </div>

        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
          {CLASSES.map((letter) => {
            const count = samplesByLetter[letter]?.length || 0;
            const isDone = count >= targetSamplesPerLetter;
            const isPartial = count > 0 && !isDone;
            const isCurrent = letter === targetLetter;

            return (
              <button
                key={letter}
                onClick={() => setTargetLetter(letter)}
                style={{
                  minWidth: '42px',
                  padding: '0.4rem 0.6rem',
                  borderRadius: '8px',
                  border: isCurrent
                    ? '2px solid #8b5cf6'
                    : isDone
                      ? '1px solid rgba(16, 185, 129, 0.4)'
                      : isPartial
                        ? '1px solid rgba(245, 158, 11, 0.4)'
                        : '1px solid rgba(255, 255, 255, 0.08)',
                  background: isCurrent
                    ? 'rgba(139, 92, 246, 0.3)'
                    : isDone
                      ? 'rgba(16, 185, 129, 0.12)'
                      : isPartial
                        ? 'rgba(245, 158, 11, 0.12)'
                        : 'rgba(255, 255, 255, 0.03)',
                  color: isCurrent ? '#fff' : isDone ? '#34d399' : isPartial ? '#fbbf24' : 'var(--text-dim)',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '0.15rem',
                  transition: 'all 0.15s ease'
                }}
              >
                <span style={{ fontWeight: 800, fontSize: '0.95rem', fontFamily: 'JetBrains Mono' }}>{letter}</span>
                <span style={{ fontSize: '0.62rem', opacity: 0.8 }}>{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Testing Studio (2 Columns) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(320px, 1.35fr) minmax(320px, 1fr)',
        gap: '1.5rem',
        alignItems: 'start'
      }}>
        {/* Left Column: Webcam Feed & Capture Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Video Viewport */}
          <div className="glass-card" style={{
            padding: '0.75rem',
            position: 'relative',
            borderRadius: '20px',
            overflow: 'hidden',
            aspectRatio: '4 / 3',
            background: '#040711',
            border: cameraActive ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid var(--border-glass)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <video
              ref={videoRef}
              playsInline
              muted
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                transform: 'scaleX(-1)',
                borderRadius: '14px',
                display: cameraActive ? 'block' : 'none'
              }}
            />

            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: '0.75rem',
                left: '0.75rem',
                width: 'calc(100% - 1.5rem)',
                height: 'calc(100% - 1.5rem)',
                objectFit: 'cover',
                transform: 'scaleX(-1)',
                pointerEvents: 'none',
                display: cameraActive ? 'block' : 'none'
              }}
            />

            {/* Offline Viewport Placeholder */}
            {!cameraActive && (
              <div style={{ textAlign: 'center', padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', color: 'var(--text-muted)' }}>
                <CameraOff size={36} color="var(--text-dim)" />
                <div>
                  <div style={{ fontWeight: 600, color: '#fff', fontSize: '1.05rem' }}>Camera Off</div>
                  <div style={{ fontSize: '0.85rem', marginTop: '0.2rem' }}>Start camera to begin real-world evaluation</div>
                </div>
                <button className="btn-primary" onClick={startCamera}>
                  <Camera size={18} />
                  <span>Start Camera</span>
                </button>
              </div>
            )}

            {/* In-viewport Target Overlay Badge */}
            {cameraActive && (
              <div style={{
                position: 'absolute',
                top: '1.25rem',
                left: '1.25rem',
                background: 'rgba(0, 0, 0, 0.7)',
                backdropFilter: 'blur(8px)',
                padding: '0.4rem 0.8rem',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                fontSize: '0.82rem',
                fontWeight: 700
              }}>
                <span style={{ color: '#a78bfa' }}>Target:</span>
                <span style={{ fontSize: '1.2rem', color: '#fff', fontFamily: 'JetBrains Mono' }}>{targetLetter}</span>
                <span style={{ color: 'var(--text-dim)' }}>&bull;</span>
                <span style={{ color: handDetected ? '#34d399' : '#f87171' }}>
                  {handDetected ? 'Hand Tracked' : 'Searching Hand...'}
                </span>
              </div>
            )}

            {/* Burst Recording Banner */}
            {isRecordingBurst && (
              <div style={{
                position: 'absolute',
                bottom: '1.25rem',
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(239, 68, 68, 0.9)',
                color: '#fff',
                padding: '0.45rem 1.25rem',
                borderRadius: '999px',
                fontWeight: 700,
                fontSize: '0.88rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                boxShadow: '0 0 20px rgba(239, 68, 68, 0.5)'
              }}>
                <span className="status-dot pulse" style={{ background: '#fff' }} />
                <span>Recording Burst: {burstCount} / {targetSamplesPerLetter}</span>
              </div>
            )}
          </div>

          {/* Capture Toolbar */}
          <div className="glass-card" style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
            padding: '1rem 1.25rem'
          }}>
            <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
              {!cameraActive ? (
                <button className="btn-primary" onClick={startCamera}>
                  <Camera size={18} />
                  <span>Start Camera</span>
                </button>
              ) : (
                <button className="btn-danger" onClick={stopCamera}>
                  <CameraOff size={18} />
                  <span>Stop Camera</span>
                </button>
              )}

              {/* Burst Capture Button */}
              {!isRecordingBurst ? (
                <button
                  className="btn-primary"
                  onClick={startBurstCapture}
                  disabled={!cameraActive || !handDetected}
                  style={{
                    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                    opacity: !cameraActive || !handDetected ? 0.5 : 1
                  }}
                >
                  <Play size={16} />
                  <span>Start Burst ({targetSamplesPerLetter} Frames)</span>
                </button>
              ) : (
                <button className="btn-danger" onClick={stopBurstCapture}>
                  <Pause size={16} />
                  <span>Stop Burst</span>
                </button>
              )}

              {/* Single Sample Capture */}
              <button
                className="btn-secondary"
                onClick={captureSingleSample}
                disabled={!cameraActive || !handDetected || isRecordingBurst}
                style={{ opacity: !cameraActive || !handDetected ? 0.5 : 1 }}
                title="Or press Spacebar"
              >
                <span>Capture 1 Frame [Space]</span>
              </button>
            </div>

            {/* Letter Navigation buttons */}
            <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
              <button className="btn-secondary" onClick={handlePrevLetter} disabled={targetLetter === 'A'} style={{ padding: '0.55rem' }}>
                <ChevronLeft size={18} />
              </button>
              <button className="btn-secondary" onClick={handleNextLetter} disabled={targetLetter === 'Z'} style={{ padding: '0.55rem' }}>
                <ChevronRight size={18} />
              </button>
              <button className="btn-secondary" onClick={handleClearCurrentLetter} title="Clear samples for this letter" style={{ padding: '0.55rem', color: '#f87171' }}>
                <RotateCcw size={16} />
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Active Target Letter Details & Live Feedback */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Target Guidance Card */}
          <div className="glass-card" style={{
            padding: '1.5rem',
            background: 'linear-gradient(135deg, rgba(25, 33, 60, 0.7) 0%, rgba(13, 20, 38, 0.8) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.35)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <span style={{ fontSize: '0.76rem', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase' }}>
                  Current Target Letter
                </span>
                <h2 style={{ fontSize: '1.4rem', fontWeight: 800 }}>{targetInfo.name}</h2>
              </div>

              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '16px',
                background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '2.5rem',
                fontWeight: 900,
                color: '#fff',
                fontFamily: 'JetBrains Mono, monospace',
                boxShadow: '0 4px 15px rgba(139, 92, 246, 0.4)'
              }}>
                {targetLetter}
              </div>
            </div>

            <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: '1.5' }}>
              <strong>Posture:</strong> {targetInfo.description}
            </div>

            <div style={{
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.25)',
              padding: '0.75rem 1rem',
              borderRadius: '8px',
              fontSize: '0.8rem',
              color: '#fef3c7',
              marginBottom: '1.25rem'
            }}>
              <strong>Watch Out:</strong> {targetInfo.confusions}
            </div>

            {/* Current Letter Progress Meter */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.4rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Samples Collected</span>
                <span style={{ fontWeight: 700, color: currentLetterSamples.length >= targetSamplesPerLetter ? '#34d399' : '#fbbf24' }}>
                  {currentLetterSamples.length} / {targetSamplesPerLetter} frames
                </span>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '999px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(100, (currentLetterSamples.length / targetSamplesPerLetter) * 100)}%`,
                  background: currentLetterSamples.length >= targetSamplesPerLetter ? '#10b981' : '#8b5cf6',
                  borderRadius: '999px',
                  transition: 'width 0.2s ease'
                }} />
              </div>
            </div>
          </div>

          {/* Live CNN Prediction Feedback vs Target */}
          <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
                Target (Ground Truth)
              </div>
              <div style={{ fontSize: '2.5rem', fontWeight: 900, fontFamily: 'JetBrains Mono', color: '#fff' }}>
                {targetLetter}
              </div>
            </div>

            <div style={{ fontSize: '1.5rem', color: 'var(--text-dim)' }}>vs</div>

            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
                Live Model Prediction
              </div>
              <div style={{
                fontSize: '2.5rem',
                fontWeight: 900,
                fontFamily: 'JetBrains Mono',
                color: livePrediction.letter === targetLetter ? '#34d399' : livePrediction.letter === '-' ? 'var(--text-dim)' : '#f87171'
              }}>
                {livePrediction.letter}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {livePrediction.confidence}% conf
              </div>
            </div>

            <div style={{ textAlign: 'center', borderLeft: '1px solid rgba(255, 255, 255, 0.08)', paddingLeft: '1.25rem' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700 }}>
                Real Accuracy ({targetLetter})
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'JetBrains Mono', color: currentLetterAcc >= 90 ? '#34d399' : currentLetterAcc >= 70 ? '#fbbf24' : '#f87171' }}>
                {currentLetterAcc}%
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                {currentLetterCorrect} / {currentLetterSamples.length} correct
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Real-World Results Section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.45rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 color="#8b5cf6" size={22} /> Real-World Benchmark Results &amp; Comparison
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Empirical evaluation across all captured webcam samples compared against the offline 98.90% validation benchmark.
          </p>
        </div>

        {/* 4 KPI Comparison Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1.25rem'
        }}>
          {/* Card 1: Real-World Accuracy */}
          <div className="glass-card" style={{ padding: '1.25rem', borderLeft: '4px solid #10b981' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Real-World Webcam Accuracy
            </span>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.25rem' }}>
              <span style={{ fontSize: '2.2rem', fontWeight: 800, color: '#fff', fontFamily: 'JetBrains Mono' }}>
                {metrics.totalSamples > 0 ? `${metrics.realWorldAccuracy}%` : 'N/A'}
              </span>
              {metrics.totalSamples > 0 && (
                <span style={{
                  fontSize: '0.82rem',
                  fontWeight: 700,
                  color: metrics.delta >= 0 ? '#34d399' : '#fbbf24'
                }}>
                  {metrics.delta >= 0 ? `+${metrics.delta}%` : `${metrics.delta}%`} vs val
                </span>
              )}
            </div>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginTop: '0.2rem', display: 'block' }}>
              {metrics.totalCorrect} / {metrics.totalSamples} frames across {metrics.classesWithSamples} letters
            </span>
          </div>

          {/* Card 2: Offline Validation Baseline */}
          <div className="glass-card" style={{ padding: '1.25rem', borderLeft: '4px solid #8b5cf6' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Offline Validation Benchmark
            </span>
            <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#fff', fontFamily: 'JetBrains Mono', marginTop: '0.25rem' }}>
              {VALIDATION_BENCHMARK_ACCURACY}%
            </div>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginTop: '0.2rem', display: 'block' }}>
              Trained 137k-param CNN on canonical gestures
            </span>
          </div>

          {/* Card 3: Macro F1-Score */}
          <div className="glass-card" style={{ padding: '1.25rem', borderLeft: '4px solid #06b6d4' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Real-World Macro F1-Score
            </span>
            <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#fff', fontFamily: 'JetBrains Mono', marginTop: '0.25rem' }}>
              {metrics.totalSamples > 0 ? metrics.macroF1 : 'N/A'}
            </div>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginTop: '0.2rem', display: 'block' }}>
              Precision: {metrics.macroPrecision} &bull; Recall: {metrics.macroRecall}
            </span>
          </div>

          {/* Card 4: Total Frames Evaluated */}
          <div className="glass-card" style={{ padding: '1.25rem', borderLeft: '4px solid #f59e0b' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Real Webcam Samples Evaluated
            </span>
            <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#fff', fontFamily: 'JetBrains Mono', marginTop: '0.25rem' }}>
              {metrics.totalSamples}
            </div>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginTop: '0.2rem', display: 'block' }}>
              Goal: {targetSamplesPerLetter * 26} total (30 per letter)
            </span>
          </div>
        </div>

        {/* Confused Pairs & Real-World Confusion Matrix */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '1.5rem',
          alignItems: 'start'
        }}>
          {/* Top Confused Letter Pairs Table */}
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={18} color="#f59e0b" /> Most Frequently Confused Letters
            </h3>
            <p style={{ color: 'var(--text-dim)', fontSize: '0.8rem', marginBottom: '1rem' }}>
              Real-world misclassifications observed during live webcam execution
            </p>

            {metrics.topConfusions.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)', fontSize: '0.88rem' }}>
                {metrics.totalSamples === 0
                  ? 'No webcam samples collected yet.'
                  : 'Perfect classification! No misclassifications observed.'}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {metrics.topConfusions.slice(0, 7).map((item, idx) => (
                  <div key={idx} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.6rem 0.85rem',
                    borderRadius: '8px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid rgba(255, 255, 255, 0.06)'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '6px',
                        background: 'rgba(239, 68, 68, 0.15)',
                        color: '#f87171',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 800,
                        fontFamily: 'JetBrains Mono'
                      }}>
                        {item.actual}
                      </span>
                      <span style={{ color: 'var(--text-dim)', fontSize: '0.82rem' }}>&rarr; predicted as</span>
                      <span style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '6px',
                        background: 'rgba(245, 158, 11, 0.15)',
                        color: '#fbbf24',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 800,
                        fontFamily: 'JetBrains Mono'
                      }}>
                        {item.predicted}
                      </span>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontWeight: 700, color: '#f87171', fontSize: '0.88rem' }}>
                        {item.count} samples
                      </span>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'block' }}>
                        {item.ratePct}% of {item.actual}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Per-Class Summary Table */}
          <div className="glass-card" style={{ padding: '1.5rem', maxHeight: '380px', overflowY: 'auto' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem' }}>
              Per-Class Real-World Accuracy
            </h3>
            <p style={{ color: 'var(--text-dim)', fontSize: '0.8rem', marginBottom: '1rem' }}>
              Detailed breakdown for each ASL letter
            </p>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-dim)', textAlign: 'left' }}>
                  <th style={{ padding: '0.5rem' }}>Letter</th>
                  <th style={{ padding: '0.5rem' }}>Samples</th>
                  <th style={{ padding: '0.5rem' }}>Correct</th>
                  <th style={{ padding: '0.5rem' }}>Accuracy</th>
                  <th style={{ padding: '0.5rem' }}>F1</th>
                </tr>
              </thead>
              <tbody>
                {CLASSES.map((letter) => {
                  const data = metrics.perClass[letter];
                  const hasSamples = data.support > 0;
                  const acc = Math.round(data.accuracy * 100);

                  return (
                    <tr key={letter} style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      background: letter === targetLetter ? 'rgba(139, 92, 246, 0.12)' : 'transparent'
                    }}>
                      <td style={{ padding: '0.45rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: '#fff' }}>
                        {letter}
                      </td>
                      <td style={{ padding: '0.45rem', color: 'var(--text-muted)' }}>
                        {data.support}
                      </td>
                      <td style={{ padding: '0.45rem', color: 'var(--text-muted)' }}>
                        {data.correct}
                      </td>
                      <td style={{ padding: '0.45rem' }}>
                        {hasSamples ? (
                          <span style={{
                            color: acc >= 90 ? '#34d399' : acc >= 70 ? '#fbbf24' : '#f87171',
                            fontWeight: 700
                          }}>
                            {acc}%
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)' }}>-</span>
                        )}
                      </td>
                      <td style={{ padding: '0.45rem', color: 'var(--text-muted)' }}>
                        {hasSamples ? data.f1.toFixed(2) : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Full 26x26 Real-World Confusion Matrix */}
        <div className="glass-card" style={{ padding: '1.5rem', overflowX: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Real-World 26&times;26 Confusion Matrix</h3>
              <p style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                Rows: Ground Truth (Actual Performed Letter) &bull; Columns: CNN Prediction Output
              </p>
            </div>
            {selectedConfusionCell && (
              <div style={{
                background: 'rgba(139, 92, 246, 0.2)',
                border: '1px solid rgba(139, 92, 246, 0.4)',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                fontSize: '0.8rem',
                color: '#fff'
              }}>
                Target <strong>{selectedConfusionCell.actual}</strong> &rarr; Predicted <strong>{selectedConfusionCell.pred}</strong>: <strong>{selectedConfusionCell.count}</strong> samples
              </div>
            )}
          </div>

          <div style={{ minWidth: '720px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '32px repeat(26, 1fr)', gap: '2px', textAlign: 'center', fontSize: '0.68rem', fontWeight: 700 }}>
              {/* Header row */}
              <div style={{ color: 'var(--text-dim)' }}>T\P</div>
              {CLASSES.map(c => (
                <div key={c} style={{ color: 'var(--text-dim)', fontFamily: 'JetBrains Mono' }}>{c}</div>
              ))}

              {/* Rows */}
              {CLASSES.map((actualClass, rIdx) => (
                <React.Fragment key={actualClass}>
                  <div style={{ color: '#a78bfa', fontFamily: 'JetBrains Mono', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {actualClass}
                  </div>
                  {CLASSES.map((predClass, cIdx) => {
                    const count = metrics.matrix[rIdx][cIdx];
                    const isDiag = rIdx === cIdx;
                    const support = metrics.perClass[actualClass]?.support || 0;
                    const hasData = support > 0;

                    let bg = 'rgba(255, 255, 255, 0.02)';
                    let color = 'rgba(255, 255, 255, 0.15)';

                    if (hasData) {
                      if (isDiag && count > 0) {
                        const intensity = count / support;
                        bg = `rgba(16, 185, 129, ${0.2 + intensity * 0.7})`;
                        color = '#fff';
                      } else if (!isDiag && count > 0) {
                        bg = 'rgba(239, 68, 68, 0.6)';
                        color = '#fff';
                      }
                    }

                    return (
                      <div
                        key={predClass}
                        onClick={() => setSelectedConfusionCell({ actual: actualClass, pred: predClass, count })}
                        style={{
                          aspectRatio: '1',
                          background: bg,
                          color: color,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: '2px',
                          cursor: 'pointer',
                          fontFamily: 'JetBrains Mono',
                          fontSize: count > 99 ? '0.55rem' : '0.65rem'
                        }}
                        title={`Actual: ${actualClass}, Predicted: ${predClass}, Count: ${count}`}
                      >
                        {count > 0 ? count : ''}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
