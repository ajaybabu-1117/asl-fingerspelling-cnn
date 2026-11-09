import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Camera, CameraOff, Sparkles, RefreshCw, Trash2, CheckCircle2,
  AlertCircle, Volume2, Plus, Delete, Layers, Activity, Eye
} from 'lucide-react';
import { getHandLandmarker, drawHandLandmarks } from '../services/handDetector';
import { predictASLLetter, getModelStatus, resetClassifierSmoothing } from '../services/cnnClassifier';

export default function LiveRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameId = useRef(null);

  // States
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStatus, setCameraStatus] = useState('Camera Off'); // 'Camera Off', 'Connecting', 'Live', 'Permission Denied', 'Error'
  const [errorMessage, setErrorMessage] = useState(null);
  const [handDetected, setHandDetected] = useState(false);

  // Prediction output
  const [prediction, setPrediction] = useState({
    letter: '-',
    confidence: 0,
    latencyMs: 0,
    topPredictions: [],
    rasterPreview: null
  });

  // History and sentence builder
  const [history, setHistory] = useState([]);
  const [spelledText, setSpelledText] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [stabilityStatus, setStabilityStatus] = useState('Idle'); // 'Idle', 'Tracking', 'Locked'

  // Last detected landmark cache & consensus window
  const lastLandmarksRef = useRef(null);
  const streamRef = useRef(null);
  const consensusBufferRef = useRef([]);
  const stableLetterRef = useRef('-');

  // Start Camera
  const startCamera = async () => {
    setErrorMessage(null);
    setCameraStatus('Connecting');

    try {
      // Pre-warm HandLandmarker
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
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraStatus('Permission Denied');
        setErrorMessage('Camera access was denied. Please allow camera permissions in your browser address bar.');
      } else {
        setCameraStatus('Error');
        setErrorMessage(`Failed to open camera: ${err.message || 'Unknown device error'}`);
      }
      setCameraActive(false);
    }
  };

  // Stop Camera
  const stopCamera = useCallback(() => {
    if (animFrameId.current) {
      cancelAnimationFrame(animFrameId.current);
      animFrameId.current = null;
    }

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

    resetClassifierSmoothing();
    consensusBufferRef.current = [];
    stableLetterRef.current = '-';
    setCameraActive(false);
    setCameraStatus('Camera Off');
    setHandDetected(false);
    setStabilityStatus('Idle');
    setPrediction({
      letter: '-',
      confidence: 0,
      latencyMs: 0,
      topPredictions: [],
      rasterPreview: null
    });
  }, []);

  // Frame processing loop
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

              // Draw skeleton overlay
              drawHandLandmarks(ctx, currentLandmarks, canvas.width, canvas.height);

              // Predict letter with CNN
              const res = await predictASLLetter(currentLandmarks);
              if (res) {
                // Consensus & hysteresis filter to eliminate flickering
                if (res.isConfident && res.letter !== '...') {
                  const buf = consensusBufferRef.current;
                  buf.push(res.letter);
                  if (buf.length > 4) buf.shift();

                  const counts = {};
                  buf.forEach(l => { counts[l] = (counts[l] || 0) + 1; });
                  const [mostFrequent, freq] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];

                  if (freq >= 3 || res.confidence >= 80) {
                    stableLetterRef.current = mostFrequent;
                    setStabilityStatus('Locked');
                  } else {
                    setStabilityStatus('Tracking');
                  }
                } else {
                  consensusBufferRef.current = [];
                  setStabilityStatus(res.confidence > 35 ? 'Aligning' : 'Idle');
                }

                setPrediction({
                  ...res,
                  letter: stableLetterRef.current !== '-' ? stableLetterRef.current : (res.isConfident ? res.letter : '...')
                });
              }
            } else {
              setHandDetected(false);
              lastLandmarksRef.current = null;
              resetClassifierSmoothing();
              consensusBufferRef.current = [];
              stableLetterRef.current = '-';
              setStabilityStatus('Idle');
            }
          } catch (err) {
            console.error('Frame detection error:', err);
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
      if (animFrameId.current) {
        cancelAnimationFrame(animFrameId.current);
      }
    };
  }, [cameraActive]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Actions
  const handleCaptureGesture = () => {
    if (prediction.letter && prediction.letter !== '-') {
      const entry = {
        id: Date.now(),
        letter: prediction.letter,
        confidence: prediction.confidence,
        time: new Date().toLocaleTimeString(),
        rasterPreview: prediction.rasterPreview
      };
      setHistory((prev) => [entry, ...prev.slice(0, 19)]);
      setSpelledText((prev) => prev + prediction.letter);
    }
  };

  const handleAddSpace = () => {
    setSpelledText((prev) => prev + ' ');
  };

  const handleBackspace = () => {
    setSpelledText((prev) => prev.slice(0, -1));
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  const handleReset = () => {
    setHistory([]);
    setSpelledText('');
    setPrediction({
      letter: '-',
      confidence: 0,
      latencyMs: 0,
      topPredictions: [],
      rasterPreview: null
    });
  };

  // Text-To-Speech
  const speakText = () => {
    if (!spelledText.trim() || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(spelledText);
    utterance.rate = 0.95;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Activity color="#8b5cf6" size={26} /> Live ASL Recognition Studio
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>
            Hold your hand up to the camera. The CNN recognizes letters A&ndash;Z from 21 MediaPipe landmarks.
          </p>
        </div>

        {/* Status Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div className={`badge-status ${cameraActive ? 'active' : cameraStatus === 'Connecting' ? 'warning' : 'inactive'}`}>
            <span className={`status-dot ${cameraActive ? 'pulse' : ''}`} />
            <span>Camera: {cameraStatus}</span>
          </div>

          <div className={`badge-status ${handDetected ? 'active' : 'inactive'}`}>
            <span className={`status-dot ${handDetected ? 'pulse' : ''}`} />
            <span>{handDetected ? 'Hand Tracked (21 Pts)' : 'No Hand in Frame'}</span>
          </div>

          <div className={`badge-status ${stabilityStatus === 'Locked' ? 'active' : stabilityStatus === 'Tracking' ? 'warning' : 'inactive'}`}>
            <span className={`status-dot ${stabilityStatus === 'Locked' ? 'pulse' : ''}`} />
            <span>State: {stabilityStatus}</span>
          </div>
        </div>
      </div>

      {/* Error Alert if permission failed */}
      {errorMessage && (
        <div style={{
          background: 'rgba(244, 63, 94, 0.12)',
          border: '1px solid rgba(244, 63, 94, 0.35)',
          borderRadius: '12px',
          padding: '1rem 1.25rem',
          color: '#fecdd3',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <AlertCircle size={20} color="#f43f5e" />
          <div style={{ fontSize: '0.92rem' }}>{errorMessage}</div>
        </div>
      )}

      {/* Main Studio Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(320px, 1.4fr) minmax(300px, 1fr)',
        gap: '1.5rem',
        alignItems: 'start'
      }}>
        {/* Left Column: Video Viewport & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="glass-card" style={{
            padding: '0.75rem',
            position: 'relative',
            borderRadius: '20px',
            overflow: 'hidden',
            aspectRatio: '4 / 3',
            background: '#040711',
            border: cameraActive ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid var(--border-glass)',
            boxShadow: cameraActive ? '0 0 30px rgba(139, 92, 246, 0.2)' : 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {/* Webcam Video */}
            <video
              ref={videoRef}
              playsInline
              muted
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                transform: 'scaleX(-1)', // Mirror mode
                borderRadius: '14px',
                display: cameraActive ? 'block' : 'none'
              }}
            />

            {/* Skeleton Canvas Overlay */}
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: '0.75rem',
                left: '0.75rem',
                width: 'calc(100% - 1.5rem)',
                height: 'calc(100% - 1.5rem)',
                objectFit: 'cover',
                transform: 'scaleX(-1)', // Match mirrored video
                pointerEvents: 'none',
                display: cameraActive ? 'block' : 'none'
              }}
            />

            {/* Placeholder when camera is OFF */}
            {!cameraActive && (
              <div style={{
                textAlign: 'center',
                padding: '2rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '1rem',
                color: 'var(--text-muted)'
              }}>
                <div style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '50%',
                  background: 'rgba(255, 255, 255, 0.05)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-dim)'
                }}>
                  <CameraOff size={30} />
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '1.1rem' }}>
                    Camera is Currently Off
                  </div>
                  <div style={{ fontSize: '0.88rem', marginTop: '0.25rem' }}>
                    Click &ldquo;Start Camera&rdquo; below to initialize MediaPipe detection.
                  </div>
                </div>
                <button className="btn-primary" onClick={startCamera}>
                  <Camera size={18} />
                  <span>Start Camera</span>
                </button>
              </div>
            )}

            {/* In-viewport Live badge */}
            {cameraActive && (
              <div style={{
                position: 'absolute',
                top: '1.25rem',
                left: '1.25rem',
                background: 'rgba(0, 0, 0, 0.65)',
                backdropFilter: 'blur(8px)',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '0.78rem',
                fontWeight: 600
              }}>
                <span className="status-dot pulse" style={{ background: '#10b981' }} />
                <span>REC &bull; 640x480</span>
              </div>
            )}
          </div>

          {/* Camera Buttons Bar */}
          <div className="glass-card" style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
            padding: '1rem 1.25rem'
          }}>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
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

              <button
                className="btn-secondary"
                onClick={handleCaptureGesture}
                disabled={!cameraActive || prediction.letter === '-'}
                style={{ opacity: !cameraActive || prediction.letter === '-' ? 0.5 : 1 }}
              >
                <Plus size={17} />
                <span>Capture Letter</span>
              </button>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn-secondary"
                onClick={handleReset}
                title="Reset history and text"
                style={{ padding: '0.65rem 1rem' }}
              >
                <RefreshCw size={16} />
                <span>Reset</span>
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Output Dashboard */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Main Prediction Banner Card */}
          <div className="glass-card" style={{
            padding: '2rem 1.5rem',
            textAlign: 'center',
            position: 'relative',
            overflow: 'hidden',
            background: 'linear-gradient(135deg, rgba(25, 33, 60, 0.7) 0%, rgba(13, 20, 38, 0.8) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.35)',
            boxShadow: '0 0 25px rgba(139, 92, 246, 0.15)'
          }}>
            <div style={{
              fontSize: '0.82rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--text-dim)',
              marginBottom: '0.5rem'
            }}>
              CNN Output Prediction
            </div>

            {/* Big Predicted Letter */}
            <div style={{
              fontSize: '6rem',
              fontWeight: 900,
              lineHeight: 1,
              fontFamily: 'JetBrains Mono, monospace',
              background: 'linear-gradient(135deg, #ffffff 0%, #a78bfa 60%, #38bdf8 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 0 20px rgba(139, 92, 246, 0.4))',
              margin: '0.5rem 0'
            }}>
              {prediction.letter}
            </div>

            {/* Confidence Bar */}
            <div style={{ maxWidth: '280px', margin: '1rem auto 0 auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Confidence Score</span>
                <span style={{ fontWeight: 700, color: prediction.confidence > 75 ? '#34d399' : '#fbbf24' }}>
                  {prediction.confidence}%
                </span>
              </div>

              <div style={{
                height: '8px',
                background: 'rgba(255, 255, 255, 0.08)',
                borderRadius: '999px',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  width: `${prediction.confidence}%`,
                  background: prediction.confidence > 75
                    ? 'linear-gradient(90deg, #8b5cf6, #10b981)'
                    : 'linear-gradient(90deg, #f59e0b, #ef4444)',
                  borderRadius: '999px',
                  transition: 'width 0.2s ease'
                }} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.4rem' }}>
                <span>Latency: {prediction.latencyMs} ms</span>
                <span>Softmax Prob</span>
              </div>
            </div>

            {/* Top 3 Alternative Classes */}
            {prediction.topPredictions && prediction.topPredictions.length > 1 && (
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                gap: '0.75rem',
                marginTop: '1.25rem',
                paddingTop: '1rem',
                borderTop: '1px solid rgba(255, 255, 255, 0.06)'
              }}>
                {prediction.topPredictions.slice(1, 4).map((alt) => (
                  <div key={alt.letter} style={{
                    fontSize: '0.78rem',
                    background: 'rgba(255, 255, 255, 0.03)',
                    padding: '0.25rem 0.6rem',
                    borderRadius: '6px',
                    border: '1px solid rgba(255, 255, 255, 0.08)'
                  }}>
                    <strong style={{ color: '#fff' }}>{alt.letter}</strong>: {(alt.probability * 100).toFixed(1)}%
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* CNN Landmark Skeleton Stream Preview (32x32) */}
          <div className="glass-card" style={{
            display: 'flex',
            alignItems: 'center',
            gap: '1rem',
            padding: '1rem 1.25rem'
          }}>
            <div style={{
              width: '56px',
              height: '56px',
              background: '#000',
              border: '1px solid rgba(139, 92, 246, 0.4)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden'
            }}>
              {prediction.rasterPreview ? (
                <img
                  src={prediction.rasterPreview}
                  alt="CNN 32x32 Landmark Raster Input"
                  style={{ width: '100%', height: '100%', imageRendering: 'pixelated' }}
                />
              ) : (
                <Layers size={22} color="var(--text-dim)" />
              )}
            </div>

            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)' }}>
                CNN Landmark Input (32&times;32&times;1)
              </div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                Handedness-invariant normalized coordinates &amp; antialiased skeleton bones fed directly to Conv2D layers.
              </div>
            </div>
          </div>

          {/* Live Fingerspelling Word/Sentence Builder */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Fingerspelled Sentence
              </span>
              <button
                onClick={speakText}
                disabled={!spelledText.trim()}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: isSpeaking ? '#34d399' : 'var(--text-muted)',
                  cursor: spelledText.trim() ? 'pointer' : 'default',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  fontSize: '0.8rem'
                }}
              >
                <Volume2 size={16} />
                <span>{isSpeaking ? 'Speaking...' : 'Speak Audio'}</span>
              </button>
            </div>

            <div style={{
              minHeight: '52px',
              background: 'rgba(0, 0, 0, 0.3)',
              borderRadius: '10px',
              padding: '0.75rem 1rem',
              border: '1px solid var(--border-glass)',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '1.25rem',
              fontWeight: 700,
              color: '#38bdf8',
              letterSpacing: '0.1em',
              wordBreak: 'break-all'
            }}>
              {spelledText || <span style={{ color: 'var(--text-dim)', fontSize: '0.9rem', fontWeight: 400 }}>Letters captured will appear here...</span>}
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
              <button
                className="btn-secondary"
                onClick={handleAddSpace}
                style={{ flex: 1, padding: '0.45rem', fontSize: '0.82rem', justifyContent: 'center' }}
              >
                Space [ _ ]
              </button>
              <button
                className="btn-secondary"
                onClick={handleBackspace}
                style={{ padding: '0.45rem 0.9rem', fontSize: '0.82rem' }}
                title="Backspace"
              >
                <Delete size={15} />
              </button>
              <button
                className="btn-secondary"
                onClick={() => setSpelledText('')}
                style={{ padding: '0.45rem 0.9rem', fontSize: '0.82rem' }}
                title="Clear text"
              >
                <Trash2 size={15} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Predictions History */}
      <div className="glass-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Recent Gesture Log</h2>
            <p style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>
              Timestamped sequence of captured ASL fingerspelled alphabet characters
            </p>
          </div>
          {history.length > 0 && (
            <button
              onClick={handleClearHistory}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#f87171',
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem'
              }}
            >
              <Trash2 size={14} /> Clear History
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-dim)', fontSize: '0.9rem' }}>
            No gestures captured yet. Click &ldquo;Capture Letter&rdquo; while holding a sign to log history.
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
            gap: '0.75rem'
          }}>
            {history.map((item) => (
              <div key={item.id} style={{
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: '10px',
                padding: '0.75rem',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem'
              }}>
                {item.rasterPreview ? (
                  <img
                    src={item.rasterPreview}
                    alt={item.letter}
                    style={{ width: '36px', height: '36px', borderRadius: '4px', background: '#000' }}
                  />
                ) : (
                  <div style={{ width: '36px', height: '36px', borderRadius: '4px', background: 'rgba(139, 92, 246, 0.2)' }} />
                )}
                <div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#fff', fontFamily: 'JetBrains Mono' }}>
                    {item.letter}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#34d399' }}>{item.confidence}%</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>{item.time}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
