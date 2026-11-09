import React from 'react';
import { Camera, Sparkles, Cpu, Eye, ShieldCheck, Zap, Layers, ArrowRight, PlayCircle, BookOpen, Award } from 'lucide-react';
import { ASL_ALPHABET } from '../data/aslData';

export default function Home({ onNavigate }) {
  const steps = [
    {
      num: '01',
      title: 'Real-Time Webcam Feed',
      desc: 'Streams high-frame-rate video directly in the browser with zero server uploads or latency.',
      icon: Camera,
      color: '#06b6d4'
    },
    {
      num: '02',
      title: 'MediaPipe Hand Landmarker',
      desc: 'Accurately tracks 21 3D keypoints (wrist, MCP, PIP, DIP, tips) in real-time.',
      icon: Eye,
      color: '#8b5cf6'
    },
    {
      num: '03',
      title: 'Spatial Normalization',
      desc: 'Translates wrist to origin and scales by hand span, making recognition invariant to camera distance.',
      icon: Layers,
      color: '#3b82f6'
    },
    {
      num: '04',
      title: 'Skeleton Rasterization',
      desc: 'Converts 21 coordinates into a 64x64 grayscale hand-landmark skeleton image, removing skin and background bias.',
      icon: Sparkles,
      color: '#ec4899'
    },
    {
      num: '05',
      title: 'Deep CNN Inference',
      desc: 'Trained multi-layer CNN extracts spatial joint patterns and computes softmax probabilities for letters A–Z.',
      icon: Cpu,
      color: '#10b981'
    }
  ];

  const highlights = [
    {
      title: '100% Invariant to Skin Tone & Background',
      desc: 'By training the CNN on landmark skeleton representations rather than raw RGB pixels, the model eliminates demographic bias, lighting sensitivity, and complex background interference.',
      icon: ShieldCheck,
      color: '#10b981'
    },
    {
      title: 'Ultra-Fast Browser Execution',
      desc: 'MediaPipe Tasks Vision WASM combined with TensorFlow.js runs smooth 30+ FPS client-side inference without requiring any expensive cloud backend or API keys.',
      icon: Zap,
      color: '#f59e0b'
    },
    {
      title: 'End-to-End Deep Learning Pipeline',
      desc: 'Complete machine learning lifecycle including landmark collection, spatial augmentation, Keras/TFJS architecture, training curves, and confusion matrix analysis.',
      icon: Cpu,
      color: '#8b5cf6'
    }
  ];

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      {/* Hero Section */}
      <section className="glass-card" style={{
        padding: '3.5rem 2.5rem',
        position: 'relative',
        overflow: 'hidden',
        border: '1px solid rgba(139, 92, 246, 0.25)',
        background: 'linear-gradient(135deg, rgba(20, 30, 55, 0.75) 0%, rgba(13, 21, 39, 0.85) 100%)'
      }}>
        {/* Glow decoration */}
        <div style={{
          position: 'absolute',
          top: '-30%',
          right: '-10%',
          width: '420px',
          height: '420px',
          background: 'radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none'
        }} />

        <div style={{ maxWidth: '820px', position: 'relative', zIndex: 1 }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 0.9rem',
            borderRadius: '999px',
            background: 'rgba(139, 92, 246, 0.15)',
            border: '1px solid rgba(139, 92, 246, 0.35)',
            color: '#c4b5fd',
            fontSize: '0.82rem',
            fontWeight: 600,
            marginBottom: '1.25rem'
          }}>
            <Sparkles size={14} /> College Major / Minor Project &bull; Machine Learning & Computer Vision
          </div>

          <h1 style={{
            fontSize: 'clamp(2.1rem, 4.2vw, 3.4rem)',
            fontWeight: 800,
            lineHeight: 1.15,
            letterSpacing: '-0.03em',
            marginBottom: '1.2rem'
          }}>
            ASL Alphabet Fingerspelling Recognition Using{' '}
            <span style={{
              background: 'linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              CNN on Hand-Landmark Images
            </span>
          </h1>

          <p style={{
            fontSize: '1.1rem',
            color: 'var(--text-muted)',
            lineHeight: 1.7,
            marginBottom: '2.2rem',
            maxWidth: '720px'
          }}>
            Real-time American Sign Language (ASL) alphabet recognition recognizing letters <strong>A through Z</strong> directly in your browser. Powered by <strong>MediaPipe Hand Landmarker (21 keypoints)</strong> and a custom <strong>Convolutional Neural Network (CNN)</strong> on normalized skeletal images.
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center' }}>
            <button
              className="btn-primary"
              style={{ fontSize: '1.05rem', padding: '0.9rem 2rem' }}
              onClick={() => onNavigate('recognition')}
            >
              <PlayCircle size={20} />
              <span>Start Real-Time Recognition</span>
            </button>

            <button
              className="btn-secondary"
              style={{ fontSize: '1.05rem', padding: '0.9rem 1.8rem' }}
              onClick={() => onNavigate('evaluation')}
            >
              <Award size={19} color="#a78bfa" />
              <span>Real-World Webcam Test</span>
            </button>
            <button
              className="btn-secondary"
              style={{ fontSize: '1.05rem', padding: '0.9rem 1.8rem' }}
              onClick={() => onNavigate('gallery')}
            >
              <BookOpen size={19} />
              <span>A–Z Alphabet Guide</span>
            </button>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <h2 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '0.5rem' }}>
            How The Dual-Stage Pipeline Works
          </h2>
          <p style={{ color: 'var(--text-muted)', maxWidth: '620px', margin: '0 auto', fontSize: '0.98rem' }}>
            Transforming raw webcam video into high-accuracy ASL character predictions through geometric normalization.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1.25rem'
        }}>
          {steps.map((st) => {
            const Icon = st.icon;
            return (
              <div key={st.num} className="glass-card" style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.85rem',
                position: 'relative',
                background: 'rgba(15, 23, 42, 0.7)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{
                    width: '42px',
                    height: '42px',
                    borderRadius: '10px',
                    background: `${st.color}20`,
                    border: `1px solid ${st.color}40`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: st.color
                  }}>
                    <Icon size={22} />
                  </div>
                  <span style={{
                    fontSize: '1.3rem',
                    fontWeight: 800,
                    color: 'rgba(255, 255, 255, 0.15)',
                    fontFamily: 'JetBrains Mono, monospace'
                  }}>
                    {st.num}
                  </span>
                </div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {st.title}
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
                  {st.desc}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Why Hand-Landmark CNN Section */}
      <section className="glass-card" style={{ padding: '2.5rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '1.5rem', textAlign: 'center' }}>
          Why CNN on Hand-Landmark Images?
        </h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.5rem'
        }}>
          {highlights.map((h, i) => {
            const Icon = h.icon;
            return (
              <div key={i} style={{
                background: 'rgba(255, 255, 255, 0.02)',
                borderRadius: '14px',
                padding: '1.5rem',
                border: '1px solid rgba(255, 255, 255, 0.06)'
              }}>
                <div style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: '8px',
                  background: `${h.color}20`,
                  color: h.color,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1rem'
                }}>
                  <Icon size={20} />
                </div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '0.6rem' }}>
                  {h.title}
                </h3>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  {h.desc}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ASL Alphabet A-Z Quick Preview */}
      <section>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
              ASL Alphabet Quick Reference (A&ndash;Z)
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>
              26 standardized American Sign Language fingerspelled letters
            </p>
          </div>
          <button
            className="btn-secondary"
            onClick={() => onNavigate('gallery')}
            style={{ fontSize: '0.88rem', padding: '0.6rem 1.2rem' }}
          >
            <span>View Full Guide with Instructions</span>
            <ArrowRight size={15} />
          </button>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(72px, 1fr))',
          gap: '0.75rem'
        }}>
          {ASL_ALPHABET.map((item) => (
            <div
              key={item.letter}
              onClick={() => onNavigate('gallery')}
              style={{
                background: 'rgba(18, 27, 49, 0.7)',
                border: '1px solid var(--border-glass)',
                borderRadius: '12px',
                padding: '0.85rem 0.5rem',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.35rem'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-3px)';
                e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.6)';
                e.currentTarget.style.background = 'rgba(139, 92, 246, 0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.borderColor = 'var(--border-glass)';
                e.currentTarget.style.background = 'rgba(18, 27, 49, 0.7)';
              }}
            >
              <span style={{
                fontSize: '1.5rem',
                fontWeight: 800,
                color: '#fff',
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                {item.letter}
              </span>
              <span style={{
                fontSize: '0.68rem',
                color: 'var(--text-dim)',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                maxWidth: '65px'
              }}>
                {item.category}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Technologies Used Grid */}
      <section className="glass-card" style={{ padding: '2rem' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1.25rem', textAlign: 'center' }}>
          Core Technologies & Architecture
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '1rem',
          textAlign: 'center'
        }}>
          {[
            { label: 'MediaPipe Tasks Vision', desc: 'Hand Landmark Model (WASM)', tag: 'Detection' },
            { label: 'TensorFlow.js', desc: 'Client-Side Neural Network', tag: 'Inference' },
            { label: 'React 19 + Vite', desc: 'Modern High-Perf Frontend', tag: 'Frontend' },
            { label: 'Python TensorFlow', desc: 'Keras Training Pipeline', tag: 'Machine Learning' },
            { label: 'Normalized Skeleton Maps', desc: '64x64 Grayscale Tensors', tag: 'Data Prep' },
            { label: 'Vercel Deployment', desc: 'Fast Edge CDN Hosting', tag: 'DevOps' }
          ].map((t, idx) => (
            <div key={idx} style={{
              background: 'rgba(255, 255, 255, 0.03)',
              borderRadius: '12px',
              padding: '1.2rem',
              border: '1px solid rgba(255, 255, 255, 0.06)'
            }}>
              <span style={{
                fontSize: '0.7rem',
                fontWeight: 700,
                color: '#8b5cf6',
                background: 'rgba(139, 92, 246, 0.15)',
                padding: '0.15rem 0.5rem',
                borderRadius: '6px'
              }}>
                {t.tag}
              </span>
              <div style={{ fontWeight: 600, fontSize: '0.95rem', marginTop: '0.6rem' }}>{t.label}</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>{t.desc}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
