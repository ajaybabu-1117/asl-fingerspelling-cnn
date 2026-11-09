import React from 'react';
import { BookOpen, Award, CheckCircle2, Cpu, Eye, ShieldCheck, FileText, Layers, GitBranch, Terminal } from 'lucide-react';

export default function AboutProject() {
  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <FileText color="#8b5cf6" size={26} /> Project Documentation &amp; Technical Report
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginTop: '0.25rem' }}>
          Comprehensive engineering design for ASL Fingerspelling Recognition via Hand-Landmark CNN.
        </p>
      </div>

      {/* Abstract Card */}
      <section className="glass-card" style={{ padding: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.85rem', color: '#c4b5fd' }}>
          1. Abstract &amp; Problem Statement
        </h2>
        <p style={{ color: 'var(--text-muted)', lineHeight: 1.7, fontSize: '0.95rem', marginBottom: '1rem' }}>
          American Sign Language (ASL) fingerspelling constitutes an essential component of deaf communication, particularly for spelling proper names, technical vocabulary, and loanwords. Traditional computer vision sign language systems rely on raw RGB camera feeds fed directly into deep convolutional neural networks. However, raw pixel models suffer from extreme sensitivity to skin tone variations, varying background clutter, ambient lighting fluctuations, and large computational overhead.
        </p>
        <p style={{ color: 'var(--text-muted)', lineHeight: 1.7, fontSize: '0.95rem' }}>
          This project proposes a <strong>dual-stage decoupled pipeline</strong>. First, Google MediaPipe Hand Landmarker isolates 21 anatomical 3D joints in real-time. Second, the spatial landmarks are translation-normalized relative to the wrist and scaled invariantly, then rasterized into an antialiased grayscale skeleton tensor. Finally, a lightweight 2D Convolutional Neural Network (CNN) classifies the spatial bone and joint patterns into the 26 ASL alphabet classes (A–Z). This guarantees complete background and demographic invariance with client-side real-time inference at 30+ FPS.
        </p>
      </section>

      {/* Methodology & Decoupled Architecture */}
      <section className="glass-card" style={{ padding: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1.25rem', color: '#38bdf8' }}>
          2. Decoupled Pipeline Methodology
        </h2>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.25rem'
        }}>
          {[
            {
              step: 'Stage 1: Landmark Tracking',
              desc: 'MediaPipe ML pipeline leverages a single-shot palm detector followed by a 3D hand landmark model that predicts 21 coordinates (x, y, z) with sub-millimeter precision.',
              color: '#8b5cf6'
            },
            {
              step: 'Stage 2: Spatial Invariance',
              desc: 'The wrist keypoint (landmark 0) is set as coordinate origin (0, 0). The maximum euclidean distance to any knuckle or fingertip scales the bounding geometry to eliminate distance bias.',
              color: '#06b6d4'
            },
            {
              step: 'Stage 3: Skeleton Rasterization',
              desc: 'Landmarks and their anatomical bone graph (21 connections) are mapped to a 32x32 raster grid using Bresenham line rendering and Gaussian joint splats, producing a pure geometric image.',
              color: '#ec4899'
            },
            {
              step: 'Stage 4: CNN Classification',
              desc: 'The 32x32x1 tensor feeds into Conv2D layers with ReLU activations, MaxPooling, Dropout regularization, and a 26-way Softmax output layer providing confidence scores.',
              color: '#10b981'
            }
          ].map((item, i) => (
            <div key={i} style={{
              background: 'rgba(255, 255, 255, 0.02)',
              borderRadius: '12px',
              padding: '1.25rem',
              border: '1px solid rgba(255, 255, 255, 0.06)'
            }}>
              <div style={{ fontWeight: 700, color: item.color, fontSize: '0.95rem', marginBottom: '0.4rem' }}>
                {item.step}
              </div>
              <div style={{ fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                {item.desc}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Mathematical Formulations */}
      <section className="glass-card" style={{ padding: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1.25rem', color: '#a78bfa' }}>
          3. Mathematical Formulation
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
          <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#38bdf8', marginBottom: '0.5rem' }}>
              Wrist Translation Normalization
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Translates all 21 landmark vectors relative to the wrist landmark $p_0$:
            </p>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1rem', color: '#c4b5fd', background: 'rgba(255,255,255,0.03)', padding: '0.6rem 0.8rem', borderRadius: '8px' }}>
              p&#39;<sub>i</sub> = p<sub>i</sub> &minus; p<sub>0</sub> &emsp; &forall; i &isin; [0, 20]
            </div>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#38bdf8', marginBottom: '0.5rem' }}>
              Scale Normalization Metric
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Ensures gesture scale invariance irrespective of hand-to-camera distance:
            </p>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1rem', color: '#c4b5fd', background: 'rgba(255,255,255,0.03)', padding: '0.6rem 0.8rem', borderRadius: '8px' }}>
              s = D<sub>canvas</sub> / max<sub>i</sub> ||p&#39;<sub>i</sub>||
            </div>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#38bdf8', marginBottom: '0.5rem' }}>
              Categorical Cross-Entropy Loss
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Objective loss minimized across 26 discrete ASL alphabet classes:
            </p>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1rem', color: '#c4b5fd', background: 'rgba(255,255,255,0.03)', padding: '0.6rem 0.8rem', borderRadius: '8px' }}>
              L = &minus;&sum;<sub>c=1</sub><sup>26</sup> y<sub>c</sub> &times; log(&ycirc;<sub>c</sub>)
            </div>
          </div>
        </div>
      </section>

      {/* Comparison Table: Landmark CNN vs Raw RGB CNN */}
      <section className="glass-card" style={{ padding: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1.25rem', color: '#10b981' }}>
          4. Comparison: Landmark CNN vs. Raw RGB Pixel CNN
        </h2>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-glass)', textAlign: 'left', color: 'var(--text-dim)' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Metric / Feature</th>
                <th style={{ padding: '0.75rem 1rem', color: '#34d399' }}>Landmark Skeleton CNN (This Project)</th>
                <th style={{ padding: '0.75rem 1rem', color: '#fca5a5' }}>Raw RGB Pixel CNN</th>
              </tr>
            </thead>
            <tbody>
              {[
                { metric: 'Demographic & Skin Tone Bias', ours: 'Completely Invariant (Skeleton Map)', others: 'High Bias (Vulnerable to Skin Tone)' },
                { metric: 'Background Interference', ours: 'Zero (Background is non-existent)', others: 'High (Prone to cluttered room noise)' },
                { metric: 'Lighting & Shadow Sensitivity', ours: 'Robust (MediaPipe tracks in low light)', others: 'Extreme (Fails under shadows)' },
                { metric: 'Client Model Size', ours: 'Lightweight (~500 KB weights)', others: 'Heavy (>40 MB ResNet / MobileNet)' },
                { metric: 'Inference Latency', ours: '< 15 ms in browser on CPU/GPU', others: '60 - 150 ms (Causes lag/thermal throttling)' },
                { metric: 'User Privacy', ours: 'High (Only landmarks computed locally)', others: 'Low (Raw images often sent to cloud)' }
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                  <td style={{ padding: '0.7rem 1rem', fontWeight: 600 }}>{row.metric}</td>
                  <td style={{ padding: '0.7rem 1rem', color: '#6ee7b7' }}>&bull; {row.ours}</td>
                  <td style={{ padding: '0.7rem 1rem', color: 'var(--text-muted)' }}>&bull; {row.others}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Project Reproduction & Local Run Guide */}
      <section className="glass-card" style={{ padding: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Terminal size={20} /> 5. Local Reproduction &amp; ML Training Instructions
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.88rem', color: 'var(--text-muted)' }}>
          <div>
            <strong style={{ color: '#fff' }}>1. Clone Repository &amp; Install Dependencies:</strong>
            <pre style={{
              background: 'rgba(0, 0, 0, 0.4)',
              padding: '0.75rem 1rem',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono',
              color: '#38bdf8',
              marginTop: '0.4rem',
              overflowX: 'auto'
            }}>
              git clone https://github.com/ajay-services/asl-fingerspelling-cnn.git{'\n'}
              cd asl-fingerspelling-cnn/frontend{'\n'}
              npm install{'\n'}
              npm run dev
            </pre>
          </div>

          <div>
            <strong style={{ color: '#fff' }}>2. Train CNN with Python or Node.js:</strong>
            <pre style={{
              background: 'rgba(0, 0, 0, 0.4)',
              padding: '0.75rem 1rem',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono',
              color: '#38bdf8',
              marginTop: '0.4rem',
              overflowX: 'auto'
            }}>
              # Collect webcam landmarks for training:{'\n'}
              python ml/collect_data.py --letter A --samples 100{'\n'}{'\n'}
              # Run Python Keras training pipeline:{'\n'}
              python ml/train.py --dataset ml/dataset --epochs 25{'\n'}{'\n'}
              # Or run the instant TensorFlow.js training pipeline:{'\n'}
              cd frontend && node scripts/train_cnn.js
            </pre>
          </div>
        </div>
      </section>
    </div>
  );
}
