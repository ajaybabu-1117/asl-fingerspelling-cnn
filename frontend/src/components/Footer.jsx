import React from 'react';
import { Cpu, Eye, Code, Award } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer">
      <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'center', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <Eye size={15} color="#06b6d4" /> MediaPipe Hand Landmarker (21 Points)
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <Cpu size={15} color="#8b5cf6" /> TensorFlow.js CNN (64x64 Landmark Images)
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <Code size={15} color="#3b82f6" /> React + Vite SPA
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <Award size={15} color="#10b981" /> College Project Ready
          </span>
        </div>
        <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
          ASL Alphabet Fingerspelling Recognition System &copy; {new Date().getFullYear()} &bull; American Sign Language (ASL) A&ndash;Z Real-Time Classification
        </div>
      </div>
    </footer>
  );
}
