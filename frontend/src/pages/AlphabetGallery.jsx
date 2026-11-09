import React, { useState, useMemo } from 'react';
import { Search, Filter, BookOpen, ExternalLink, Sparkles, AlertCircle, CheckCircle2, Play } from 'lucide-react';
import { ASL_ALPHABET, ASL_CATEGORIES } from '../data/aslData';

export default function AlphabetGallery({ onPracticeLetter }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedLetter, setSelectedLetter] = useState(null);

  const filteredLetters = useMemo(() => {
    return ASL_ALPHABET.filter((item) => {
      const matchesSearch =
        item.letter.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.category.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesCategory =
        selectedCategory === 'All' || item.category === selectedCategory;

      return matchesSearch && matchesCategory;
    });
  }, [searchQuery, selectedCategory]);

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <BookOpen color="#8b5cf6" size={26} /> ASL Alphabet Reference Guide (A&ndash;Z)
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginTop: '0.25rem' }}>
          Explore authentic American Sign Language fingerspelling hand postures, finger alignments, and common sign distinctions.
        </p>
      </div>

      {/* Search and Filters Bar */}
      <div className="glass-card" style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        padding: '1.25rem'
      }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Search Box */}
          <div style={{
            flex: '1',
            minWidth: '260px',
            position: 'relative',
            display: 'flex',
            alignItems: 'center'
          }}>
            <Search size={18} style={{ position: 'absolute', left: '1rem', color: 'var(--text-dim)' }} />
            <input
              type="text"
              placeholder="Search by letter (e.g. 'A', 'K') or description (e.g. 'fist', 'peace', 'thumb')..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem 1rem 0.75rem 2.8rem',
                borderRadius: '10px',
                border: '1px solid var(--border-glass)',
                background: 'rgba(7, 11, 22, 0.6)',
                color: '#fff',
                fontSize: '0.92rem',
                outline: 'none'
              }}
            />
          </div>

          <div style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>
            Showing <strong>{filteredLetters.length}</strong> of 26 letters
          </div>
        </div>

        {/* Category Pills */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {ASL_CATEGORIES.map((cat) => {
            const isSelected = selectedCategory === cat;
            return (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                style={{
                  padding: '0.35rem 0.85rem',
                  borderRadius: '999px',
                  fontSize: '0.8rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  border: isSelected ? '1px solid rgba(139, 92, 246, 0.6)' : '1px solid rgba(255, 255, 255, 0.08)',
                  background: isSelected ? 'rgba(139, 92, 246, 0.25)' : 'rgba(255, 255, 255, 0.03)',
                  color: isSelected ? '#fff' : 'var(--text-muted)',
                  transition: 'all 0.2s ease'
                }}
              >
                {cat}
              </button>
            );
          })}
        </div>
      </div>

      {/* Alphabet Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))',
        gap: '1.25rem'
      }}>
        {filteredLetters.map((item) => (
          <div
            key={item.letter}
            className="glass-card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '1rem',
              padding: '1.4rem',
              position: 'relative',
              background: 'linear-gradient(135deg, rgba(20, 29, 53, 0.6) 0%, rgba(13, 20, 38, 0.7) 100%)'
            }}
          >
            {/* Top header */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.8rem',
                    fontWeight: 900,
                    color: '#fff',
                    fontFamily: 'JetBrains Mono, monospace',
                    boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)'
                  }}>
                    {item.letter}
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{item.name}</h3>
                    <span style={{
                      fontSize: '0.72rem',
                      color: '#a78bfa',
                      background: 'rgba(139, 92, 246, 0.15)',
                      padding: '0.15rem 0.5rem',
                      borderRadius: '6px'
                    }}>
                      {item.category}
                    </span>
                  </div>
                </div>

                <span style={{
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  padding: '0.2rem 0.55rem',
                  borderRadius: '999px',
                  background: item.difficulty === 'Easy' ? 'rgba(16, 185, 129, 0.15)' : item.difficulty === 'Medium' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                  color: item.difficulty === 'Easy' ? '#34d399' : item.difficulty === 'Medium' ? '#fbbf24' : '#f87171'
                }}>
                  {item.difficulty}
                </span>
              </div>

              {/* Description */}
              <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
                {item.description}
              </p>

              {/* Distinction tip */}
              <div style={{
                marginTop: '0.85rem',
                padding: '0.65rem 0.85rem',
                borderRadius: '8px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                fontSize: '0.78rem',
                color: 'var(--text-dim)',
                lineHeight: 1.5
              }}>
                <strong style={{ color: '#cbd5e1' }}>Note: </strong> {item.confusions}
              </div>
            </div>

            {/* Action footer */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingTop: '0.75rem',
              borderTop: '1px solid rgba(255, 255, 255, 0.06)'
            }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                ASL Canonical Shape
              </span>
              <button
                className="btn-secondary"
                onClick={() => onPracticeLetter(item.letter)}
                style={{ padding: '0.4rem 0.85rem', fontSize: '0.78rem' }}
              >
                <Play size={13} />
                <span>Practice &lsquo;{item.letter}&rsquo;</span>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
