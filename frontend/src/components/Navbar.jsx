import React from 'react';
import { Camera, BookOpen, BarChart3, Info, Home, Sparkles, Activity } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, cameraActive }) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'recognition', label: 'Live Recognition', icon: Camera, badge: cameraActive ? 'LIVE' : null },
    { id: 'gallery', label: 'ASL Alphabet', icon: BookOpen },
    { id: 'performance', label: 'Model Metrics', icon: BarChart3 },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  return (
    <nav className="navbar">
      <div className="nav-brand" onClick={() => setActiveTab('home')} style={{ cursor: 'pointer' }}>
        <div className="brand-icon">
          <Sparkles size={20} color="#ffffff" />
        </div>
        <div className="brand-text">
          <div style={{ fontSize: '1.05rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
            ASL <span>Fingerspelling</span> CNN
          </div>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 500 }}>
            MediaPipe + TensorFlow.js Real-Time
          </div>
        </div>
      </div>

      <ul className="nav-links">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <li key={item.id}>
              <button
                className={`nav-btn ${isActive ? 'active' : ''}`}
                onClick={() => setActiveTab(item.id)}
              >
                <Icon size={16} />
                <span>{item.label}</span>
                {item.badge && (
                  <span style={{
                    fontSize: '0.65rem',
                    background: '#10b981',
                    color: '#fff',
                    padding: '0.1rem 0.4rem',
                    borderRadius: '999px',
                    fontWeight: 700,
                    letterSpacing: '0.05em'
                  }}>
                    {item.badge}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
