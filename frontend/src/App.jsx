import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import LiveRecognition from './pages/LiveRecognition';
import WebcamEvaluation from './pages/WebcamEvaluation';
import AlphabetGallery from './pages/AlphabetGallery';
import ModelPerformance from './pages/ModelPerformance';
import AboutProject from './pages/AboutProject';
import { loadASLModel } from './services/cnnClassifier';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [cameraActive, setCameraActive] = useState(false);

  // Pre-load model in background on app start
  useEffect(() => {
    loadASLModel().catch(err => {
      console.warn('Initial model preload notification:', err);
    });
  }, []);

  const handlePracticeLetter = (letter) => {
    setActiveTab('recognition');
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        cameraActive={cameraActive}
      />

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'home' && (
          <Home onNavigate={setActiveTab} />
        )}
        {activeTab === 'recognition' && (
          <LiveRecognition />
        )}
        {activeTab === 'evaluation' && (
          <WebcamEvaluation />
        )}
        {activeTab === 'gallery' && (
          <AlphabetGallery onPracticeLetter={handlePracticeLetter} />
        )}
        {activeTab === 'performance' && (
          <ModelPerformance />
        )}
        {activeTab === 'about' && (
          <AboutProject />
        )}
      </main>

      <Footer />
    </div>
  );
}
