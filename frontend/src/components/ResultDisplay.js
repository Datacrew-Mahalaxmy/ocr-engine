// ResultDisplay.jsx
import React, { useEffect, useState } from 'react';

const ResultDisplay = ({ result, processing, onDownload }) => {
  const [animatedConfidence, setAnimatedConfidence] = useState(0);
  const [activeTab, setActiveTab] = useState('visual');

  useEffect(() => {
    if (result?.confidence) {
      const timer = setTimeout(() => {
        setAnimatedConfidence(result.confidence * 100);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [result]);

  if (processing) {
    return (
      <div className="creative-processing">
        <div className="processing-animation">
          <div className="pulse-ring"></div>
          <div className="processing-icon">🔍</div>
        </div>
        <h3>Analyzing Document</h3>
        <p>Our AI is extracting and classifying content...</p>
        <div className="processing-steps">
          <span className="step active">📄 Extracting</span>
          <span className="step">🔬 Analyzing</span>
          <span className="step">🎯 Classifying</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="creative-empty">
        <div className="empty-illustration">
          <svg width="200" height="200" viewBox="0 0 100 100">
            <circle cx="50" cy="40" r="30" fill="#f0f9ff" stroke="#6366f1" strokeWidth="2" strokeDasharray="4 4"/>
            <path d="M30 70 L50 50 L70 70" stroke="#6366f1" strokeWidth="3" fill="none" strokeLinecap="round"/>
            <rect x="40" y="75" width="20" height="15" fill="#e0e7ff" rx="4"/>
          </svg>
        </div>
        <h3>Ready for Analysis</h3>
        <p>Upload a document to see intelligent results with rich visualizations</p>
        <div className="empty-features">
          <span>📊 Confidence Metrics</span>
          <span>🎯 Type Classification</span>
          <span>📝 Text Extraction</span>
        </div>
      </div>
    );
  }

  const confidencePercentage = (result.confidence * 100).toFixed(1);
  const confidenceColor = confidencePercentage >= 80 ? '#10b981' : confidencePercentage >= 50 ? '#f59e0b' : '#ef4444';
  
  // Word cloud data
  const allText = Object.values(result.extractedText || {}).join(' ');
  const words = allText.split(/\s+/)
    .filter(word => word.length > 3)
    .reduce((acc, word) => {
      acc[word] = (acc[word] || 0) + 1;
      return acc;
    }, {});
  
  const topWords = Object.entries(words)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);

  return (
    <div className="creative-result">
      {/* Tab Navigation for Result Views */}
      <div className="result-tabs">
        <button 
          className={`tab-btn ${activeTab === 'visual' ? 'active' : ''}`}
          onClick={() => setActiveTab('visual')}
        >
          <span>📊</span> Visual Dashboard
        </button>
        <button 
          className={`tab-btn ${activeTab === 'text' ? 'active' : ''}`}
          onClick={() => setActiveTab('text')}
        >
          <span>📝</span> Extracted Text
        </button>
      </div>

      {activeTab === 'visual' ? (
        /* Visual Dashboard Tab */
        <div className="visual-dashboard">
          {/* Header Status Card */}
          <div className={`status-hero ${result.isMatch ? 'match' : 'mismatch'}`}>
            <div className="status-icon">
              {result.isMatch ? '✓' : '✕'}
            </div>
            <div className="status-content">
              <h2>{result.isMatch ? 'Document Verified' : 'Type Mismatch Detected'}</h2>
              <p>{result.isMatch ? 'Perfect match with expected document type' : 'Document type does not match expectation'}</p>
            </div>
          </div>

          {/* Main Metrics Grid */}
          <div className="metrics-dashboard">
            {/* Confidence Gauge */}
            <div className="metric-card gauge-card">
              <h4>Confidence Score</h4>
              <div className="gauge-container">
                <svg viewBox="0 0 120 120" className="gauge-svg">
                  <defs>
                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#ef4444" />
                      <stop offset="50%" stopColor="#f59e0b" />
                      <stop offset="100%" stopColor="#10b981" />
                    </linearGradient>
                  </defs>
                  <circle cx="60" cy="60" r="54" fill="none" stroke="#e5e7eb" strokeWidth="12"/>
                  <circle
                    cx="60"
                    cy="60"
                    r="54"
                    fill="none"
                    stroke="url(#gaugeGradient)"
                    strokeWidth="12"
                    strokeLinecap="round"
                    strokeDasharray="339.292"
                    strokeDashoffset={339.292 - (339.292 * animatedConfidence) / 100}
                    transform="rotate(-90 60 60)"
                    style={{ transition: 'stroke-dashoffset 1s ease' }}
                  />
                </svg>
                <div className="gauge-center">
                  <span className="gauge-value">{confidencePercentage}%</span>
                  <span className="gauge-label">Confidence</span>
                </div>
              </div>
            </div>

            {/* Type Comparison Card */}
            <div className="metric-card type-card">
              <h4>Document Type Analysis</h4>
              <div className="type-comparison">
                <div className="type-item">
                  <span className="type-label">Expected</span>
                  <span className="type-badge expected">{result.expectedDisplay}</span>
                </div>
                <div className="vs-divider">VS</div>
                <div className="type-item">
                  <span className="type-label">Detected</span>
                  <span className="type-badge detected">{result.actualDisplay}</span>
                </div>
              </div>
              <div className="match-indicator">
                <div className="match-bar">
                  <div 
                    className="match-fill" 
                    style={{ 
                      width: result.isMatch ? '100%' : '0%',
                      background: result.isMatch ? '#10b981' : '#ef4444'
                    }}
                  />
                </div>
                <span>{result.isMatch ? 'Exact Match' : 'No Match'}</span>
              </div>
            </div>

            {/* Statistics Card */}
            <div className="metric-card stats-card">
              <h4>Document Statistics</h4>
              <div className="stats-list">
                <div className="stat-row">
                  <span>📄 Total Regions</span>
                  <strong>{result.totalRegions || 0}</strong>
                </div>
                <div className="stat-row">
                  <span>📝 Words Extracted</span>
                  <strong>{allText.split(/\s+/).length}</strong>
                </div>
                <div className="stat-row">
                  <span>🔤 Characters</span>
                  <strong>{allText.length}</strong>
                </div>
                <div className="stat-row">
                  <span>📊 Classification Method</span>
                  <strong>{result.classification?.method || 'N/A'}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Word Cloud Visualization */}
          {topWords.length > 0 && (
            <div className="word-cloud-section">
              <h4>📊 Key Terms Extracted</h4>
              <div className="word-cloud">
                {topWords.map(([word, count], index) => (
                  <span
                    key={index}
                    className="cloud-word"
                    style={{
                      fontSize: `${Math.min(1 + count * 0.3, 2.5)}rem`,
                      opacity: 0.6 + (count / topWords[0][1]) * 0.4,
                      color: `hsl(${index * 18}, 70%, 50%)`
                    }}
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="result-actions">
            <button className="action-btn primary" onClick={onDownload}>
              <span>⬇️</span>
              Download JSON Report
            </button>
            {result.confidence < 0.5 && (
              <div className="warning-message">
                <span>⚠️</span>
                Low confidence detection - manual verification recommended
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Extracted Text Tab */
        <div className="text-tab">
          <div className="text-header">
            <h3>📝 Extracted Text Content</h3>
            <span className="text-stats">{allText.split(/\s+/).length} words • {allText.length} characters</span>
          </div>
          
          <div className="text-content-wrapper">
            {Object.entries(result.extractedText || {}).map(([page, text]) => (
              <div key={page} className="page-card">
                <div className="page-header">
                  <span className="page-number">Page {page}</span>
                  <span className="word-count">{text.split(/\s+/).length} words</span>
                </div>
                <pre className="ocr-text">{text}</pre>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;