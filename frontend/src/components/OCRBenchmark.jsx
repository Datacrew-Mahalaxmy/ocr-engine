// OCRBenchmark.jsx
import React, { useState } from 'react';
import './OCRBenchmark.css';

const OCRBenchmark = ({ 
  result: benchmarkResult, 
  processing: isProcessing,
  onCompare,
  textractFile,
  doctrFile,
  error,
  setError
}) => {
  const [selectedModel, setSelectedModel] = useState("all-MiniLM-L6-v2");

  // Model options for dropdown
  const modelOptions = [
    { value: "all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2 (Fast - Default)", description: "Balanced speed and accuracy" },
    { value: "all-mpnet-base-v2", label: "all-mpnet-base-v2 (High Accuracy)", description: "Best accuracy, slower" },
    { value: "multi-qa-mpnet-base-dot-v1", label: "multi-qa-mpnet-base-dot-v1", description: "Optimized for QA" }
  ];

  // Handle compare button click
  const handleCompareClick = () => {
    if (onCompare) {
      onCompare(selectedModel);
    }
  };

  const getMetricClass = (value, type) => {
    if (type === 'error') {
      if (value <= 10) return 'success';
      if (value <= 20) return 'warning';
      return 'danger';
    }
    if (value >= 80) return 'success';
    if (value >= 60) return 'warning';
    return 'danger';
  };

  // Get current model's data from results
  const currentModelData = benchmarkResult?.models?.[selectedModel];

  return (
    <div className="ocr-benchmark">
      {/* MODEL SELECTOR - Always visible at the top */}
      <div className="model-section">
        <div className="model-label">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 18v-4M12 8v4" />
          </svg>
          <span>SELECT MODEL FOR COMPARISON</span>
        </div>
        <div className="model-select-wrapper">
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isProcessing}
            className="model-select"
          >
            {modelOptions.map(model => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <svg className="select-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
        <div className="model-description">
          {modelOptions.find(m => m.value === selectedModel)?.description}
        </div>
      </div>

      {/* Compare Button - Below model selector */}
      <button 
        className="compare-btn" 
        onClick={handleCompareClick} 
        disabled={!textractFile || !doctrFile || isProcessing}
      >
        {isProcessing ? (
          <>
            <svg className="loading-spinner" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v2M12 12v2M12 18v2" />
            </svg>
            COMPUTING SIMILARITY...
          </>
        ) : (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            COMPARE DOCUMENTS USING {selectedModel}
          </>
        )}
      </button>

      {/* File Status Indicators */}
      <div className="file-status">
        <div className={`status-indicator ${textractFile ? 'active' : 'inactive'}`}>
          <span className="status-dot"></span>
          Textract JSON: {textractFile ? textractFile.name : 'Not selected'}
        </div>
        <div className={`status-indicator ${doctrFile ? 'active' : 'inactive'}`}>
          <span className="status-dot"></span>
          DocTR JSON: {doctrFile ? doctrFile.name : 'Not selected'}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          <div className="error-content">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <circle cx="12" cy="16" r="1" fill="currentColor" />
            </svg>
            {error}
          </div>
          <button className="error-close" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* Results Dashboard */}
      {benchmarkResult && currentModelData && (
        <div className="dashboard">
          {/* Document Info */}
          <div className="doc-info">
            <div className="doc-badges">
              <div className="badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4h16v16H4z" />
                  <line x1="8" y1="8" x2="16" y2="8" />
                  <line x1="8" y1="12" x2="16" y2="12" />
                  <line x1="8" y1="16" x2="12" y2="16" />
                </svg>
                {benchmarkResult.document?.name || "document.pdf"}
              </div>
              <div className="badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 8v4l3 3" />
                </svg>
                {currentModelData.metrics?.processing_time?.display || "0ms"}
              </div>
            </div>
            <div className="badge highlight">
              <span>Model: {selectedModel}</span>
            </div>
          </div>

          {/* Stats Row */}
          <div className="stats-row">
            {/* Overall Accuracy Card */}
            <div className="accuracy-card">
              <div className="accuracy-header">
                <span>Overall Accuracy</span>
                <span className={`status-badge ${getMetricClass(currentModelData.metrics?.overall_accuracy?.value)}`}>
                  {currentModelData.metrics?.overall_accuracy?.status || 'N/A'}
                </span>
              </div>
              <div className="accuracy-value">
                <div className="accuracy-number">{currentModelData.metrics?.overall_accuracy?.value || 0}%</div>
                <div className="accuracy-label">Semantic Similarity</div>
              </div>
              <div className="accuracy-footer">
                <span>Reference: Textract</span>
                <span>vs DocTR</span>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="metrics-grid">
              <div className={`metric-item ${getMetricClass(currentModelData.metrics?.semantic_similarity?.value)}`}>
                <div className="metric-icon">🎯</div>
                <div className="metric-content">
                  <h4>Semantic Similarity</h4>
                  <div className="metric-main">
                    <span className="value">{currentModelData.metrics?.semantic_similarity?.value || 0}</span>
                    <span className="unit">%</span>
                  </div>
                  <div className="metric-progress">
                    <div className="progress-bar" style={{ width: `${currentModelData.metrics?.semantic_similarity?.value || 0}%` }} />
                  </div>
                </div>
              </div>

              <div className={`metric-item ${getMetricClass(currentModelData.metrics?.word_error_rate?.value, 'error')}`}>
                <div className="metric-icon">📝</div>
                <div className="metric-content">
                  <h4>Word Error Rate</h4>
                  <div className="metric-main">
                    <span className="value">{currentModelData.metrics?.word_error_rate?.value || 0}</span>
                    <span className="unit">%</span>
                  </div>
                  <div className="metric-progress">
                    <div className="progress-bar" style={{ width: `${currentModelData.metrics?.word_error_rate?.value || 0}%` }} />
                  </div>
                </div>
              </div>

              <div className={`metric-item ${getMetricClass(currentModelData.metrics?.character_error_rate?.value, 'error')}`}>
                <div className="metric-icon">🔤</div>
                <div className="metric-content">
                  <h4>Character Error Rate</h4>
                  <div className="metric-main">
                    <span className="value">{currentModelData.metrics?.character_error_rate?.value || 0}</span>
                    <span className="unit">%</span>
                  </div>
                  <div className="metric-progress">
                    <div className="progress-bar" style={{ width: `${currentModelData.metrics?.character_error_rate?.value || 0}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Comparison Table */}
          {currentModelData.detailed_comparison && currentModelData.detailed_comparison.length > 0 ? (
            <div className="table-section">
              <div className="table-header">
                <h3>Detailed Comparison</h3>
                <div className="table-stats">
                  <div className="table-stat">
                    <span className="dot exact" />
                    <span>Exact Match</span>
                  </div>
                  <div className="table-stat">
                    <span className="dot similar" />
                    <span>Similar</span>
                  </div>
                  <div className="table-stat">
                    <span className="dot different" />
                    <span>Different</span>
                  </div>
                </div>
              </div>

              <div className="comparison-table">
                <div className="table-row header">
                  <div className="table-cell">Textract</div>
                  <div className="table-cell">Similarity</div>
                  <div className="table-cell">DocTR</div>
                  <div className="table-cell">Status</div>
                </div>
                <div className="comparison-rows">
                  {currentModelData.detailed_comparison.map((item, index) => {
                    let status = 'different';
                    let matchClass = '';
                    let matchIndicator = '';
                    
                    if (item.match_status === 'exact' || item.similarity_score >= 95) {
                      status = 'exact';
                      matchClass = 'exact';
                      matchIndicator = '✓';
                    } else if (item.match_status === 'similar' || item.similarity_score >= 70) {
                      status = 'similar';
                      matchClass = 'similar';
                      matchIndicator = '●';
                    } else {
                      status = 'different';
                      matchClass = 'different';
                      matchIndicator = '✕';
                    }

                    return (
                      <div key={index} className={`table-row ${status}`}>
                        <div className="table-cell">{item.textract_text || '—'}</div>
                        <div className="table-cell">
                          <div className="similarity-score">{item.similarity_score}%</div>
                        </div>
                        <div className="table-cell">{item.doctr_text || '—'}</div>
                        <div className="table-cell">
                          <div className={`match-indicator ${matchClass}`}>
                            {matchIndicator}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-table-message">
              <p>No detailed comparison available for this model.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default OCRBenchmark;