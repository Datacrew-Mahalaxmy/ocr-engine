// OCRBenchmark.jsx
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './OCRBenchmark.css';

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const OCRBenchmark = () => {
  const [textractFile, setTextractFile] = useState(null);
  const [doctrFile, setDoctrFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelName, setModelName] = useState("all-MiniLM-L6-v2");

  // Textract JSON dropzone
  const onTextractDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setTextractFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps: getTextractRootProps, getInputProps: getTextractInputProps, isDragActive: isTextractDragActive } = useDropzone({
    onDrop: onTextractDrop,
    accept: {
      'application/json': ['.json']
    },
    maxFiles: 1
  });

  // DocTR JSON dropzone
  const onDoctrDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setDoctrFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps: getDoctrRootProps, getInputProps: getDoctrInputProps, isDragActive: isDoctrDragActive } = useDropzone({
    onDrop: onDoctrDrop,
    accept: {
      'application/json': ['.json']
    },
    maxFiles: 1
  });

  const handleCompare = async () => {
    if (!textractFile || !doctrFile) {
      setError('Please upload both Textract and DocTR JSON files');
      return;
    }

    setProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('textract_json', textractFile);
    formData.append('doctr_json', doctrFile);
    formData.append('model_name', modelName);

    try {
      const response = await axios.post(`${API_BASE}/compare-with-textract`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes
      });

      setResult(response.data);
    } catch (err) {
      console.error('Comparison error:', err);
      setError(err.response?.data?.detail || err.message || 'Comparison failed');
    } finally {
      setProcessing(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 90) return '#28a745';
    if (score >= 70) return '#ffc107';
    return '#dc3545';
  };

  const getMatchIcon = (status) => {
    switch(status) {
      case 'exact': return '✅';
      case 'similar': return '🟡';
      case 'different': return '❌';
      case 'missing': return '⚠️';
      default: return '•';
    }
  };

  return (
    <div className="ocr-benchmark">
      <header className="benchmark-header">
        <h1>📊 OCR Benchmark Dashboard</h1>
      </header>

      <div className="upload-section">
        <div className="upload-card">
          <h3>📄 Textract JSON (Reference)</h3>
          <div 
            {...getTextractRootProps()} 
            className={`dropzone ${isTextractDragActive ? 'active' : ''} ${textractFile ? 'has-file' : ''}`}
          >
            <input {...getTextractInputProps()} />
            {textractFile ? (
              <div className="file-info">
                <span className="file-icon">📄</span>
                <span className="file-name">{textractFile.name}</span>
              </div>
            ) : (
              <div className="dropzone-content">
                <span className="upload-icon">📤</span>
                <p>Drop Textract JSON here</p>
                <p className="small">or click to browse</p>
              </div>
            )}
          </div>
        </div>

        <div className="upload-card">
          <h3>🤖 DocTR JSON (Candidate)</h3>
          <div 
            {...getDoctrRootProps()} 
            className={`dropzone ${isDoctrDragActive ? 'active' : ''} ${doctrFile ? 'has-file' : ''}`}
          >
            <input {...getDoctrInputProps()} />
            {doctrFile ? (
              <div className="file-info">
                <span className="file-icon">🤖</span>
                <span className="file-name">{doctrFile.name}</span>
              </div>
            ) : (
              <div className="dropzone-content">
                <span className="upload-icon">📤</span>
                <p>Drop DocTR JSON here</p>
                <p className="small">or click to browse</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="model-selector">
        <label>Sentence-BERT Model:</label>
        <select value={modelName} onChange={(e) => setModelName(e.target.value)} disabled={processing}>
          <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Fast)</option>
          <option value="all-mpnet-base-v2">all-mpnet-base-v2 (Accurate)</option>
          <option value="multi-qa-mpnet-base-dot-v1">multi-qa-mpnet-base (Best)</option>
        </select>
      </div>

      <button 
        className="compare-btn"
        onClick={handleCompare}
        disabled={!textractFile || !doctrFile || processing}
      >
        {processing ? '⏳ Computing Similarity...' : '🔍 Compare with SBERT'}
      </button>

      {error && (
        <div className="error-message">
          ❌ {error}
          <button onClick={() => setError(null)}>Clear</button>
        </div>
      )}

      {result && (
        <div className="dashboard">
          {/* Document Info */}
          <div className="doc-info">
            <span className="doc-badge">📄 {result.document.name || 'Document'}</span>
            <span className="doc-badge">📊 Reference: {result.document.reference}</span>
            <span className="doc-badge">🤖 Engine: {result.document.engine}</span>
          </div>

          {/* Overall Accuracy */}
          <div className="overall-accuracy" style={{ backgroundColor: getScoreColor(result.metrics.overall_accuracy.value) }}>
            <div className="accuracy-label">Overall Accuracy</div>
            <div className="accuracy-value">{result.metrics.overall_accuracy.value}%</div>
            <div className="accuracy-status">[{result.metrics.overall_accuracy.status}]</div>
          </div>

          {/* Metrics Grid */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-title">Semantic Similarity</div>
              <div className="metric-value" style={{ color: getScoreColor(result.metrics.semantic_similarity.value) }}>
                {result.metrics.semantic_similarity.display}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Word Error Rate</div>
              <div className="metric-value" style={{ color: getScoreColor(100 - result.metrics.word_error_rate.value) }}>
                {result.metrics.word_error_rate.display}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Character Error Rate</div>
              <div className="metric-value" style={{ color: getScoreColor(100 - result.metrics.character_error_rate.value) }}>
                {result.metrics.character_error_rate.display}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Processing Time</div>
              <div className="metric-value">{result.metrics.processing_time.display}</div>
            </div>
          </div>

          {/* Detailed Comparison */}
          <div className="detailed-comparison">
            <h3>📋 Detailed Comparison</h3>
            
            <div className="comparison-header">
              <div className="header-textract">Textract (Reference)</div>
              <div className="header-similarity">Similarity</div>
              <div className="header-doctr">DocTR (Candidate)</div>
            </div>

            <div className="comparison-rows">
              {result.detailed_comparison.map((item, idx) => (
                <div key={idx} className={`comparison-row ${item.match_status}`}>
                  <div className="row-textract">
                    <span className="match-icon">{getMatchIcon(item.match_status)}</span>
                    {item.textract}
                  </div>
                  <div className="row-similarity">
                    <div className="similarity-bar-container">
                      <div 
                        className="similarity-bar" 
                        style={{ 
                          width: `${item.similarity}%`,
                          backgroundColor: getScoreColor(item.similarity)
                        }}
                      ></div>
                      <span className="similarity-text">{item.similarity}%</span>
                    </div>
                  </div>
                  <div className="row-doctr">{item.doctr}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Stats Summary */}
          <div className="stats-summary">
            <div className="stat-item">
              <span className="stat-label">Textract Blocks:</span>
              <span className="stat-value">{result.stats.textract_blocks}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">DocTR Blocks:</span>
              <span className="stat-value">{result.stats.doctr_blocks}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Matched Pairs:</span>
              <span className="stat-value">{result.stats.matched_pairs}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Unmatched:</span>
              <span className="stat-value">{result.stats.unmatched_textract}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OCRBenchmark;