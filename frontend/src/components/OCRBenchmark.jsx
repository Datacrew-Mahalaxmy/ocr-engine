// OCRBenchmark.jsx - Main React component for the OCR Benchmarking Dashboard
// OCRBenchmark.jsx
import React, { useState, useCallback, useEffect } from 'react';
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
  const [selectedModel, setSelectedModel] = useState("all-MiniLM-L6-v2");
  const [availableModels, setAvailableModels] = useState([]);

  const onTextractDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setTextractFile(acceptedFiles[0]);
    }
  }, []);

  const onDoctrDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setDoctrFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps: getTextractRootProps, getInputProps: getTextractInputProps, isDragActive: isTextractDragActive } = useDropzone({
    onDrop: onTextractDrop,
    accept: { 'application/json': ['.json'] },
    maxFiles: 1
  });

  const { getRootProps: getDoctrRootProps, getInputProps: getDoctrInputProps, isDragActive: isDoctrDragActive } = useDropzone({
    onDrop: onDoctrDrop,
    accept: { 'application/json': ['.json'] },
    maxFiles: 1
  });

  const handleCompare = async () => {
    if (!textractFile || !doctrFile) {
      setError('Please upload both JSON files');
      return;
    }

    setProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('textract_json', textractFile);
    formData.append('doctr_json', doctrFile);
    formData.append('model_name', selectedModel);

    try {
      const response = await axios.post(`${API_BASE}/compare-with-textract`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000, // 5 minutes
      });
      
      const data = response.data;
      setResult(data);
      setAvailableModels(data.available_models || []);
      setSelectedModel(data.selected_model || "all-MiniLM-L6-v2");
      
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Comparison failed');
    } finally {
      setProcessing(false);
    }
  };

  const removeFile = (type) => {
    if (type === 'textract') setTextractFile(null);
    if (type === 'doctr') setDoctrFile(null);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const getMetricClass = (value, type) => {
    if (type === 'error') {
      if (value <= 10) return '';
      if (value <= 20) return 'warning';
      return 'danger';
    }
    if (value >= 80) return '';
    if (value >= 60) return 'warning';
    return 'danger';
  };

  // Get current model's data
  const currentModelData = result?.models?.[selectedModel];

  return (
    <div className="ocr-benchmark">
      <header className="benchmark-header">
        <h1>
          OCR Benchmark
          <span>Dashboard</span>
        </h1>
      </header>

      {/* Upload Section */}
      <div className="upload-grid">
        {/* Textract Upload */}
        <div className="upload-card">
          <div className="upload-header">
            <div className="upload-icon-wrapper">📄</div>
            <div>
              <h3>Textract JSON</h3>
              <p>Reference document</p>
            </div>
          </div>
          <div
            {...getTextractRootProps()}
            className={`dropzone ${isTextractDragActive ? 'active' : ''} ${textractFile ? 'has-file' : ''}`}
          >
            <input {...getTextractInputProps()} />
            {textractFile ? (
              <div className="file-preview">
                <div className="file-icon">📄</div>
                <div className="file-details">
                  <div className="file-name">{textractFile.name}</div>
                  <div className="file-size">{formatFileSize(textractFile.size)}</div>
                </div>
                <button 
                  className="remove-file"
                  onClick={(e) => { e.stopPropagation(); removeFile('textract'); }}
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="dropzone-content">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 16v-8M8 12l4-4 4 4" />
                  <path d="M20 16v4H4v-4" />
                </svg>
                <p>Drop your Textract JSON here</p>
                <span className="small">or click to browse</span>
              </div>
            )}
          </div>
        </div>

        {/* DocTR Upload */}
        <div className="upload-card">
          <div className="upload-header">
            <div className="upload-icon-wrapper">🤖</div>
            <div>
              <h3>DocTR JSON</h3>
              <p>Candidate document</p>
            </div>
          </div>
          <div
            {...getDoctrRootProps()}
            className={`dropzone ${isDoctrDragActive ? 'active' : ''} ${doctrFile ? 'has-file' : ''}`}
          >
            <input {...getDoctrInputProps()} />
            {doctrFile ? (
              <div className="file-preview">
                <div className="file-icon">🤖</div>
                <div className="file-details">
                  <div className="file-name">{doctrFile.name}</div>
                  <div className="file-size">{formatFileSize(doctrFile.size)}</div>
                </div>
                <button 
                  className="remove-file"
                  onClick={(e) => { e.stopPropagation(); removeFile('doctr'); }}
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="dropzone-content">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 16v-8M8 12l4-4 4 4" />
                  <path d="M20 16v4H4v-4" />
                </svg>
                <p>Drop your DocTR JSON here</p>
                <span className="small">or click to browse</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Selector */}
      {result && availableModels.length > 0 && (
        <div className="model-section">
          <div className="model-label">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 18v-4M12 8v4" />
            </svg>
            Select Model to Display
          </div>
          <div className="model-select-wrapper">
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)} 
              disabled={processing}
            >
              {availableModels.map(model => (
                <option key={model} value={model}>
                  {model} {result.model_info?.[model]?.description ? `- ${result.model_info[model].description}` : ''}
                </option>
              ))}
            </select>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </div>
      )}

      {/* Compare Button */}
      <button className="compare-btn" onClick={handleCompare} disabled={!textractFile || !doctrFile || processing}>
        {processing ? (
          <>
            <svg className="loading-spinner" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v2M12 12v2M12 18v2" />
            </svg>
            Computing Similarity...
          </>
        ) : (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Compare Documents
          </>
        )}
      </button>

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
      {result && currentModelData && (
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
                {result.document?.name || "document.pdf"}
              </div>
              <div className="badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 8v4l3 3" />
                </svg>
                {currentModelData.metrics?.processing_time?.display || "0ms"}
              </div>
            </div>
            <div className="badge">
              Model: {selectedModel}
            </div>
          </div>

          {/* Stats Row */}
          <div className="stats-row">
            {/* Overall Accuracy Card */}
            <div className="accuracy-card">
              <div className="accuracy-header">
                <span>Overall Accuracy</span>
                <span className="status-badge">{currentModelData.metrics?.overall_accuracy?.status || 'N/A'}</span>
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