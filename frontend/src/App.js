// App.js
import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ResultDisplay from './components/ResultDisplay';
import DocumentSelector from './components/DocumentSelector';
import OCRBenchmark from './components/OCRBenchmark';
import axios from 'axios';

const API_BASE = "http://localhost:8000";

function App() {
  const [activeTab, setActiveTab] = useState('identifier');
  const [selectedDoc, setSelectedDoc] = useState('');
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleIdentify = async () => {
    if (!file || !selectedDoc) {
      setError('Please select a document type and upload a file');
      return;
    }

    setProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/upload?engine=doctr`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 600000,
      });

      const data = response.data;
      
      const actualType = data.classification?.raw_type || 'unknown';
      const confidence = data.classification?.confidence || 0;
      
      const isMatch =
        actualType?.toLowerCase().replace(/\s+/g, "_") ===
        selectedDoc?.toLowerCase().replace(/\s+/g, "_");

      setResult({
        isMatch,
        expectedType: selectedDoc,
        expectedDisplay: formatDocName(selectedDoc),
        actualType: actualType,
        actualDisplay: formatDocName(actualType),
        confidence: confidence,
        extractedText: data.text_by_page || {},
        totalRegions: data.total_regions || 0,
        classification: data.classification || {},
        rawResults: data,
        filename: file?.name || 'document.pdf'
      });

    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to process document');
    } finally {
      setProcessing(false);
    }
  };

  const formatDocName = (name) => {
    if (!name) return 'Unknown';
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  const downloadJSON = () => {
    if (!result) return;
    
    const baseFilename = result.filename || 'ocr_result';
    let safeFilename = 'ocr_result';
    
    try {
      safeFilename = baseFilename.replace(/\.[^/.]+$/, '') || 'ocr_result';
    } catch (e) {
      console.warn('Error parsing filename:', e);
      safeFilename = 'ocr_result';
    }
    
    const downloadData = {
      metadata: {
        filename: result.filename || 'unknown',
        timestamp: new Date().toISOString(),
        expected_type: result.expectedType,
        actual_type: result.actualType,
        is_match: result.isMatch,
        confidence: result.confidence,
        total_regions: result.totalRegions || 0
      },
      classification: result.classification || {},
      text_by_page: result.extractedText || {},
      detailed_results: result.rawResults?.detailed_results || []
    };
    
    const jsonString = JSON.stringify(downloadData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${safeFilename}_ocr_results.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app">
      {/* Stunning Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>Document Intelligence Platform</h1>
            <p>
              <span></span>
              
            </p>
          </div>
          <div className="header-right">
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav-bar">
        <button 
          className={`nav-item ${activeTab === 'identifier' ? 'active' : ''}`}
          onClick={() => setActiveTab('identifier')}
        >
          <svg className="nav-icon" viewBox="0 0 24 24">
            <path d="M4 4h16v16H4z" />
            <line x1="8" y1="8" x2="16" y2="8" />
            <line x1="8" y1="12" x2="16" y2="12" />
            <line x1="8" y1="16" x2="12" y2="16" />
          </svg>
          Document Identifier
        </button>
        <button 
          className={`nav-item ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <svg className="nav-icon" viewBox="0 0 24 24">
            <path d="M21 12v4a2 2 0 01-2 2H5a2 2 0 01-2-2V6a2 2 0 012-2h7" />
            <polyline points="15 3 21 3 21 9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </svg>
          OCR Benchmark
        </button>
      </nav>

      <div className="container">
        {/* Tab Content */}
        {activeTab === 'identifier' ? (
          <div className="identifier-layout">
            {/* Left Panel - Upload */}
            <div className="left-panel">
              <div className="panel-header">
                <div className="panel-icon">📤</div>
                <h2>Upload Document</h2>
              </div>
              
              <div className="doc-selector">
                <label>Document Type</label>
                <DocumentSelector 
                  selectedDoc={selectedDoc}
                  onSelect={setSelectedDoc}
                  disabled={processing}
                />
              </div>

              <div className="upload-area">
                <label className="upload-label">Upload File</label>
                <FileUpload 
                  onFileSelect={setFile}
                  disabled={processing}
                />
              </div>

              {file && (
                <div className="file-preview">
                  <div className="file-icon">📄</div>
                  <div className="file-details">
                    <div className="file-name">{file.name}</div>
                    <div className="file-meta">
                      <span>{(file.size / 1024).toFixed(1)} KB</span>
                      <span>{file.type || 'document'}</span>
                    </div>
                  </div>
                </div>
              )}

              <button 
                className="identify-btn"
                onClick={handleIdentify}
                disabled={!file || !selectedDoc || processing}
              >
                {processing ? (
                  <>
                    <svg className="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 6v2M12 12v2M12 18v2" />
                    </svg>
                    Processing...
                  </>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 8v8M8 12h8" />
                    </svg>
                    Identify Document
                  </>
                )}
              </button>

              {error && (
                <div className="error-message">
                  <span>❌ {error}</span>
                  <button onClick={() => setError(null)}>✕</button>
                </div>
              )}
            </div>

            {/* Right Panel - Results */}
            <div className="right-panel">
              <div className="result-header">
                <h2>
                  <span>📊</span>
                  Analysis Results
                </h2>
                {result && (
                  <button className="download-btn" onClick={downloadJSON}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Download JSON
                  </button>
                )}
              </div>

              <ResultDisplay 
                result={result} 
                processing={processing} 
              />
            </div>
          </div>
        ) : (
          /* OCR Benchmark Tab */
          <div className="benchmark-container">
            <OCRBenchmark apiBase={API_BASE} />
          </div>
        )}
      </div>

      <footer className="footer">
        <span>⚡ Document Intelligence Platform v2.0</span>
        <span style={{ margin: '0 16px' }}>•</span>
        <span>Powered by Advanced AI</span>
        <span style={{ margin: '0 16px' }}>•</span>
        <span>Enterprise Grade Security</span>
      </footer>
    </div>
  );
}

export default App;