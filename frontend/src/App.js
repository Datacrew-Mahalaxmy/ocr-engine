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
  // Document Identifier State
  const [selectedDoc, setSelectedDoc] = useState('');
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // OCR Benchmark State
  const [textractFile, setTextractFile] = useState(null);
  const [doctrFile, setDoctrFile] = useState(null);
  const [benchmarkProcessing, setBenchmarkProcessing] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState(null);
  const [benchmarkError, setBenchmarkError] = useState(null);

  // Handle Document Identification
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
        headers: { 'Content-Type': 'multipart/form-data' },
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

  // Handle OCR Benchmark Comparison - Now receives model from OCRBenchmark
  const handleCompare = async (selectedModel) => {
    if (!textractFile || !doctrFile) {
      setBenchmarkError('Please upload both JSON files');
      return;
    }

    setBenchmarkProcessing(true);
    setBenchmarkError(null);
    setBenchmarkResult(null);

    const formData = new FormData();
    formData.append('textract_json', textractFile);
    formData.append('doctr_json', doctrFile);
    formData.append('model_name', selectedModel);

    try {
      const response = await axios.post(`${API_BASE}/compare-with-textract`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      });
      
      const data = response.data;
      setBenchmarkResult(data);
      
    } catch (err) {
      setBenchmarkError(err.response?.data?.detail || err.message || 'Comparison failed');
    } finally {
      setBenchmarkProcessing(false);
    }
  };

  // Helper Functions
  const formatDocName = (name) => {
    if (!name) return 'Unknown';
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  const downloadJSON = () => {
    if (!result) return;
    
    const jsonString = JSON.stringify(result.rawResults, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `ocr_result_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const downloadBenchmarkJSON = () => {
    if (!benchmarkResult) return;
    
    const jsonString = JSON.stringify(benchmarkResult, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `benchmark_results_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const removeFile = (type) => {
    if (type === 'textract') setTextractFile(null);
    if (type === 'doctr') setDoctrFile(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>Document Intelligence Platform</h1>
          <p>
            <span>✨</span>
            Advanced OCR Analysis & Benchmarking
            <span>✨</span>
          </p>
        </div>
      </header>

      {/* Two Column Layout */}
      <div className="dashboard-grid">
        {/* LEFT COLUMN - Document Identifier */}
        <div className="identifier-section">
          <div className="section-header">
            <h2>
              <span>📄</span>
              Document Identifier
            </h2>
            {result && (
              <button className="download-btn" onClick={downloadJSON}>
                <span>⬇️</span>
                Download JSON
              </button>
            )}
          </div>

          <div className="section-content">
            {/* Document Type Selection - Using DocumentSelector component which has its own label */}
            <div className="input-group">
              <DocumentSelector 
                selectedDoc={selectedDoc}
                onSelect={setSelectedDoc}
                disabled={processing}
              />
            </div>

            {/* File Upload */}
            <div className="input-group">
              <label className="input-label">UPLOAD DOCUMENT</label>
              <FileUpload 
                onFileSelect={setFile}
                disabled={processing}
              />
            </div>

            {/* Show selected file */}
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

            {/* Identify Button */}
            <button 
              className="identify-btn"
              onClick={handleIdentify}
              disabled={!file || !selectedDoc || processing}
            >
              {processing ? (
                <>
                  <span className="spinner"></span>
                  Processing...
                </>
              ) : (
                <>
                  <span>🔍</span>
                  IDENTIFY DOCUMENT
                </>
              )}
            </button>

            {/* Show any errors */}
            {error && (
              <div className="error-message">
                <div className="error-content">
                  <span>❌</span>
                  {error}
                </div>
                <button onClick={() => setError(null)}>✕</button>
              </div>
            )}

            {/* Show Results */}
            <ResultDisplay 
              result={result} 
              processing={processing} 
            />
          </div>
        </div>

        {/* RIGHT COLUMN - OCR Benchmark */}
        <div className="benchmark-section">
          <div className="section-header">
            <h2>
              <span>📊</span>
              OCR Benchmark
            </h2>
            {benchmarkResult && (
              <button className="download-btn" onClick={downloadBenchmarkJSON}>
                <span>⬇️</span>
                Download JSON
              </button>
            )}
          </div>

          <div className="section-content">
            {/* Textract JSON Upload */}
            <div className="upload-group">
              <label>
                <span>📄</span>
                TEXTRACT JSON
              </label>
              <div className="file-input-wrapper">
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => setTextractFile(e.target.files[0])}
                  className="file-input"
                  id="textract-upload"
                />
                <label htmlFor="textract-upload" className="file-input-button">
                  Choose File
                </label>
                <span className="file-input-text">
                  {textractFile ? textractFile.name : 'No file chosen'}
                </span>
              </div>
              {textractFile && (
                <div className="file-info">
                  <span className="file-name">{textractFile.name}</span>
                  <span className="file-size">{(textractFile.size / 1024).toFixed(1)} KB</span>
                  <button className="remove-btn" onClick={() => removeFile('textract')}>✕</button>
                </div>
              )}
            </div>

            {/* DocTR JSON Upload */}
            <div className="upload-group">
              <label>
                <span>🤖</span>
                DOCTR JSON
              </label>
              <div className="file-input-wrapper">
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => setDoctrFile(e.target.files[0])}
                  className="file-input"
                  id="doctr-upload"
                />
                <label htmlFor="doctr-upload" className="file-input-button">
                  Choose File
                </label>
                <span className="file-input-text">
                  {doctrFile ? doctrFile.name : 'No file chosen'}
                </span>
              </div>
              {doctrFile && (
                <div className="file-info">
                  <span className="file-name">{doctrFile.name}</span>
                  <span className="file-size">{(doctrFile.size / 1024).toFixed(1)} KB</span>
                  <button className="remove-btn" onClick={() => removeFile('doctr')}>✕</button>
                </div>
              )}
            </div>

            {/* OCRBenchmark Component - Now handles model selection and comparison */}
            <OCRBenchmark 
              result={benchmarkResult}
              processing={benchmarkProcessing}
              onCompare={handleCompare}
              textractFile={textractFile}
              doctrFile={doctrFile}
              error={benchmarkError}
              setError={setBenchmarkError}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <span>⚡ Document Intelligence Platform v2.0</span>
      </footer>
    </div>
  );
}

export default App;