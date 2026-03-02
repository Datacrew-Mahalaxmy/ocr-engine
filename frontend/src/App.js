// App.js (complete file with fixes)
import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ResultDisplay from './components/ResultDisplay';
import DocumentSelector from './components/DocumentSelector';
import OCRBenchmark from './components/OCRBenchmark';
import axios from 'axios';

const API_BASE = "http://localhost:8000";  // Change this to your actual API URL

function App() {
  const [activeTab, setActiveTab] = useState('identifier'); // 'identifier' or 'benchmark'
  const [selectedDoc, setSelectedDoc] = useState('');
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await axios.get(`${API_BASE}/health`);
      setApiStatus('connected');
    } catch (err) {
      console.error('Health check failed:', err);
      setApiStatus('disconnected');
    }
  };

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
        timeout: 600000, // 10 minutes
      });

      const data = response.data;
      console.log("API Response:", data); // For debugging
      
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
        rawResults: data, // Store the full response for download
        filename: file?.name || 'document.pdf' // Add fallback filename
      });

    } catch (err) {
      console.error("Error:", err);
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

  // Function to download JSON - FIXED VERSION
  const downloadJSON = () => {
    if (!result) return;
    
    // Create a safe filename - use a default if result.filename is undefined
    const baseFilename = result.filename || 'ocr_result';
    let safeFilename = 'ocr_result';
    
    try {
      safeFilename = baseFilename.replace(/\.[^/.]+$/, '') || 'ocr_result';
    } catch (e) {
      console.warn('Error parsing filename:', e);
      safeFilename = 'ocr_result';
    }
    
    // Create a more complete JSON with all OCR data
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
      // Include detailed results if available from API
      detailed_results: result.rawResults?.detailed_results || []
    };
    
    const jsonString = JSON.stringify(downloadData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    // Create download link
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
      <header className="header">
        <h1>📄 Document Intelligence Platform</h1>
        <p>Identify → Compare → Analyze</p>
      </header>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'identifier' ? 'active' : ''}`}
          onClick={() => setActiveTab('identifier')}
        >
          🔍 Document Identifier
        </button>
        <button 
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          📊 OCR Benchmark (SBERT)
        </button>
      </div>

      <div className="container">
        {/* API Status Banner */}
        {apiStatus === 'checking' && (
          <div className="info-banner">Checking API connection...</div>
        )}
        {apiStatus === 'disconnected' && (
          <div className="error-banner">
            ⚠️ Cannot connect to API. Make sure API is running at {API_BASE}
            <button onClick={checkApiHealth} className="retry-btn">Retry</button>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'identifier' ? (
          /* Document Identifier Tab */
          <div className="main-content">
            <div className="left-column">
              <h2>📤 Upload Document</h2>
              
              <DocumentSelector 
                selectedDoc={selectedDoc}
                onSelect={setSelectedDoc}
                disabled={processing || apiStatus !== 'connected'}
              />

              <FileUpload 
                onFileSelect={setFile}
                disabled={processing || apiStatus !== 'connected'}
              />

              {file && (
                <div className="file-info">
                  <p><strong>File:</strong> {file.name}</p>
                  <p><strong>Size:</strong> {(file.size / 1024).toFixed(1)} KB</p>
                  <p><strong>Type:</strong> {file.type}</p>
                </div>
              )}

              <button 
                className="identify-btn"
                onClick={handleIdentify}
                disabled={!file || !selectedDoc || processing || apiStatus !== 'connected'}
              >
                {processing ? '⏳ Processing...' : '🔍 Identify Document'}
              </button>

              {error && (
                <div className="error-message">
                  ❌ {error}
                  <button onClick={() => setError(null)}>Clear</button>
                </div>
              )}
            </div>

            <div className="right-column">
              <h2>✅ Result</h2>
              <ResultDisplay 
                result={result} 
                processing={processing} 
                onDownload={downloadJSON}  // Pass download function
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
        Document Intelligence Platform | Identify with Rules • Compare with SBERT
      </footer>
    </div>
  );
}

export default App;