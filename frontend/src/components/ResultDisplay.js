// ResultDisplay.js
import React from 'react';

const ResultDisplay = ({ result, processing, onDownload }) => {
  if (processing) {
    return (
      <div className="result-placeholder">
        <div className="spinner"></div>
        <p>Processing document...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-placeholder">
        <span className="placeholder-icon">📄</span>
        <p>Upload a document to see result</p>
      </div>
    );
  }

  return (
    <div className="result-container">
      <div className={`result-banner ${result.isMatch ? 'yes' : 'no'}`}>
        {result.isMatch ? '✓ YES' : '✗ NO'}
      </div>

      <div className="comparison-box">
        <div className="comparison-item">
          <p className="label">You Selected:</p>
          <span className="badge">{result.expectedDisplay}</span>
        </div>
        
        <div className="comparison-item">
          <p className="label">System Detected:</p>
          <span className="badge">{result.actualDisplay}</span>
        </div>
        
        <div className="comparison-item">
          <p className="label">Confidence:</p>
          <span className="badge" style={{background: result.confidence > 0.7 ? '#28a745' : '#ffc107'}}>
            {Math.round(result.confidence * 100)}%
          </span>
        </div>
      </div>

      {/* Download Button */}
      <div className="download-section">
        <button 
          className="download-btn"
          onClick={onDownload}
          title="Download JSON file"
        >
          ⬇️ Download JSON
        </button>
      </div>

      {/* Extracted Text Section */}
      {result.extractedText && Object.keys(result.extractedText).length > 0 && (
        <div className="extracted-text-box">
          <h3>📝 Extracted Text</h3>
          <p className="text-info">Total text regions found: {result.totalRegions}</p>
          
          {Object.entries(result.extractedText).map(([page, text]) => (
            <div key={page} className="page-text">
              <h4>Page {page}</h4>
              <pre className="ocr-text">{text}</pre>
            </div>
          ))}
        </div>
      )}

      {/* Show warning if low confidence */}
      {result.confidence < 0.4 && (
        <div className="warning-box">
          ⚠️ Low confidence detection - please verify manually
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;