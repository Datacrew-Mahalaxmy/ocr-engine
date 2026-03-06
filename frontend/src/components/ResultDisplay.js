// ResultDisplay.jsx
import React, { useEffect, useState } from 'react';

const ResultDisplay = ({ result, processing }) => {
  const [activeTab, setActiveTab] = useState('statement');
  const [voiceMessage, setVoiceMessage] = useState('');

  useEffect(() => {
    if (result) {
      generateVoiceStatement();
    }
  }, [result]);

  const generateVoiceStatement = () => {
    const confidence = (result.confidence * 100).toFixed(1);
    const expected = result.expectedDisplay;
    const detected = result.actualDisplay;
    
    const allText = Object.values(result.extractedText || {}).join(' ');
    const wordCount = allText.split(/\s+/).filter(word => word.length > 0).length;
    
    let statement = '';
    
    if (result.isMatch) {
      statement = `✅ DOCUMENT VERIFIED. Expected ${expected} and detected ${detected} with ${confidence}% confidence. ${wordCount} words extracted from ${result.totalRegions || 0} regions.`;
    } else {
      statement = `❌ DOCUMENT REJECTED. Expected ${expected} but detected ${detected} with only ${confidence}% confidence. ${wordCount} words extracted from ${result.totalRegions || 0} regions.`;
    }
    
    setVoiceMessage(statement);
  };

  if (processing) {
    return (
      <div className="processing-message">
        <div className="processing-icon">🔍</div>
        <p>Processing document... Please wait.</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="empty-message">
        <p>👆 Upload a document to see analysis results</p>
      </div>
    );
  }

  const allText = Object.values(result.extractedText || {}).join(' ');
  const wordCount = allText.split(/\s+/).filter(word => word.length > 0).length;

  return (
    <div className="result-container">
      {/* Tab Navigation */}
      <div className="result-tabs-minimal">
        <button 
          className={`tab-btn-minimal ${activeTab === 'statement' ? 'active' : ''}`}
          onClick={() => setActiveTab('statement')}
        >
          <span>🤖</span> Machine Statement
        </button>
        <button 
          className={`tab-btn-minimal ${activeTab === 'text' ? 'active' : ''}`}
          onClick={() => setActiveTab('text')}
        >
          <span>📝</span> Extracted Text ({wordCount} words)
        </button>
      </div>

      {/* Machine Statement Tab */}
      {activeTab === 'statement' && (
        <div className="statement-container">
          <div className={`statement-badge ${result.isMatch ? 'success' : 'error'}`}>
            {result.isMatch ? '✓ VERIFIED' : '✗ REJECTED'}
          </div>
          <div className="statement-content">
            <div className="machine-icon-large">🤖</div>
            <p className="machine-statement">{voiceMessage}</p>
          </div>
        </div>
      )}

      {/* Extracted Text Tab */}
      {activeTab === 'text' && (
        <div className="extracted-text-container">
          <div className="text-header-minimal">
            <h3>📄 Extracted Text</h3>
            <span className="text-stats-minimal">{wordCount} words • {allText.length} characters</span>
          </div>
          
          <div className="text-content-minimal">
            {Object.keys(result.extractedText || {}).length > 0 ? (
              Object.entries(result.extractedText).map(([page, text]) => (
                <div key={page} className="page-card-minimal">
                  <div className="page-header-minimal">
                    <span className="page-number-minimal">Page {page}</span>
                    <span className="word-count-minimal">{text.split(/\s+/).filter(w => w.length > 0).length} words</span>
                  </div>
                  <pre className="ocr-text-minimal">{text}</pre>
                </div>
              ))
            ) : (
              <div className="empty-text-minimal">
                <p>No text extracted from this document</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;