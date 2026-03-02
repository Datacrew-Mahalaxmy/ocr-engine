"""
app.py - Simple Streamlit frontend for Document Identification
Shows only YES/NO result - no confidence scores or details
"""

import streamlit as st
import requests
from PIL import Image
import io
import time
from pathlib import Path

# API configuration
API_URL = "http://localhost:5001"

st.set_page_config(
    page_title="Document Identifier",
    page_icon="✅",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-yes {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 3rem;
        border-radius: 10px;
        text-align: center;
        font-size: 4rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .result-no {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 3rem;
        border-radius: 10px;
        text-align: center;
        font-size: 4rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .comparison-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .doc-badge {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ALL DOCUMENT TYPES - Flat list
ALL_DOCUMENT_TYPES = [
    "aadhaar_card", "aadhaar_consent", "asset_details", "bank_statement",
    "cibil_report", "credit_appraisal_memo", "disbursement_report", "driving_licence",
    "family_tree", "foreclosure_letter", "form_sixty", "gst_certificate",
    "income_calculation", "insurance_form", "legal_report", "mail",
    "modt", "muthoot_one_app", "nach", "pan_card", "pd_document",
    "penny_drop", "rcu_sampling_report", "property_verification",
    "request_for_disbursal", "technical_report", "trade_license",
    "udyam_registration", "vernacular", "vetting_report"
]

# Display names
DISPLAY_NAMES = {
    "aadhaar_card": "Aadhaar Card",
    "aadhaar_consent": "Aadhaar Consent",
    "asset_details": "Asset Details",
    "bank_statement": "Bank Statement",
    "cibil_report": "CIBIL Report",
    "credit_appraisal_memo": "Credit Appraisal Memo",
    "disbursement_report": "Disbursement Report",
    "driving_licence": "Driving Licence",
    "family_tree": "Family Tree",
    "foreclosure_letter": "Foreclosure Letter",
    "form_sixty": "Form 60",
    "gst_certificate": "GST Certificate",
    "income_calculation": "Income Calculation",
    "insurance_form": "Insurance Form",
    "legal_report": "Legal Report",
    "mail": "Mail",
    "modt": "MODT",
    "muthoot_one_app": "Muthoot One App",
    "nach": "NACH",
    "pan_card": "PAN Card",
    "pd_document": "PD Document",
    "penny_drop": "Penny Drop",
    "property_verification": "Property Verification",
    "rcu_sampling_report": "RCU Sampling Report",
    "request_for_disbursal": "Request for Disbursal",
    "technical_report": "Technical Report",
    "trade_license": "Trade License",
    "udyam_registration": "Udyam Registration",
    "vernacular": "Vernacular",
    "vetting_report": "Vetting Report"
}

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'identification_result' not in st.session_state:
    st.session_state.identification_result = None
if 'error' not in st.session_state:
    st.session_state.error = None

def check_api():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def identify_document(file, expected_type):
    """Upload and identify document"""
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        params = {'engine': 'doctr'}
        
        response = requests.post(
            f"{API_URL}/upload", 
            files=files,
            params=params,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Get classification
            classification = result.get('classification', {})
            actual_type = classification.get('raw_type', 'unknown')
            
            # Determine if match
            is_match = (actual_type == expected_type)
            
            return {
                'success': True,
                'is_match': is_match,
                'expected_type': expected_type,
                'expected_display': DISPLAY_NAMES.get(expected_type, expected_type.replace('_', ' ').title()),
                'actual_type': actual_type,
                'actual_display': DISPLAY_NAMES.get(actual_type, actual_type.replace('_', ' ').title())
            }
        else:
            return {'error': f'Server error: {response.status_code}'}
                
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to API'}
    except Exception as e:
        return {'error': str(e)}

# Header
st.markdown('<div class="main-header"><h1>✅ Document Identifier</h1><p>Upload → Select → Get YES/NO</p></div>', 
            unsafe_allow_html=True)

# Check API connection
if not check_api():
    st.markdown("""
    <div class="error-box">
        <h4>⚠️ Cannot connect to OCR API</h4>
        <p>Please make sure the Flask API is running: <code>python api.py</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Retry"):
        st.rerun()
    st.stop()

# Main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📤 Upload Document")
    
    # Document type selector
    selected_raw = st.selectbox(
        "Select Expected Document Type",
        options=ALL_DOCUMENT_TYPES,
        format_func=lambda x: DISPLAY_NAMES.get(x, x.replace('_', ' ').title())
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF or image",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )
    
    if uploaded_file:
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        
        # Identify button
        if st.button("🔍 Identify Document", type="primary", use_container_width=True):
            st.session_state.processing = True
            st.session_state.identification_result = None
            st.session_state.error = None
            
            with st.spinner("Processing..."):
                result = identify_document(uploaded_file, selected_raw)
                
                if 'error' in result:
                    st.session_state.error = result['error']
                else:
                    st.session_state.identification_result = result
                
                st.session_state.processing = False
                st.rerun()
    
    # Show error if any
    if st.session_state.error:
        st.markdown(f'<div class="error-box">❌ {st.session_state.error}</div>', 
                   unsafe_allow_html=True)
        if st.button("Clear Error"):
            st.session_state.error = None
            st.session_state.identification_result = None
            st.rerun()

with col_right:
    st.markdown("### ✅ Result")
    
    if st.session_state.identification_result:
        result = st.session_state.identification_result
        
        # Show YES/NO result
        if result['is_match']:
            st.markdown("""
            <div class="result-yes">
                ✓ YES
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-no">
                ✗ NO
            </div>
            """, unsafe_allow_html=True)
        
        # Show simple comparison
        st.markdown('<div class="comparison-box">', unsafe_allow_html=True)
        st.markdown("**You Selected:**")
        st.markdown(f'<span class="doc-badge">{result["expected_display"]}</span>', unsafe_allow_html=True)
        
        st.markdown("**System Detected:**")
        st.markdown(f'<span class="doc-badge">{result["actual_display"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.processing:
        st.info("⏳ Processing document...")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p style="font-size: 3rem;">📄 → ✅</p>
            <p>Upload a document to see result</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Document Identifier | YES/NO Only
</div>
""", unsafe_allow_html=True)