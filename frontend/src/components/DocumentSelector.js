/* DocumentSelector.js - React component for selecting expected document type */
import React from 'react';

const DOCUMENT_TYPES = [
  { value: 'aadhaar_card', label: 'Aadhaar Card' },
  { value: 'aadhaar_consent', label: 'Aadhaar Consent' },
  { value: 'asset_details', label: 'Asset Details' },
  { value: 'bank_statement', label: 'Bank Statement' },
  { value: 'cibil_report', label: 'CIBIL Report' },
  { value: 'credit_appraisal_memo', label: 'Credit Appraisal Memo' },
  { value: 'disbursement_report', label: 'Disbursement Report' },
  { value: 'driving_licence', label: 'Driving Licence' },
  { value: 'family_tree', label: 'Family Tree' },
  { value: 'foreclosure_letter', label: 'Foreclosure Letter' },
  { value: 'form_sixty', label: 'Form 60' },
  { value: 'gst_certificate', label: 'GST Certificate' },
  { value: 'income_calculation', label: 'Income Calculation' },
  { value: 'insurance_form', label: 'Insurance Form' },
  { value: 'legal_report', label: 'Legal Report' },
  { value: 'mail', label: 'Mail' },
  { value: 'modt', label: 'MODT' },
  { value: 'muthoot_one_app', label: 'Muthoot One App' },
  { value: 'nach', label: 'NACH' },
  { value: 'pan_card', label: 'PAN Card' },
  { value: 'pd_document', label: 'PD Document' },
  { value: 'penny_drop', label: 'Penny Drop' },
  { value: 'property_verification', label: 'Property Verification' },
  { value: 'rcu_sampling_report', label: 'RCU Sampling Report' },
  { value: 'request_for_disbursal', label: 'Request for Disbursal' },
  { value: 'technical_report', label: 'Technical Report' },
  { value: 'trade_license', label: 'Trade License' },
  { value: 'udyam_registration', label: 'Udyam Registration' },
  { value: 'vernacular', label: 'Vernacular' },
  { value: 'vetting_report', label: 'Vetting Report' }
];

const DocumentSelector = ({ selectedDoc, onSelect, disabled }) => {
  return (
    <div className="selector-container">
      <label htmlFor="document-type">Select Expected Document Type</label>
      <select
        id="document-type"
        value={selectedDoc}
        onChange={(e) => onSelect(e.target.value)}
        disabled={disabled}
        className="document-select"
      >
        <option value="">-- Select a document type --</option>
        {DOCUMENT_TYPES.map((doc) => (
          <option key={doc.value} value={doc.value}>
            {doc.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default DocumentSelector;