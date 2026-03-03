"""
custom_classifier.py - Enhanced with distinguishing rules for similar documents
"""

import logging
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class BusinessDocumentClassifier:
    """
    Document classifier with exactly 30 document types
    Enhanced with distinguishing rules for similar documents
    """
    
    def __init__(self):
        # ===== YOUR EXACT 30 DOCUMENTS =====
        self.DOCUMENT_SIGNATURES = {
            
            # 1. AADHAAR CARD
            'aadhaar_card': {
                'keywords': [
                    "GOVERNMENT OF INDIA",
                    "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
                    "UNIQUE IDENTIFICATION",
                    "आधार",
                    "UIDAI"
                ],
                'required': ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION"],
                'patterns': [
                    r'\d{4}\s*\d{4}\s*\d{4}',  # 12-digit Aadhaar number
                ],
                'unique_terms': ['आधार', 'UIDAI', 'AADHAAR'],  # Terms unique to this doc
                'forbidden_terms': [],  # Terms that should NOT appear
                'layout': 'id_card',
                'page_count_range': (1, 1),
                'has_photo': True,
                'weight': 3.0
            },
            
            # 2. AADHAAR CONSENT
            'aadhaar_consent': {
                'keywords': [
                    "consent of aadhaar", "aadhaar holder",
                    "voluntarily submitting my aadhaar", "kyc directions",
                    "use and share information", "offline verification"
                ],
                'required': ["consent", "aadhaar"],
                'patterns': [],
                'unique_terms': ['consent', 'voluntarily submitting', 'offline verification'],
                'forbidden_terms': ['GOVERNMENT OF INDIA', 'UIDAI'],  # Not an ID card
                'layout': 'form',
                'page_count_range': (1, 2),
                'weight': 2.0
            },
            
            # 3. ASSET DETAILS
            'asset_details': {
                'keywords': [
                    "ASSET DETAILS", "Net Worth", "asset creation", "immovable property"
                ],
                'required': ["ASSET DETAILS"],
                'patterns': [],
                'unique_terms': ['Net Worth', 'asset creation', 'immovable'],
                'forbidden_terms': ['loan', 'mortgage', 'credit'],  # Not financial
                'layout': 'table',
                'page_count_range': (1, 5),
                'weight': 2.5
            },
            
            # 4. BANK STATEMENT
            'bank_statement': {
                'keywords': [
                    "account no", "micr code", "ifsc code", "balance", "branch",
                    "account statement", "account details", "customer address",
                    "debit", "credit", "statement of account", "cif no",
                    "cheque", "savings bank", "pass book"
                ],
                'required': ["statement", "account"],
                'patterns': [
                    r'opening balance',
                    r'closing balance',
                ],
                'unique_terms': ['IFSC', 'MICR', 'CIF', 'opening balance', 'closing balance'],
                'forbidden_terms': ['CIBIL', 'credit score', 'loan approval'],  # Not credit report
                'layout': 'tabular',
                'page_count_range': (1, 50),
                'has_table': True,
                'weight': 1.5
            },
            
            # 5. CIBIL REPORT
            'cibil_report': {
                'keywords': [
                    "CIBIL", "CONSUMER CIR", "CONSUMER INFORMATION:",
                    "CIBIL TRANSUNION SCORE", "CIBIL Report", "Consumer Credit Data"
                ],
                'required': ["CIBIL", "TRANSUNION"],  # More specific
                'patterns': [
                    r'CIBIL',
                    r'Credit Score'
                ],
                'unique_terms': [
                    'CIBIL',
                    'TRANSUNION',
                    'Credit Score',
                    'CIR',
                    'Consumer Credit'
                ],
                # Penalize email patterns
                'forbidden_terms': [
                    'subject:',
                    'from:',
                    'to:',
                    'sent:',
                    'forwarded'
                ],
                'layout': 'report',
                'page_count_range': (3, 15),
                'weight': 3.0
            },
            
            # 6. CREDIT APPRAISAL MEMO
            'credit_appraisal_memo': {
                'keywords': [
                    "CREDIT APPRAISAL MEMO", "LOAN DETAILS", "TERMS OF APPROVAL",
                    "CUSTOMER PROFILE", "LOAN TENURE", "APPLICANT DETAILS"
                ],
                'required': ["CREDIT APPRAISAL", "LOAN"],
                'patterns': [],
                'unique_terms': ['APPRAISAL MEMO', 'TERMS OF APPROVAL', 'LOAN TENURE'],
                'forbidden_terms': ['opening balance', 'closing balance'],  # Not bank statement
                'layout': 'report',
                'page_count_range': (5, 30),
                'weight': 3.0
            },
            
            # 7. DISBURSEMENT REPORT
            'disbursement_report': {
                'keywords': [
                    "DISBURSAL REPORT", "Customer Details", "Application ID",
                    "Disbursal Status", "Disbursal Date", "Disbursed Amount",
                    "Product Type", "Scheme", "Asset Cost"
                ],
                'required': ["DISBURSAL", "Disbursed"],
                'patterns': [],
                'unique_terms': ['DISBURSAL', 'Disbursed Amount', 'Application ID'],
                'forbidden_terms': [],
                'layout': 'report',
                'page_count_range': (1, 10),
                'weight': 2.0
            },
            
            # 8. DRIVING LICENCE
            'driving_licence': {
                'keywords': [
                    "driving licence", "DL No", "Date of Issue", "Valid Till"
                ],
                'required': ["driving licence"],
                'patterns': [],
                'unique_terms': ['DL No', 'driving licence', 'Valid Till'],
                'forbidden_terms': ['GOVERNMENT OF INDIA', 'UIDAI'],  # Not Aadhaar
                'layout': 'id_card',
                'page_count_range': (1, 2),
                'has_photo': True,
                'weight': 3.0
            },
            
            # 9. FAMILY TREE
            'family_tree': {
                'keywords': [
                    "FAMILY TREE", "DETAILS OF SIBLINGS", "All relationship",
                    "Relationship with Property Owner"
                ],
                'required': ["FAMILY TREE", "Relationship"],
                'patterns': [],
                'unique_terms': ['FAMILY TREE', 'siblings', 'relationship with'],
                'forbidden_terms': ['account', 'balance', 'credit'],  # Not financial
                'layout': 'tree',
                'page_count_range': (1, 3),
                'weight': 3.0
            },
            
            # 10. FORECLOSURE LETTER
            'foreclosure_letter': {
                'keywords': [
                    "FORECLOSURE LETTER", "loan closure", "FORECLOSURE",
                    "status of loan", "entirely repaid", "full closure"
                ],
                'required': ["FORECLOSURE", "closure"],
                'patterns': [],
                'unique_terms': ['FORECLOSURE', 'loan closure', 'repaid', 'full closure'],
                'forbidden_terms': ['opening balance', 'statement'],  # Not bank statement
                'layout': 'letter',
                'page_count_range': (1, 3),
                'weight': 3.0
            },
            
            # 11. FORM SIXTY
            'form_sixty': {
                'keywords': [
                    "form no. 60", "form no 60", "form of declaration",
                    "are you assessed to tax", "Declaration under section 139A"
                ],
                'required': ["form no. 60", "declaration"],
                'patterns': [],
                'unique_terms': ['form no. 60', 'section 139A', 'assessed to tax'],
                'forbidden_terms': ['GST', 'UDYAM'],  # Not GST certificate
                'layout': 'form',
                'page_count_range': (1, 2),
                'weight': 3.0
            },
            
            # 12. GST CERTIFICATE
            'gst_certificate': {
                'keywords': [
                    "GST", "Form GST REG 06", "Form GSTR 3B", "Legal Name",
                    "Trade Name", "type of registration", "igst", "cgst", "sgst"
                ],
                'required': ["GST"],
                'patterns': [],
                'unique_terms': ['GST', 'GSTR', 'IGST', 'CGST', 'SGST', 'REG 06'],
                'forbidden_terms': ['PAN', 'INCOME TAX'],  # Not PAN card
                'layout': 'certificate',
                'page_count_range': (1, 2),
                'weight': 2.0
            },
            
            # 13. INCOME CALCULATION
            'income_calculation': {
                'keywords': [
                    "income calculation", "income details", "gross monthly income",
                    "net monthly income", "total income"
                ],
                'required': ["income calculation", "income"],
                'patterns': [],
                'unique_terms': ['gross monthly', 'net monthly', 'calculation'],
                'forbidden_terms': ['loan', 'mortgage', 'property'],  # Not loan doc
                'layout': 'sheet',
                'page_count_range': (1, 3),
                'weight': 3.0
            },
            
            # 14. INSURANCE FORM
            'insurance_form': {
                'keywords': [
                    "insurance proposal", "proposal form", "policy number",
                    "insurance company", "life insurance", "nominee", "sum insured"
                ],
                'required': ["insurance", "policy"],
                'patterns': [],
                'unique_terms': ['insurance', 'nominee', 'sum insured', 'policy number'],
                'forbidden_terms': ['balance', 'account'],  # Not bank statement
                'layout': 'form',
                'page_count_range': (2, 10),
                'weight': 2.5
            },
            
            # 15. LEGAL REPORT
            'legal_report': {
                'keywords': [
                    "legal due diligence report", "legal scrutiny report",
                    "Chain of title", "title search report", "legal assessment report",
                    "description of property", "list of documents scrutinised",
                    "chain of title", "name of the owner"
                ],
                'required': ["legal", "assessment"],  # Keep as is
                'patterns': [],
                # EXPANDED unique terms
                'unique_terms': [
                    'legal due diligence',
                    'legal scrutiny',
                    'legal assessment',
                    'scrutiny report',
                    'title search',
                    'list of documents scrutinised',
                    'internal legal',  # From your text: "INTERNAL LEGAL ASSESSMENT"
                    'internal report', # From your text
                    'assessment report' # From your text
                ],
                # Penalize mortgage terms
                'forbidden_terms': [
                    'memorandum of deposit',
                    'mortgage deed',
                    'equitable mortgage',
                    'section 58',
                    'transfer of property act',
                    'mortgagor',
                    'mortgagee'
                ],
                'layout': 'report',
                'page_count_range': (3, 20),
                'weight': 3.5  # INCREASED from 3.0
            },
            
            # 16. MAIL
            'mail': {
                'keywords': [
                    "subject", "from", "forwarded message", "sent"
                ],
                'required': ["subject:", "from:"],
                'patterns': [
                    r'From:.*@',
                    r'Subject:',
                    r'Sent:',
                    r'To:',
                    r'Cc:'
                ],
                # EXPANDED unique terms
                'unique_terms': [
                    'subject:',
                    'from:',
                    'to:',
                    'cc:',
                    'bcc:',
                    'forwarded message',
                    'sent:',
                    '@',
                    're:',  # Reply indicator
                    'fw:',  # Forward indicator
                    '-----Original Message-----'  # Email separator
                ],
                # Penalize financial/legal terms heavily
                'forbidden_terms': [
                    'cibil',
                    'credit score',
                    'loan',
                    'mortgage',
                    'disbursement',
                    'sanction',
                    'applicant',
                    'co-applicant',
                    'branch',
                    'zone',
                    'memorandum',
                    'mortgage',
                    'deed',
                    'property',
                    'valuation'
                ],
                'layout': 'letter',
                'page_count_range': (1, 5),
                'weight': 4.0  # INCREASED significantly
            },
            
            # 17. MODT
            'modt': {
                'keywords': [
                    "memorandum of deposit of title deeds", "memorandum of deposit",
                    "deposit of title deeds", "equitable mortgage", "mortgage",
                    "mortgage deed", "this deed of mortgage", "deed of mortgage",
                    "section 58", "transfer of property act", "mortgagor", "mortgagee"
                ],
                'required': ["memorandum of deposit", "mortgage deed"],  # More specific
                'patterns': [],
                'unique_terms': [
                    'memorandum of deposit',
                    'title deeds',
                    'equitable mortgage',
                    'mortgagor',
                    'mortgagee',
                    'section 58',
                    'transfer of property act',
                    'deed of mortgage'
                ],
                # EXPANDED forbidden terms
                'forbidden_terms': [
                    'legal due diligence',
                    'scrutiny report',
                    'legal assessment',
                    'due diligence',
                    'internal report',  # Common in legal reports
                    'applicant name',   # Common in applications, not MODT
                    'co-applicant',     # Common in applications
                    'branch',           # Banking term
                    'zone'              # Banking term
                ],
                'layout': 'legal',
                'page_count_range': (5, 30),
                'weight': 3.0
            },
            
            # 18. MUTHOOT ONE APP
            'muthoot_one_app': {
                'keywords': [
                    "VERIFY YOUR KYC", "ACCOUNT & SETTINGS", "Manage UCIC"
                ],
                'required': ["KYC", "UCIC"],
                'patterns': [],
                'unique_terms': ['UCIC', 'MUTHOOT', 'VERIFY KYC'],
                'forbidden_terms': ['account statement', 'balance'],  # Not bank statement
                'layout': 'screenshot',
                'page_count_range': (1, 1),
                'weight': 2.5
            },
            
            # 19. NACH
            'nach': {
                'keywords': [
                    "ENach", "AUTH SUCCESS", "Security Enach"
                ],
                'required': ["ENach"],
                'patterns': [],
                'unique_terms': ['ENach', 'AUTH SUCCESS', 'Security Enach'],
                'forbidden_terms': [],
                'layout': 'form',
                'page_count_range': (1, 2),
                'weight': 3.0
            },
            
            # 20. PAN CARD
            'pan_card': {
                'keywords': [
                    "INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER CARD",
                    "GOVT OF INDIA"
                ],
                'required': ["INCOME TAX", "PERMANENT ACCOUNT"],
                'patterns': [
                    r'[A-Z]{5}\d{4}[A-Z]',
                ],
                'unique_terms': ['PERMANENT ACCOUNT NUMBER', 'PAN', 'INCOME TAX DEPARTMENT'],
                'forbidden_terms': ['GST', 'UIDAI'],  # Not GST or Aadhaar
                'layout': 'id_card',
                'page_count_range': (1, 1),
                'weight': 3.0
            },
            
            # 21. PD DOCUMENT
            'pd_document': {
                'keywords': [
                    "personal discussion sheet", "lending business",
                    "pd done by", "pd date", "address of the pd visit",
                    "applicant and co applicant details", "joint property owner"
                ],
                'required': ["pd done by"],
                'patterns': [],
                'unique_terms': ['personal discussion', 'pd done by', 'pd date', 'pd visit'],
                'forbidden_terms': [],
                'layout': 'document',
                'page_count_range': (1, 10),
                'weight': 3.0
            },
            
            # 22. PENNY DROP
            'penny_drop': {
                'keywords': [
                    "penny drop"
                ],
                'required': ["penny drop"],
                'patterns': [],
                'unique_terms': ['penny drop', 'verification'],
                'forbidden_terms': ['statement', 'balance', 'account no'],  # Not bank statement
                'layout': 'report',
                'page_count_range': (1, 1),
                'weight': 3.0
            },
            
            # 23. PROPERTY VERIFICATION
            'property_verification': {
                'keywords': [
                    "property verification", "property verification report",
                    "residence verification", "date of visit", "co ordinates of property",
                    "boundaries of the property", "address of the property",
                    "landmark nearby", "family members residing", "neighbour feedback"
                ],
                'required': ["property verification", "visit"],
                'patterns': [],
                'unique_terms': ['date of visit', 'coordinates', 'boundaries', 'landmark', 'neighbour feedback'],
                'forbidden_terms': ['loan', 'mortgage'],  # Not loan doc
                'layout': 'report',
                'page_count_range': (2, 10),
                'weight': 2.5
            },
            
            # 24. RCU SAMPLING REPORT
            'rcu_sampling_report': {
                'keywords': [
                    "rcu sampling report", "rcu agency", "guarantor name",
                    "sampling details", "agency dedupe", "field executive"
                ],
                'required': ["rcu", "sampling"],
                'patterns': [],
                'unique_terms': ['rcu', 'sampling', 'dedupe', 'field executive'],
                'forbidden_terms': [],
                'layout': 'report',
                'page_count_range': (1, 5),
                'weight': 2.5
            },
            
            # 25. REQUEST FOR DISBURSAL
            'request_for_disbursal': {
                'keywords': [
                    "REQUEST FOR DISBURSAL",
                    "YOU TO ISSUE OUR LOAN DISBURSAL CHEQUE"
                ],
                'required': ["REQUEST FOR DISBURSAL"],
                'patterns': [],
                'unique_terms': ['REQUEST FOR DISBURSAL', 'DISBURSAL CHEQUE'],
                'forbidden_terms': ['statement', 'balance'],  # Not bank statement
                'layout': 'letter',
                'page_count_range': (1, 2),
                'weight': 3.0
            },
            
            # 26. TECHNICAL REPORT
            'technical_report': {
                'keywords': [
                    "technical report", "valuation report", "property valuation",
                    "scrutiny report", "technical scrutiny report", "verification type"
                ],
                'required': ["technical", "valuation"],
                'patterns': [],
                'unique_terms': ['technical', 'valuation', 'scrutiny'],
                'forbidden_terms': ['legal', 'title', 'mortgage'],  # Not legal doc
                'layout': 'report',
                'page_count_range': (3, 100),
                'weight': 2.0
            },
            
            # 27. TRADE LICENSE
            'trade_license': {
                'keywords': [
                    "form 11", "name of gram panchayat", "trade registration no",
                    "trade registration date", "trade registration certificate"
                ],
                'required': ["trade registration"],
                'patterns': [],
                'unique_terms': ['gram panchayat', 'trade registration', 'form 11'],
                'forbidden_terms': ['GST', 'UDYAM'],  # Not GST/Udyam
                'layout': 'certificate',
                'page_count_range': (1, 3),
                'weight': 3.0
            },
            
            # 28. UDYAM REGISTRATION
            'udyam_registration': {
                'keywords': [
                    "UDYAM REGISTRATION CERTIFICATE", "NAME OF ENTERPRISE"
                ],
                'required': ["UDYAM"],
                'patterns': [],
                'unique_terms': ['UDYAM', 'ENTERPRISE'],
                'forbidden_terms': ['GST', 'TRADE LICENSE'],  # Not trade license
                'layout': 'certificate',
                'page_count_range': (1, 2),
                'weight': 3.0
            },
            
            # 29. VERNACULAR
            'vernacular': {
                'keywords': [
                    "language", "declaration", "english", "hindi", "malyalam",
                    "tamil", "telugu", "kannada", "marathi", "gujarati", "bengali"
                ],
                'required': ["language"],
                'patterns': [
                    r'[^\x00-\x7F]{10,}',
                ],
                'unique_terms': [],  # Detected by non-English pattern
                'forbidden_terms': [],
                'layout': 'document',
                'page_count_range': (1, 20),
                'weight': 1.5
            },
            
            # 30. VETTING REPORT
            'vetting_report': {
                'keywords': [
                    "CERTIFICATE OF VETTING", "DESCRIPTION OF PROPERTY",
                    "Name of vendor", "vetting report"
                ],
                'required': ["VETTING"],
                'patterns': [],
                'unique_terms': ['VETTING', 'vendor'],
                'forbidden_terms': ['legal', 'title', 'mortgage'],  # Not legal doc
                'layout': 'report',
                'page_count_range': (2, 10),
                'weight': 3.0
            }
        }
        
        logger.info(f"✅ Classifier initialized with {len(self.DOCUMENT_SIGNATURES)} document types")
    
    def extract_text_features(self, results: List[Dict]) -> Dict:
        """Extract text features from OCR results"""
        if not results:
            return {}
        
        # Combine all text
        all_text = ' '.join([r.get('text', '') for r in results])
        all_text_original = ' '.join([r.get('text', '') for r in results])
        all_text_lower = all_text.lower()
        
        # Get page count
        pages = set(r.get('page', 1) for r in results)
        page_count = len(pages)
        
        # Check for tables
        if len(results) > 10:
            y_positions = [r['bbox'][1] for r in results]
            y_clusters = len(set([int(y/20) for y in y_positions]))
            has_table = y_clusters < len(results) * 0.3
        else:
            has_table = False
        
        # Check for form fields
        has_fields = any(':' in r.get('text', '') for r in results[:20])
        
        # Check for photo/logo
        has_photo_area = any('photo' in r.get('text', '').lower() for r in results[:10])
        
        # Check for non-English text
        has_non_english = any(ord(c) > 127 for c in all_text if c.isalpha())
        
        return {
            'all_text': all_text,
            'all_text_original': all_text_original,
            'all_text_lower': all_text_lower,
            'page_count': page_count,
            'has_table': has_table,
            'has_fields': has_fields,
            'has_photo_area': has_photo_area,
            'has_non_english': has_non_english,
            'total_regions': len(results),
        }
    
    def calculate_unique_score(self, text: str, unique_terms: list) -> float:
        """Bonus score for terms unique to this document type"""
        text_lower = text.lower()
        score = 0.0
        
        for term in unique_terms:
            if term.lower() in text_lower:
                score += 3.0  # High bonus for unique terms
                logger.debug(f"      ✨ Unique term match: '{term}'")
        
        return score
    
    def calculate_forbidden_penalty(self, text: str, forbidden_terms: list) -> float:
        """Penalty if forbidden terms appear"""
        text_lower = text.lower()
        penalty = 0.0
        
        for term in forbidden_terms:
            if term.lower() in text_lower:
                penalty += 2.0  # Penalty for forbidden terms
                logger.debug(f"      ⚠️ Forbidden term found: '{term}'")
        
        return penalty
    
    def classify(self, results: List[Dict], image: Optional[Image.Image] = None) -> Dict:
        """
        Classify document with enhanced distinguishing rules
        """
        logger.info("=" * 60)
        logger.info("🔍 Starting document classification")
        logger.info("=" * 60)
        
        if not results:
            logger.warning("⚠️ No results to classify")
            return {
                'document_type': 'unknown',
                'raw_type': 'unknown',
                'confidence': 0.0,
                'reasons': ['No text extracted']
            }
        
        # Extract features
        features = self.extract_text_features(results)
        text = features.get('all_text', '')
        text_original = features.get('all_text_original', '')
        
        logger.info(f"📄 Total text regions: {features['total_regions']}")
        logger.info(f"📄 Page count: {features['page_count']}")
        
        scores = {}
        all_reasons = []
        all_matches = {}
        
        for doc_type, config in self.DOCUMENT_SIGNATURES.items():
            logger.debug(f"\n--- Scoring {doc_type} ---")
            total_score = 0
            doc_matches = []
            
            # Method 1: Keywords (1 point each)
            kw_score, kw_matches = self.calculate_keyword_score(text, config.get('keywords', []))
            total_score += kw_score
            doc_matches.extend(kw_matches)
            
            # Method 2: Patterns (2 points each)
            pattern_score, pattern_matches = self.calculate_pattern_score(text_original, config.get('patterns', []))
            total_score += pattern_score
            doc_matches.extend(pattern_matches)
            
            # Method 3: Layout (up to 3.5 points)
            layout_score = 0
            if config.get('has_table') and features.get('has_table'):
                layout_score += 1.5
            if config.get('layout') == 'form' and features.get('has_fields'):
                layout_score += 1.0
            if config.get('has_photo') and features.get('has_photo_area'):
                layout_score += 1.0
            total_score += layout_score
            
            # Method 4: Page count (up to 1 point)
            page_range = config.get('page_count_range', (1, 100))
            if page_range[0] <= features['page_count'] <= page_range[1]:
                page_score = 1.0
            elif features['page_count'] < page_range[0]:
                page_score = 0.5
            else:
                page_score = 0.3
            total_score += page_score
            
            # Method 5: Required terms (2 points each)
            req_score, req_matches = self.calculate_required_score(text, config.get('required', []))
            total_score += req_score
            doc_matches.extend(req_matches)
            
            # NEW: Unique terms bonus (3 points each) - helps distinguish similar docs
            unique_score = self.calculate_unique_score(text, config.get('unique_terms', []))
            total_score += unique_score
            
            # NEW: Forbidden terms penalty (-2 points each) - prevents wrong classification
            forbidden_penalty = self.calculate_forbidden_penalty(text, config.get('forbidden_terms', []))
            total_score -= forbidden_penalty
            
            # Apply document weight
            weight = config.get('weight', 1.0)
            weighted_score = total_score * weight
            
            if weighted_score > 0:
                scores[doc_type] = weighted_score
                all_matches[doc_type] = doc_matches
                
                if weighted_score > 1:
                    logger.debug(f"  → Final score: {weighted_score:.2f} (raw: {total_score}, weight: {weight})")
        
        if not scores:
            logger.warning("⚠️ No matching patterns found")
            return {
                'document_type': 'unknown',
                'raw_type': 'unknown',
                'confidence': 0.0,
                'reasons': ['No matching patterns found'],
                'stats': {
                    'page_count': features['page_count'],
                    'total_regions': features['total_regions']
                }
            }
        
        # Get top candidates
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 TOP SCORES:")
        for doc_type, score in sorted_scores[:5]:
            logger.info(f"  {doc_type}: {score:.2f}")
        
        top_candidates = []
        for doc_type, score in sorted_scores[:5]:
            top_candidates.append({
                'type': doc_type.replace('_', ' ').title(),
                'raw_type': doc_type,
                'score': round(score, 2)
            })
        
        best_type, best_score = sorted_scores[0]
        second_type, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0)
        
        # Calculate confidence based on:
        # 1. Absolute score value
        # 2. Margin over second place (how clearly it won)
        margin = best_score - second_score
        
        if best_score >= 15:
            confidence = 0.95
        elif best_score >= 10:
            confidence = 0.85
        elif best_score >= 7:
            confidence = 0.75
        elif best_score >= 5:
            confidence = 0.60
        elif best_score >= 3:
            confidence = 0.40
        else:
            confidence = 0.20
        
        # Boost confidence if margin is large (clear winner)
        if margin > 5:
            confidence = min(confidence + 0.1, 0.98)
        elif margin < 1 and second_type:  # Too close!
            confidence = max(confidence - 0.2, 0.1)
            logger.warning(f"⚠️ Very close match: {best_type} ({best_score:.2f}) vs {second_type} ({second_score:.2f})")
        
        # Build reasons
        reasons = [
            f"Top match: {best_type.replace('_', ' ').title()} (score: {best_score:.2f})",
        ]
        
        if second_type:
            reasons.append(f"Runner-up: {second_type.replace('_', ' ').title()} ({second_score:.2f})")
        
        if margin < 2:
            reasons.append("⚠️ Very close match - verify manually")
        
        result = {
            'document_type': best_type.replace('_', ' ').title(),
            'raw_type': best_type,
            'confidence': round(confidence, 3),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'top_candidates': top_candidates,
            'reasons': reasons[:5],
            'matches': all_matches.get(best_type, [])[:15],
            'stats': {
                'page_count': features['page_count'],
                'total_regions': features['total_regions'],
                'margin': round(margin, 2)
            }
        }
        
        if confidence < 0.4:
            result['warning'] = 'Low confidence - please verify manually'
        elif margin < 2:
            result['warning'] = 'Close match - verify manually'
        
        logger.info(f"\n✅ Result: {result['document_type']} (conf: {confidence})")
        if margin < 2:
            logger.warning(f"⚠️ Margin over 2nd place: {margin:.2f}")
        logger.info("=" * 60)
        
        return result
    
    # Keep your existing helper methods
    def calculate_keyword_score(self, text: str, keywords: list) -> Tuple[float, list]:
        text_lower = text.lower()
        score = 0.0
        matches = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += 1.0
                matches.append(keyword[:30])
        return score, matches
    
    def calculate_pattern_score(self, text: str, patterns: list) -> Tuple[float, list]:
        score = 0.0
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 2.0
                matches.append(f"pattern:{pattern[:20]}")
        return score, matches
    
    def calculate_required_score(self, text: str, required_terms: list) -> Tuple[float, list]:
        text_lower = text.lower()
        score = 0.0
        matches = []
        for term in required_terms:
            if term.lower() in text_lower:
                score += 2.0
                matches.append(f"REQUIRED:{term[:20]}")
        return score, matches


# Global instance
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = BusinessDocumentClassifier()
        logger.info("✅ Document classifier initialized with exactly 30 document types")
    return _classifier

def classify_document(results: List[Dict], image: Optional[Image.Image] = None) -> Dict:
    return get_classifier().classify(results, image)