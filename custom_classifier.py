"""
custom_classifier.py - Document classifier with exactly your 30 document types
No merging, only renaming where needed for consistency
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
    Based on your provided list - no merging
    """
    
    def __init__(self):
        # ===== YOUR EXACT 30 DOCUMENTS (only renamed for consistency) =====
        
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
                'required': ["CIBIL", "Credit"],
                'patterns': [],
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
                'required': ["legal", "title"],
                'patterns': [],
                'layout': 'report',
                'page_count_range': (3, 20),
                'weight': 2.0
            },
            
            # 16. MAIL
            'mail': {
                'keywords': [
                    "subject", "from", "forwarded message", "sent"
                ],
                'required': ["subject", "from"],
                'patterns': [],
                'layout': 'letter',
                'page_count_range': (1, 5),
                'weight': 2.0
            },
            
            # 17. MODT
            'modt': {
                'keywords': [
                    "memorandum of deposit of title deeds", "memorandum of deposit",
                    "deposit of title deeds", "equitable mortgage", "mortgage",
                    "mortgage deed", "this deed of mortgage", "deed of mortgage",
                    "section 58", "transfer of property act", "mortgagor", "mortgagee"
                ],
                'required': ["memorandum of deposit", "mortgage"],
                'patterns': [],
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
        
        # Check for tables (multiple aligned items)
        if len(results) > 10:
            y_positions = [r['bbox'][1] for r in results]
            y_clusters = len(set([int(y/20) for y in y_positions]))
            has_table = y_clusters < len(results) * 0.3
        else:
            has_table = False
        
        # Check for form fields (labels followed by values)
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
    
    def calculate_keyword_score(self, text: str, keywords: list) -> Tuple[float, list]:
        """Method 1: Keyword matching"""
        text_lower = text.lower()
        score = 0.0
        matches = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += 1.0
                matches.append(keyword[:30])
        
        return score, matches
    
    def calculate_pattern_score(self, text: str, patterns: list) -> Tuple[float, list]:
        """Method 2: Pattern matching (regex)"""
        score = 0.0
        matches = []
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 2.0  # Patterns are stronger indicators
                matches.append(f"pattern:{pattern[:20]}")
        
        return score, matches
    
    def calculate_layout_score(self, features: Dict, doc_type: str, config: dict) -> Tuple[float, list]:
        """Method 3: Layout analysis"""
        score = 0.0
        reasons = []
        
        # Check table presence
        if config.get('has_table') and features.get('has_table'):
            score += 1.5
            reasons.append("table layout matches")
        
        # Check form fields
        if config.get('layout') == 'form' and features.get('has_fields'):
            score += 1.0
            reasons.append("form layout matches")
        
        # Check photo area
        if config.get('has_photo') and features.get('has_photo_area'):
            score += 1.0
            reasons.append("photo area detected")
        
        # Check non-English for vernacular
        if doc_type == 'vernacular' and features.get('has_non_english'):
            score += 3.0
            reasons.append("non-english text detected")
        
        return score, reasons
    
    def calculate_page_score(self, features: Dict, page_range: tuple) -> Tuple[float, list]:
        """Method 4: Page count analysis"""
        page_count = features.get('page_count', 1)
        min_pages, max_pages = page_range
        
        if min_pages <= page_count <= max_pages:
            return 1.0, ["page count matches"]
        elif page_count < min_pages:
            return 0.5, ["fewer pages than typical"]
        else:
            return 0.3, ["more pages than typical"]
    
    def calculate_required_score(self, text: str, required_terms: list) -> Tuple[float, list]:
        """Method 5: Required terms validation"""
        text_lower = text.lower()
        score = 0.0
        matches = []
        
        for term in required_terms:
            if term.lower() in text_lower:
                score += 2.0  # Required terms are important
                matches.append(f"REQUIRED:{term[:20]}")
        
        return score, matches
    
    def classify(self, results: List[Dict], image: Optional[Image.Image] = None) -> Dict:
        """
        Classify document using all 5 methods
        """
        if not results:
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
        
        # Calculate scores using all 5 methods
        scores = {}
        all_reasons = []
        all_matches = {}
        
        for doc_type, config in self.DOCUMENT_SIGNATURES.items():
            total_score = 0
            doc_matches = []
            doc_reasons = []
            
            # Method 1: Keywords
            kw_score, kw_matches = self.calculate_keyword_score(text, config.get('keywords', []))
            total_score += kw_score
            doc_matches.extend(kw_matches)
            
            # Method 2: Patterns
            pattern_score, pattern_matches = self.calculate_pattern_score(text_original, config.get('patterns', []))
            total_score += pattern_score
            doc_matches.extend(pattern_matches)
            
            # Method 3: Layout
            layout_score, layout_reasons = self.calculate_layout_score(features, doc_type, config)
            total_score += layout_score
            doc_reasons.extend(layout_reasons)
            
            # Method 4: Page count
            page_score, page_reasons = self.calculate_page_score(features, config.get('page_count_range', (1, 100)))
            total_score += page_score
            doc_reasons.extend(page_reasons)
            
            # Method 5: Required terms
            req_score, req_matches = self.calculate_required_score(text, config.get('required', []))
            total_score += req_score
            doc_matches.extend(req_matches)
            
            # Apply document weight
            total_score *= config.get('weight', 1.0)
            
            if total_score > 0:
                scores[doc_type] = total_score
                all_matches[doc_type] = doc_matches
                all_reasons.extend([f"{doc_type}: {r}" for r in doc_reasons[:2]])
        
        if not scores:
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
        top_candidates = []
        
        for doc_type, score in sorted_scores[:5]:
            top_candidates.append({
                'type': doc_type.replace('_', ' ').title(),
                'raw_type': doc_type,
                'score': round(score, 2)
            })
        
        best_type, best_score = sorted_scores[0]
        
        # Calculate confidence (normalized)
        max_possible = 30.0  # Approximate maximum possible score
        confidence = min(best_score / max_possible, 0.95)
        
        # Build reasons
        reasons = [
            f"Top match: {best_type.replace('_', ' ').title()}",
            f"Matched {len(all_matches.get(best_type, []))} keywords/patterns"
        ]
        
        if confidence > 0.7:
            reasons.append("High confidence match")
        elif confidence > 0.4:
            reasons.append("Medium confidence match")
        else:
            reasons.append("Low confidence - verify manually")
        
        # Add unique reasons
        unique_reasons = list(set([r for r in all_reasons if best_type in r]))[:3]
        reasons.extend(unique_reasons)
        
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
                'has_table': features.get('has_table', False),
                'has_fields': features.get('has_fields', False)
            }
        }
        
        # Add warning for low confidence
        if confidence < 0.4:
            result['warning'] = 'Low confidence - please verify manually'
        
        return result


# Global instance
_classifier = None

def get_classifier():
    """Get or create global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = BusinessDocumentClassifier()
        logger.info("✅ Document classifier initialized with exactly 30 document types")
    return _classifier

def classify_document(results: List[Dict], image: Optional[Image.Image] = None) -> Dict:
    """Convenience function to classify a document"""
    return get_classifier().classify(results, image)