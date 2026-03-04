"""
sbert_similarity.py - Compare Textract and DocTR JSON using Sentence-BERT
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Available models
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fastest, good balance",
        "dimensions": 384
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2", 
        "description": "Balanced, better accuracy",
        "dimensions": 768
    },
    "multi-qa-mpnet-base-dot-v1": {
        "name": "multi-qa-mpnet-base-dot-v1",
        "description": "Most accurate for semantic search", 
        "dimensions": 768
    }
}

class SBERTComparator:
    """
    Compare Textract and DocTR JSON outputs using Sentence-BERT semantic similarity
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence-BERT model with the given model name string
        """
        logger.info(f"📥 Loading SBERT model: {model_name}")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        model_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ SBERT loaded on {self.device} | Model: {model_name} | Dimensions: {model_dim}")
    
    def extract_text_from_textract(self, textract_json) -> List[Dict]:
        """
        Extract text blocks from Textract JSON
        Handles both:
        - Full AWS Textract format (dict with 'Blocks' key)
        - Simple list format (your aadhaar card (5).json)
        """
        text_blocks = []
        
        # Case 1: It's a list (your format)
        if isinstance(textract_json, list):
            logger.info(f"Textract JSON is a list with {len(textract_json)} items")
            for item in textract_json:
                text = item.get('text', '').strip()
                if text:
                    text_blocks.append({
                        'text': text,
                        'confidence': item.get('confidence', 100) / 100,
                        'page': item.get('page', 1),
                        'source': 'textract'
                    })
        
        # Case 2: It's a dict with 'Blocks' key (AWS format)
        elif isinstance(textract_json, dict):
            blocks = textract_json.get('Blocks', [])
            logger.info(f"Textract JSON has {len(blocks)} blocks")
            for block in blocks:
                if block.get('BlockType') in ['LINE', 'WORD']:
                    text = block.get('Text', '').strip()
                    if text:
                        text_blocks.append({
                            'text': text,
                            'confidence': block.get('Confidence', 100) / 100,
                            'block_type': block.get('BlockType'),
                            'page': block.get('Page', 1),
                            'id': block.get('Id', ''),
                            'source': 'textract'
                        })
        
        logger.info(f"Extracted {len(text_blocks)} text blocks from Textract")
        return text_blocks
    
    def extract_text_from_doctr(self, doctr_json) -> List[Dict]:
        """
        Extract text blocks from DocTR JSON
        Handles both:
        - Your download format (with metadata wrapper)
        - Raw list of results
        """
        text_blocks = []
        
        # Case 1: Your download format (with metadata wrapper)
        if isinstance(doctr_json, dict):
            # Check if it has text_by_page
            if 'text_by_page' in doctr_json:
                logger.info("DocTR JSON is in download format with text_by_page")
                for page_num, page_text in doctr_json.get('text_by_page', {}).items():
                    if page_text.strip():
                        lines = page_text.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                text_blocks.append({
                                    'text': line.strip(),
                                    'confidence': doctr_json.get('metadata', {}).get('confidence', 1.0),
                                    'page': int(page_num),
                                    'source': 'doctr'
                                })
            
            # Check if it has detailed_results
            elif 'detailed_results' in doctr_json:
                logger.info("DocTR JSON has detailed_results")
                for item in doctr_json.get('detailed_results', []):
                    text = item.get('text', '').strip()
                    if text:
                        text_blocks.append({
                            'text': text,
                            'confidence': item.get('confidence', 1.0),
                            'bbox': item.get('bbox', []),
                            'page': item.get('page', 1),
                            'source': 'doctr'
                        })
        
        # Case 2: Raw list of results (your API output format)
        elif isinstance(doctr_json, list):
            logger.info(f"DocTR JSON is a raw list with {len(doctr_json)} items")
            for item in doctr_json:
                text = item.get('text', '').strip()
                if text:
                    text_blocks.append({
                        'text': text,
                        'confidence': item.get('confidence', 1.0),
                        'bbox': item.get('bbox', []),
                        'page': item.get('page', 1),
                        'source': 'doctr'
                    })
        
        logger.info(f"Extracted {len(text_blocks)} text blocks from DocTR")
        return text_blocks
    
    def compute_similarity_matrix(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between two lists of texts
        """
        if not texts1 or not texts2:
            return np.array([])
        
        # Encode all texts
        emb1 = self.model.encode(texts1, convert_to_tensor=True)
        emb2 = self.model.encode(texts2, convert_to_tensor=True)
        
        # Compute cosine similarity matrix
        similarity_matrix = util.pytorch_cos_sim(emb1, emb2)
        
        return similarity_matrix.cpu().numpy()
    
    def align_and_compare(self, textract_blocks: List[Dict], doctr_blocks: List[Dict]) -> Dict:
        """
        Align and compare Textract and DocTR blocks using semantic similarity
        """
        start_time = time.time()
        
        if not textract_blocks or not doctr_blocks:
            return {
                'overall_accuracy': 0,
                'semantic_similarity': 0,
                'word_error_rate': 100,
                'character_error_rate': 100,
                'processing_time': 0,
                'processing_time_display': '0.0s',
                'aligned_pairs': [],
                'stats': {
                    'textract_blocks': len(textract_blocks),
                    'doctr_blocks': len(doctr_blocks),
                    'matched_pairs': 0,
                    'unmatched_textract': len(textract_blocks)
                }
            }
        
        # Extract texts
        textract_texts = [b['text'] for b in textract_blocks]
        doctr_texts = [b['text'] for b in doctr_blocks]
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(textract_texts, doctr_texts)
        
        # Greedy alignment
        aligned_pairs = []
        used_textract = set()
        used_doctr = set()
        
        # Get all possible pairs with similarity > 0.3
        pairs = []
        for i in range(len(textract_texts)):
            for j in range(len(doctr_texts)):
                pairs.append((i, j, float(sim_matrix[i][j])))
        
        # Sort by similarity descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching
        for i, j, sim in pairs:
            if i not in used_textract and j not in used_doctr and sim > 0.3:
                aligned_pairs.append({
                    'textract': {
                        'text': textract_texts[i],
                        'confidence': textract_blocks[i].get('confidence', 1.0),
                        'page': textract_blocks[i].get('page', 1),
                        'index': i
                    },
                    'doctr': {
                        'text': doctr_texts[j],
                        'confidence': doctr_blocks[j].get('confidence', 1.0),
                        'page': doctr_blocks[j].get('page', 1),
                        'index': j
                    },
                    'similarity': sim,
                    'similarity_percent': round(sim * 100, 1)
                })
                used_textract.add(i)
                used_doctr.add(j)
        
        # Add unmatched Textract blocks
        for i, text in enumerate(textract_texts):
            if i not in used_textract:
                aligned_pairs.append({
                    'textract': {
                        'text': text,
                        'confidence': textract_blocks[i].get('confidence', 1.0),
                        'page': textract_blocks[i].get('page', 1),
                        'index': i
                    },
                    'doctr': None,
                    'similarity': 0.0,
                    'similarity_percent': 0
                })
        
        # Calculate metrics
        matched_pairs = [p for p in aligned_pairs if p['doctr']]
        similarities = [p['similarity'] for p in matched_pairs]
        
        semantic_similarity = np.mean(similarities) if similarities else 0
        overall_accuracy = semantic_similarity * 100
        
        # Word Error Rate (semantic version)
        matched_words = sum(1 for p in matched_pairs if p['similarity'] > 0.8)
        word_error_rate = (1 - (matched_words / len(textract_texts))) * 100 if textract_texts else 100
        
        # Character Error Rate (approximate from semantic similarity)
        char_error_rate = (1 - semantic_similarity) * 100
        
        processing_time = time.time() - start_time
        
        # Sort by page for display
        aligned_pairs.sort(key=lambda x: (
            x['textract']['page'] if x['textract'] else 999,
            x['textract']['index'] if x['textract'] else 0
        ))
        
        return {
            'overall_accuracy': round(overall_accuracy, 1),
            'semantic_similarity': round(semantic_similarity * 100, 1),
            'word_error_rate': round(word_error_rate, 1),
            'character_error_rate': round(char_error_rate, 1),
            'processing_time': round(processing_time, 2),
            'processing_time_display': f"{processing_time:.1f}s",
            'aligned_pairs': aligned_pairs[:20],  # Limit for display
            'stats': {
                'textract_blocks': len(textract_blocks),
                'doctr_blocks': len(doctr_blocks),
                'matched_pairs': len(matched_pairs),
                'unmatched_textract': len(textract_blocks) - len(matched_pairs)
            }
        }


# Cache for loaded models
_model_instances = {}

def get_comparator(model_name: str) -> SBERTComparator:
    """Get or create comparator instance for specific model"""
    if model_name not in _model_instances:
        _model_instances[model_name] = SBERTComparator(model_name)
    return _model_instances[model_name]


# THIS IS THE FUNCTION YOU NEED - compare_with_all_models
def compare_with_all_models(textract_path: str, doctr_path: str) -> Dict:
    """
    Compare Textract and DocTR using ALL available models
    Returns results from all models
    """
    # Load JSON files once
    with open(textract_path, 'r', encoding='utf-8') as f:
        textract_data = json.load(f)
    
    with open(doctr_path, 'r', encoding='utf-8') as f:
        doctr_data = json.load(f)
    
    # Extract text blocks once (same for all models)
    # Use first model to extract (extraction doesn't depend on model)
    temp_comparator = get_comparator("all-MiniLM-L6-v2")
    textract_blocks = temp_comparator.extract_text_from_textract(textract_data)
    doctr_blocks = temp_comparator.extract_text_from_doctr(doctr_data)
    
    # Run all models in parallel
    all_results = {}
    
    with ThreadPoolExecutor(max_workers=len(AVAILABLE_MODELS)) as executor:
        future_to_model = {}
        
        for model_name in AVAILABLE_MODELS.keys():
            comparator = get_comparator(model_name)
            future = executor.submit(
                comparator.align_and_compare, 
                textract_blocks, 
                doctr_blocks
            )
            future_to_model[future] = model_name
        
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result(timeout=300)
                all_results[model_name] = result
                logger.info(f"✅ Model {model_name} completed")
            except Exception as e:
                logger.error(f"❌ Model {model_name} failed: {e}")
                all_results[model_name] = {"error": str(e)}
    
    return {
        "models": all_results,
        "model_info": AVAILABLE_MODELS,
        "stats": {
            "textract_blocks": len(textract_blocks),
            "doctr_blocks": len(doctr_blocks)
        }
    }


def compare_single_model(textract_path: str, doctr_path: str, model_name: str) -> Dict:
    """Compare using a single specified model"""
    # Load JSON files
    with open(textract_path, 'r', encoding='utf-8') as f:
        textract_data = json.load(f)
    
    with open(doctr_path, 'r', encoding='utf-8') as f:
        doctr_data = json.load(f)
    
    # Get comparator for specified model
    comparator = get_comparator(model_name)
    
    # Extract text blocks
    textract_blocks = comparator.extract_text_from_textract(textract_data)
    doctr_blocks = comparator.extract_text_from_doctr(doctr_data)
    
    # Compare
    result = comparator.align_and_compare(textract_blocks, doctr_blocks)
    
    return {
        "model": model_name,
        "result": result,
        "model_info": AVAILABLE_MODELS.get(model_name, {})
    }