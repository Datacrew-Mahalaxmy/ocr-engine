"""
utils.py - Enhanced Post-processing with Priority 1,2,3 fixes
Priority 1: Date Normalization
Priority 2: Better Garbage Pattern Removal
Priority 3: Hindi Script Detection/Filtering
"""

import json
import logging
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

logger = logging.getLogger(__name__)


class GenericTextCleaner:
    """
    Generic text cleaner with Priority 1,2,3 fixes
    """
    
    def __init__(self):
        # ==================== PRIORITY 2: GARBAGE PATTERNS ====================
        self.GARBAGE_PATTERNS = [
            r'^[0-9]{1,2}$',              # Single/double digits alone
            r'^[0-9]{10,}$',                # Long number sequences (page numbers)
            r'^[^a-zA-Z0-9\s]+$',           # Pure symbols
            r'^_{2,}$',                      # Underline sequences
            r'^-{2,}$',                       # Dash sequences
            r'^\.{2,}$',                       # Dot sequences
            r'^[*]{2,}$',                      # Star sequences
            r'^[=]{2,}$',                       # Equal sequences
            r'^[+]{2,}$',                       # Plus sequences
            r'^[~]{2,}$',                       # Tilde sequences
            r'^\d+\.$',                         # Numbered list markers (1., 2.)
            r'^\d+\s*\d+\s*\d+$',               # Multiple numbers (page numbers)
            r'^[A-Z]{1,2}$',                     # Single/double uppercase (I, A, AB)
            r'^[a-z]{1,2}$',                     # Single/double lowercase
            r'^[>@&*#%]+$',                      # Special characters like >>, &&, @@
            r'^[•●○■□▪▫]',                      # Bullet points
            r'^[│┃┄┅┆┇┈┉┊┋]',                  # Box drawing characters
        ]
        
        # Patterns that might be valid despite looking like garbage
        self.SUSPICIOUS_PATTERNS = [
            r'^[A-Z]{3}$',                  # Three uppercase (like ABC) - might be acronym
            r'^\d{3}$',                      # Three digits - might be part of number
            r'^\d{4}$',                      # Four digits - might be Aadhaar part
        ]
        
        # Common OCR noise patterns
        self.OCR_NOISE_PATTERNS = [
            (r'\s+', ' '),   # Multiple spaces to single
            (r'\s+([.,!?;:])', r'\1'),  # Space before punctuation
        ]

    def clean_text(self, text: str) -> str:
        """
        Main cleaning function with Priority 1,2,3 fixes
        """
        if not text or len(text.strip()) < 2:
            return ""
        
        original = text
        text = text.strip()
        
        # Fix line breaks
        text = self._fix_line_breaks(text)
        
        # ==================== PRIORITY 1: DATE NORMALIZATION ====================
        text = self._normalize_dates(text)
        
        # Fix OCR noise
        text = self._fix_ocr_noise(text)
        
        # ==================== PRIORITY 2: GARBAGE DETECTION ====================
        if self._is_garbage(text):
            return ""
        
        # Final normalization
        text = ' '.join(text.split())
        
        if text != original and len(text) > 0:
            logger.debug(f"🧹 Cleaned: '{original[:50]}...' -> '{text[:50]}...'")
        
        return text

    def _fix_line_breaks(self, text: str) -> str:
        """Fix words broken by line breaks"""
        if not text:
            return text
        
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        words = text.split()
        if len(words) <= 1:
            return text
        
        fixed_words = []
        i = 0
        
        while i < len(words):
            # Single letter + longer word (a bcd -> abcd)
            if (i < len(words) - 1 and 
                len(words[i]) == 1 and 
                len(words[i+1]) > 1 and
                words[i].isalpha() and 
                words[i+1].isalpha()):
                fixed_words.append(words[i] + words[i+1])
                i += 2
            else:
                fixed_words.append(words[i])
                i += 1
        
        return ' '.join(fixed_words)

    # ==================== PRIORITY 1: DATE NORMALIZATION ====================
    
    def _normalize_dates(self, text: str) -> str:
        """
        Fix common date OCR errors and standardize format
        """
        if not text:
            return text
        
        # Fix common date OCR errors
        corrections = [
            # Fix years like 201977 -> 1977 (but keep 2025, 2024 etc.)
            (r'(\d{1,2}[/-]\d{1,2}[/-])(20\d{2})(\d{2})', lambda m: m.group(1) + m.group(2)[:4]),  # 21/2/201985 -> 21/2/2019
            (r'(\d{1,2}[/-]\d{1,2}[/-])(\d{4})(\d{2})', lambda m: m.group(1) + m.group(2)),  # Remove extra digits
            
            # Fix missing slashes
            (r'(\d{2})(\d{2})(\d{4})', r'\1/\2/\3'),  # 01011985 -> 01/01/1985
            
            # Standardize separators
            (r'(\d{2})[-.](\d{2})[-.](\d{4})', r'\1/\2/\3'),  # 01-01-1985 or 01.01.1985 -> 01/01/1985
            (r'(\d{1,2})[-.](\d{1,2})[-.](\d{2})', r'\1/\2/20\3'),  # 01-01-85 -> 01/01/1985
            
            # Fix OCR errors in dates
            (r'0x', '08'),  # 31-0x-2025 -> 31-08-2025
            (r'xx', ''),    # Remove stray xx
        ]
        
        for pattern, replacement in corrections:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        # Validate and fix impossible years
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{4}'
        dates = re.findall(date_pattern, text)
        
        for date in dates:
            # Handle both / and - separators
            sep = '/' if '/' in date else '-'
            parts = date.split(sep)
            if len(parts) == 3:
                day, month, year = parts
                # Pad day and month to 2 digits
                day = day.zfill(2)
                month = month.zfill(2)
                year_int = int(year)
                
                # Fix impossible years
                if year_int < 1900 or year_int > 2030:
                    # Try to extract last 4 digits
                    year_str = str(year_int)
                    if len(year_str) > 4:
                        fixed_year = year_str[-4:]
                        if 1900 <= int(fixed_year) <= 2030:
                            fixed_date = f"{day}/{month}/{fixed_year}"
                            text = text.replace(date, fixed_date)
                            logger.debug(f"📅 Fixed date: {date} -> {fixed_date}")
        
        return text

    def _fix_ocr_noise(self, text: str) -> str:
        """Fix common OCR noise patterns"""
        for pattern, replacement in self.OCR_NOISE_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text

    # ==================== PRIORITY 2: GARBAGE DETECTION ====================
    
    def _should_keep_despite_garbage(self, text: str) -> bool:
        """
        Check if text should be kept even if it matches garbage patterns
        """
        # Keep if it's a 4-digit number (could be Aadhaar part)
        if re.match(r'^\d{4}$', text):
            return True
        
        # Keep if it's a 10-digit number (could be phone)
        if re.match(r'^\d{10}$', text):
            return True
        
        # Keep if it's a 12-digit number with spaces (Aadhaar)
        if re.match(r'^\d{4}\s*\d{4}\s*\d{4}$', text):
            return True
        
        # Keep if it's a PAN number format
        if re.match(r'^[A-Z]{5}\d{4}[A-Z]$', text):
            return True
        
        # Keep if it has letters AND digits (like ABC123)
        if re.search(r'[A-Za-z]', text) and re.search(r'\d', text):
            return True
        
        return False

    def _is_garbage(self, text: str) -> bool:
        """
        Enhanced garbage detection - PRIORITY 2
        """
        if not text or len(text.strip()) < 2:
            return True
        
        # First check if it's something we should keep despite patterns
        if self._should_keep_despite_garbage(text):
            return False
        
        # Check against garbage patterns
        for pattern in self.GARBAGE_PATTERNS:
            if re.match(pattern, text):
                logger.debug(f"🗑️ Garbage pattern match: '{text}' matches {pattern}")
                return True
        
        # Character composition analysis
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = text.count(' ')
        others = len(text) - letters - digits - spaces
        
        # If mostly symbols, it's garbage
        if others > letters + digits and letters + digits < 3:
            return True
        
        # If it has letters, check vowel ratio for longer words
        if letters > 3:
            vowels = sum(1 for c in text.lower() if c in 'aeiou')
            if vowels == 0:
                return True  # No vowels in longer text = likely garbage
        
        return False

    def merge_line_groups(self, results: List[Dict]) -> List[Dict]:
        """
        Merge text that should be on same line
        """
        if len(results) <= 1:
            return results
        
        # Make a copy to avoid modifying original
        items = []
        for r in results:
            item = r.copy()
            items.append(item)
        
        # Sort by page, then y-position (top to bottom)
        items.sort(key=lambda x: (x.get('page', 1), x['bbox'][1]))
        
        merged = []
        current = items[0].copy()
        
        for next_item in items[1:]:
            # Different pages - can't merge
            if current.get('page', 1) != next_item.get('page', 1):
                current['text'] = self.clean_text(current['text'])
                if current['text']:
                    merged.append(current)
                current = next_item.copy()
                continue
            
            # Calculate vertical center
            y1_current = current['bbox'][1]
            y2_current = current['bbox'][3]
            y1_next = next_item['bbox'][1]
            y2_next = next_item['bbox'][3]
            
            center_current = (y1_current + y2_current) / 2
            center_next = (y1_next + y2_next) / 2
            height_current = y2_current - y1_current
            
            # Calculate horizontal gap
            x_gap = next_item['bbox'][0] - current['bbox'][2]
            
            # If vertically close and not too far horizontally
            if (abs(center_current - center_next) < height_current * 0.8 and
                x_gap < height_current * 3):
                # Same line - merge
                current['text'] += ' ' + next_item['text']
                current['bbox'][2] = max(current['bbox'][2], next_item['bbox'][2])
                current['bbox'][3] = max(current['bbox'][3], next_item['bbox'][3])
                current['confidence'] = min(current['confidence'], next_item['confidence'])
            else:
                # Different line - add current and move to next
                current['text'] = self.clean_text(current['text'])
                if current['text']:
                    merged.append(current)
                current = next_item.copy()
        
        # Add last item
        current['text'] = self.clean_text(current['text'])
        if current['text']:
            merged.append(current)
        
        logger.debug(f"📏 Line merging: {len(results)} -> {len(merged)}")
        return merged


class EnhancedPostProcessor:
    """
    Enhanced OCR post-processor with Priority 1,2,3 fixes
    """

    def __init__(self, config: dict, doc_type: str = 'default'):
        # Basic settings
        self.min_confidence = config.get("min_confidence", 0.5)
        self.min_text_length = config.get("min_text_length", 2)
        self.nms_iou_threshold = config.get("nms_iou_threshold", 0.5)
        self.merge_threshold = config.get("merge_threshold", 20)
        self.enable_reading_order = config.get("enable_reading_order", True)
        self.enable_merge = config.get("enable_merge", True)
        self.column_detection = config.get("column_detection", True)
        
        # Language settings
        self.english_only = config.get("english_only", True)
        self.doc_type = doc_type
        
        # Add text cleaner
        self.text_cleaner = GenericTextCleaner()
        
        # Unicode ranges for Indian scripts
        self.INDIAN_SCRIPT_RANGES = (
            (0x0900, 0x097F),   # Devanagari (Hindi, Sanskrit, Marathi, Nepali)
            (0x0980, 0x09FF),   # Bengali & Assamese
            (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
            (0x0A80, 0x0AFF),   # Gujarati
            (0x0B00, 0x0B7F),   # Oriya (Odia)
            (0x0B80, 0x0BFF),   # Tamil
            (0x0C00, 0x0C7F),   # Telugu
            (0x0C80, 0x0CFF),   # Kannada
            (0x0D00, 0x0D7F),   # Malayalam
            (0x0D80, 0x0DFF),   # Sinhala
        )
        
        # Devanagari range specifically for Hindi
        self.DEVANAGARI_RANGE = (0x0900, 0x097F)
        
        # Statistics tracking
        self.stats = {
            'garbage': 0,
            'non_english': 0,
            'hindi_removed': 0,
            'too_short': 0,
            'low_conf': 0,
            'kept': 0
        }

    def process(self, results: List[Dict]) -> List[Dict]:
        """Main processing pipeline with Priority 1,2,3 fixes"""
        if not results:
            return results

        logger.info(f"🔧 Starting enhanced post-processing with Priority 1,2,3 fixes")
        original_count = len(results)

        # Step 1: Filter corrupted pages
        results = self._filter_corrupted_pages(results)
        
        # ==================== PRIORITY 3: HINDI SCRIPT DETECTION/FILTERING ====================
        if self.english_only:
            results = self._remove_non_english(results)
        
        # Step 2: Basic filtering
        results = self._basic_filter(results)
        
        # Step 3: Remove overlapping boxes
        results = remove_overlapping_boxes(results, iou_threshold=self.nms_iou_threshold)

        # Step 4: Column-aware sorting
        if self.enable_reading_order:
            results = self._column_aware_sort(results)

        # Step 5: Merge lines with text cleaning (includes Priority 1 & 2)
        if self.enable_merge:
            results = self.text_cleaner.merge_line_groups(results)
        else:
            for r in results:
                r['text'] = self.text_cleaner.clean_text(r['text'])
            results = [r for r in results if r['text']]

        logger.info(f"📊 Filter stats: kept {len(results)}/{original_count}")
        logger.info(f"   Reasons: {self.stats}")
        
        return results

    # ==================== PRIORITY 3: HINDI SCRIPT DETECTION ====================
    
    def _is_hindi_text(self, text: str) -> bool:
        """
        Detect Hindi/Devanagari script - PRIORITY 3
        """
        if not text:
            return False
        
        devanagari_chars = 0
        total_chars = 0
        
        for char in text:
            code = ord(char)
            if code >= self.DEVANAGARI_RANGE[0] and code <= self.DEVANAGARI_RANGE[1]:
                devanagari_chars += 1
                total_chars += 1
            elif char.isalpha():
                total_chars += 1
        
        if total_chars == 0:
            return False
        
        # If more than 20% of alphabetic chars are Devanagari, it's Hindi
        return (devanagari_chars / total_chars) > 0.2

    def _is_english_text(self, text: str) -> bool:
        """Check if text is English (ASCII + common punctuation)"""
        if not text:
            return False
        
        for char in text:
            code = ord(char)
            
            # Allow ASCII range
            if 32 <= code <= 126:
                continue
            
            # Allow common punctuation
            if char in ' .,!?;:\'"-()[]{}@#$%&*+=/\\|<>~`':
                continue
            
            # Check if it's an Indian script character
            for start, end in self.INDIAN_SCRIPT_RANGES:
                if start <= code <= end:
                    return False
            
            # If it's any other non-ASCII character, reject
            return False
        
        return True

    def _remove_non_english(self, results: List[Dict]) -> List[Dict]:
        """
        Remove non-English text (including Hindi) - PRIORITY 3
        """
        cleaned = []
        
        for r in results:
            text = r.get('text', '')
            
            if self._is_english_text(text):
                cleaned.append(r)
            else:
                # Check if it's Hindi specifically
                if self._is_hindi_text(text):
                    self.stats['hindi_removed'] = self.stats.get('hindi_removed', 0) + 1
                    logger.debug(f"🇮🇳 Hindi removed: '{text}'")
                else:
                    self.stats['non_english'] += 1
                    logger.debug(f"🌐 Non-English removed: '{text}'")
        
        return cleaned

    def _basic_filter(self, results: List[Dict]) -> List[Dict]:
        """Basic statistical filtering"""
        cleaned = []
        for r in results:
            text = r.get('text', '').strip()
            conf = r.get('confidence', 0)
            
            if len(text) < self.min_text_length:
                self.stats['too_short'] += 1
                continue
            if conf < self.min_confidence:
                self.stats['low_conf'] += 1
                continue
            
            # Apply text cleaning (includes Priority 1 & 2)
            cleaned_text = self.text_cleaner.clean_text(text)
            if cleaned_text:
                r['text'] = cleaned_text
                cleaned.append(r)
                self.stats['kept'] += 1
            else:
                self.stats['garbage'] += 1
        
        return cleaned

    def _filter_corrupted_pages(self, results: List[Dict]) -> List[Dict]:
        """Remove entire corrupted pages"""
        if not results:
            return results
        
        pages = {}
        for r in results:
            page = r.get('page', 1)
            pages.setdefault(page, []).append(r)
        
        clean_results = []
        for page_num, page_results in pages.items():
            all_text = ' '.join([r.get('text', '') for r in page_results])
            if len(all_text) < 10:
                continue
            if self._is_page_corrupted(page_results):
                logger.info(f"🗑️ Filtered corrupted page {page_num}")
                continue
            clean_results.extend(page_results)
        
        return clean_results

    def _is_page_corrupted(self, page_results: List[Dict]) -> bool:
        """Check if page is corrupted"""
        all_text = ' '.join([r.get('text', '') for r in page_results])
        if len(all_text) < 20:
            return False
        unique_chars = set(all_text.replace(' ', ''))
        return len(unique_chars) <= 3

    def _column_aware_sort(self, results: List[Dict]) -> List[Dict]:
        """Sort by columns for reading order"""
        if not results:
            return results
        
        results.sort(key=lambda x: (x.get('page', 1), x['bbox'][1], x['bbox'][0]))
        return results


# ===========================================================================
# Backward compatibility
# ===========================================================================
PostProcessor = EnhancedPostProcessor


# ===========================================================================
# Bounding Box Utilities
# ===========================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def remove_overlapping_boxes(results: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Remove duplicate detections using IoU"""
    if not results:
        return results
    
    ranked = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    keep = []
    
    while ranked:
        best = ranked.pop(0)
        keep.append(best)
        ranked = [r for r in ranked if calculate_iou(best["bbox"], r["bbox"]) < iou_threshold]
    
    return keep


# ===========================================================================
# Text Extraction Utilities
# ===========================================================================

def extract_text_by_page(results: List[Dict]) -> Dict[int, str]:
    """Extract text grouped by page"""
    pages = {}
    for r in results:
        page = r.get("page", 1)
        pages.setdefault(page, []).append(r.get("text", ""))
    return {p: "\n".join(texts) for p, texts in sorted(pages.items())}


def export_to_txt(results: List[Dict], output_path) -> None:
    """Export results to text file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pages = extract_text_by_page(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in sorted(pages.keys()):
            f.write(f"--- Page {page} ---\n")
            f.write(pages[page])
            f.write("\n\n")
    
    logger.info(f"📄 Exported text to {output_path}")


def save_results(results: List[Dict], output_path) -> None:
    """Save results as JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    clean = []
    for r in results:
        clean_r = {k: v for k, v in r.items() 
                  if k not in ['_col', 'crop'] and isinstance(v, (str, int, float, bool, list))}
        clean.append(clean_r)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved results to {output_path}")


# ===========================================================================
# Visualization Utilities
# ===========================================================================

def visualize_results(image: Image.Image, results: List[Dict], output_path: Optional[str] = None,
                     show_confidence: bool = True, alpha: float = 0.15) -> Image.Image:
    """Draw bounding boxes on image"""
    vis = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(vis)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for r in results:
        bbox = [int(v) for v in r["bbox"]]
        conf = float(r.get("confidence", 1.0))
        color_hex = _conf_to_color(conf)
        color_rgb = _hex_to_rgb(color_hex)

        draw_ov.rectangle(bbox, fill=(*color_rgb, int(255 * alpha)))
        draw.rectangle(bbox, outline=color_hex, width=max(1, int(conf * 3)))

        label = r["text"][:35] + ("…" if len(r["text"]) > 35 else "")
        if show_confidence:
            label += f"  {conf:.2f}"

        tx, ty = bbox[0], max(0, bbox[1] - 15)
        try:
            tb = draw.textbbox((tx, ty), label, font=font)
            if tb[2] < image.width and tb[3] < image.height:
                draw.rectangle(tb, fill=color_hex)
                draw.text((tx, ty), label, fill="white", font=font)
        except:
            pass

    vis = Image.alpha_composite(vis, overlay).convert("RGB")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        vis.save(output_path)
        logger.info(f"🖼️ Visualisation → {output_path}")

    return vis


def _conf_to_color(conf: float) -> str:
    """Convert confidence to color"""
    conf = max(0.0, min(1.0, conf))
    if conf >= 0.7:
        g = int(150 + 105 * (conf - 0.7) / 0.3)
        return f"#00{g:02X}00"
    elif conf >= 0.4:
        t = (conf - 0.4) / 0.3
        g = int(180 * t)
        return f"#FF{g:02X}00"
    else:
        return "#CC2200"


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# Backward compatibility
PostProcessor = EnhancedPostProcessor