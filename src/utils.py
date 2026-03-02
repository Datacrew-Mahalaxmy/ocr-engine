"""
utils.py - Post-processing, visualisation, and I/O utilities
Enhanced version for Aadhaar cards and Indian documents
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Enhanced OCR post-processor for Aadhaar cards and Indian documents.
    Handles: English extraction, garbage filtering, corrupted page detection.
    """

    def __init__(self, config: dict):
        # Basic settings
        self.min_confidence = config.get("min_confidence", 0.5)
        self.min_text_length = config.get("min_text_length", 2)
        self.nms_iou_threshold = config.get("nms_iou_threshold", 0.5)
        self.merge_threshold = config.get("merge_threshold", 20)
        self.enable_reading_order = config.get("enable_reading_order", True)
        self.enable_merge = config.get("enable_merge", True)
        self.column_detection = config.get("column_detection", True)
        self.filter_noise = config.get("filter_noise", True)
        self.min_col_gap_fraction = config.get("min_col_gap_fraction", 0.06)
        self.large_text_boost = config.get("large_text_merge_boost", 2.5)
        self.large_text_height_px = config.get("large_text_height_px", 40)
        
        # Column settings
        self.max_merge_x_fraction = config.get("max_merge_x_fraction", 0.30)
        self.logo_zone_x = config.get("logo_zone_x_fraction", 0.20)
        self.logo_zone_y = config.get("logo_zone_y_fraction", 0.18)
        self.logo_top_band_y = config.get("logo_top_band_y_fraction", 0.06)
        
        # Feature toggles
        self.enable_desc_stitch = config.get("enable_desc_stitch", True)
        self.enable_text_normalisation = config.get("enable_text_normalisation", True)
        
        # English-only settings
        self.english_only = config.get("english_only", True)
        
        # Unicode ranges for Indian scripts (to filter out)
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
        
        # Internal tracking
        self._last_rejection_reason = 'unknown'

    def process(self, results: List[Dict]) -> List[Dict]:
        """Main processing pipeline"""
        if not results:
            return results

        # Step 0: Filter out entire corrupted pages
        results = self._filter_corrupted_pages(results)
        
        # Step 1: Statistical garbage filtering
        results = self._statistical_filter(results)
        
        if not results:
            return results

        # Step 2: Remove duplicate overlapping boxes (NMS)
        results = remove_overlapping_boxes(results, iou_threshold=self.nms_iou_threshold)

        # Step 3: Column-aware reading order sort
        if self.enable_reading_order:
            results = self._column_aware_sort(results)

        # Step 4: Merge words into lines
        if self.enable_merge:
            results = self._merge_lines(results)

        # Step 5: Stitch multi-line table descriptions
        if self.enable_desc_stitch:
            results = self._stitch_table_descriptions(results)

        # Step 6: Text normalisation (just remove extra spaces)
        if self.enable_text_normalisation:
            for r in results:
                r["text"] = ' '.join(r["text"].split())

        return results

    # ==================== CORRUPTED PAGE DETECTION ====================
    
    def _is_corrupted_page(self, results: List[Dict]) -> bool:
        """Detect if a page is corrupted (like all 1's or all symbols)"""
        if not results:
            return True
        
        # Get all text from the page
        all_text = ' '.join([r.get('text', '') for r in results])
        
        if len(all_text) < 10:
            return False  # Too short to judge
        
        # Check for repeating patterns (like all 1's)
        if len(all_text) > 20:
            # Check if it's all the same character or digits
            unique_chars = set(all_text.replace(' ', ''))
            
            # If very few unique characters, might be corrupted
            if len(unique_chars) <= 3:
                # Check if it's just numbers or symbols
                all_digits = all(c.isdigit() or c.isspace() for c in all_text)
                all_symbols = all(ord(c) > 0x2000 or c.isspace() for c in all_text)
                
                if all_digits or all_symbols:
                    logger.info(f"Corrupted page detected: {all_text[:50]}...")
                    return True
        
        return False
    
    def _filter_corrupted_pages(self, results: List[Dict]) -> List[Dict]:
        """Remove entire pages that are corrupted"""
        if not results:
            return results
        
        # Group by page
        pages = {}
        for r in results:
            page = r.get('page', 1)
            if page not in pages:
                pages[page] = []
            pages[page].append(r)
        
        # Filter each page
        clean_results = []
        for page_num, page_results in pages.items():
            if not self._is_corrupted_page(page_results):
                clean_results.extend(page_results)
            else:
                logger.info(f"Filtered corrupted page {page_num}")
        
        return clean_results

    # ==================== STATISTICAL FILTERING ====================
    
    def _statistical_filter(self, results: List[Dict]) -> List[Dict]:
        """Pure statistical filtering - works on ANY document"""
        if not results:
            return results

        out = []
        stats = {
            'non_english': 0,
            'low_conf': 0,
            'too_short': 0,
            'statistical_reject': 0,
            'kept': 0
        }

        for r in results:
            text = r.get("text", "").strip()
            conf = r.get("confidence", 0)

            # Basic quality
            if len(text) < self.min_text_length:
                stats['too_short'] += 1
                continue

            # Confidence filter
            if conf < self.min_confidence:
                stats['low_conf'] += 1
                continue

            # English-only filter (remove Hindi and other Indian scripts)
            if self.english_only:
                if not self._is_english_text(text):
                    stats['non_english'] += 1
                    continue

            # Apply statistical tests
            if not self._statistically_valid(text):
                stats['statistical_reject'] += 1
                continue

            out.append(r)
            stats['kept'] += 1

        # Log statistics
        if stats['kept'] > 0 or sum(stats.values()) > 0:
            logger.info(f"Filter stats: {stats}")

        return out
    
    def _is_english_text(self, text: str) -> bool:
        """Check if text contains only English/Latin characters"""
        for char in text:
            code = ord(char)
            
            # Allow ASCII range (English letters, numbers, punctuation)
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
    
    def _statistically_valid(self, text: str) -> bool:
        """
        Enhanced statistical validation for all types of input
        """
        if len(text) < 2:
            return False
        
        # Check for repeating symbols (like  from corrupted PDFs)
        symbol_count = sum(1 for c in text if ord(c) > 0x2000)  # Unicode symbols
        if symbol_count > 0 and symbol_count / len(text) > 0.3:
            return False
        
        # Check for long runs of the same digit (like "111111")
        for i in range(len(text)-4):
            if i+5 <= len(text) and all(c == text[i] for c in text[i:i+5]):
                if text[i].isdigit():
                    return False
        
        # Character type counts
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = text.count(' ')
        
        # If it's all digits, keep it (like 7419, 0949, 4800)
        if digits == len(text) - spaces:
            return True
        
        # If it's a mix but mostly digits, keep it (like "VID: 9125 2113")
        if digits > 0 and letters > 0:
            if digits / len(text) > 0.3:
                return True
        
        # If it has no letters at all but has digits, keep it
        if letters == 0 and digits > 0:
            return True
        
        # Regular statistical checks for text
        if letters > 0:
            vowels = sum(1 for c in text.lower() if c in 'aeiou')
            vowel_ratio = vowels / letters
            
            # If it has vowels, probably real text
            if vowel_ratio > 0.15:
                return True
            
            # Check for consonant clusters (garbage like "STRPAT")
            words = text.split()
            for word in words:
                if len(word) > 4:
                    max_consecutive = 0
                    current = 0
                    for char in word.lower():
                        if char.isalpha() and char not in 'aeiou':
                            current += 1
                            max_consecutive = max(max_consecutive, current)
                        else:
                            current = 0
                    if max_consecutive > 5:
                        return False
            
            return vowel_ratio > 0.1
        
        return True

    # ==================== COLUMN-AWARE SORTING ====================
    
    def _column_aware_sort(self, results: List[Dict]) -> List[Dict]:
        """Sort results by columns for proper reading order"""
        pages: Dict[int, List[Dict]] = {}
        for r in results:
            pages.setdefault(r.get("page", 1), []).append(r)

        ordered = []
        for page_num in sorted(pages.keys()):
            page_items = pages[page_num]

            if self.column_detection and len(page_items) > 6:
                page_items = self._assign_columns(page_items)
            else:
                for r in page_items:
                    r["_col"] = 0

            med_h = self._median_height(page_items)
            band_size = max(1, med_h * 0.55)

            def sort_key(r, pn=page_num, bs=band_size):
                col = r.get("_col", 0)
                y1 = r["bbox"][1]
                band = int(y1 / bs)
                x1 = r["bbox"][0]
                return (pn, col, band, x1)

            page_items = sorted(page_items, key=sort_key)
            ordered.extend(page_items)

        return ordered

    def _assign_columns(self, items: List[Dict]) -> List[Dict]:
        """Assign column numbers based on x-position"""
        if not items:
            return items

        page_w = items[0].get("page_width", 0) or max(r["bbox"][2] for r in items)
        if page_w <= 0:
            for r in items:
                r["_col"] = 0
            return items

        n_bins = 200
        hist = np.zeros(n_bins, dtype=int)
        for r in items:
            xc = (r["bbox"][0] + r["bbox"][2]) / 2.0 / page_w
            b = min(int(xc * n_bins), n_bins - 1)
            hist[b] += 1

        kernel = np.array([1, 2, 4, 6, 8, 6, 4, 2, 1], dtype=float)
        kernel /= kernel.sum()
        smoothed = np.convolve(hist, kernel, mode="same")

        threshold = smoothed.max() * 0.03
        min_gap_bins = max(3, int(self.min_col_gap_fraction * n_bins))

        boundaries = []
        gap_start = None
        for i, v in enumerate(smoothed):
            if v <= threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    if (i - gap_start) >= min_gap_bins:
                        gap_centre = (gap_start + i) // 2
                        boundaries.append(int(gap_centre / n_bins * page_w))
                    gap_start = None

        boundaries = boundaries[:7]

        for r in items:
            xc = (r["bbox"][0] + r["bbox"][2]) / 2.0
            col = sum(1 for b in boundaries if xc > b)
            r["_col"] = col

        return items

    # ==================== LINE MERGING ====================
    
    def _merge_lines(self, results: List[Dict]) -> List[Dict]:
        """Merge adjacent word boxes into lines"""
        if len(results) <= 1:
            return results

        merged = []
        cur = results[0].copy()
        th = self.merge_threshold

        for nxt in results[1:]:
            # Hard boundary 1: different page
            if cur.get("page") != nxt.get("page"):
                merged.append(cur)
                cur = nxt.copy()
                continue

            # Hard boundary 2: different detected column
            if cur.get("_col", 0) != nxt.get("_col", 0):
                merged.append(cur)
                cur = nxt.copy()
                continue

            # Hard boundary 3: X-gap fraction guard
            page_w = cur.get("page_width", 0) or nxt.get("page_width", 0) or 0
            x_gap = nxt["bbox"][0] - cur["bbox"][2]
            if page_w > 0 and x_gap > self.max_merge_x_fraction * page_w:
                merged.append(cur)
                cur = nxt.copy()
                continue

            cur_h = cur["bbox"][3] - cur["bbox"][1]
            nxt_h = nxt["bbox"][3] - nxt["bbox"][1]
            cy = (cur["bbox"][1] + cur["bbox"][3]) / 2
            ny = (nxt["bbox"][1] + nxt["bbox"][3]) / 2
            h_gap = abs(cy - ny)

            both_large = (cur_h >= self.large_text_height_px and
                          nxt_h >= self.large_text_height_px)
            effective_th = th * self.large_text_boost if both_large else th

            same_line = h_gap < max(cur_h * 0.6, effective_th)
            close_x = -effective_th * 2 < x_gap < effective_th * 5

            if same_line and close_x:
                cur["text"] += " " + nxt["text"]
                cur["bbox"][2] = max(cur["bbox"][2], nxt["bbox"][2])
                cur["bbox"][3] = max(cur["bbox"][3], nxt["bbox"][3])
                cur["confidence"] = min(cur["confidence"], nxt["confidence"])
            else:
                merged.append(cur)
                cur = nxt.copy()

        merged.append(cur)

        for r in merged:
            r.pop("_col", None)

        return merged

    # ==================== TABLE DESCRIPTION STITCHING ====================
    
    def _stitch_table_descriptions(self, results: List[Dict]) -> List[Dict]:
        """Stitch multi-line table descriptions"""
        if not results:
            return results

        pages: Dict[int, List[Dict]] = {}
        for r in results:
            pages.setdefault(r.get("page", 1), []).append(r)

        out: List[Dict] = []

        for page_num in sorted(pages.keys()):
            items = pages[page_num]
            if not items:
                continue

            page_w = items[0].get("page_width", 0) or max(r["bbox"][2] for r in items)
            if page_w <= 0:
                out.extend(items)
                continue

            med_h = self._median_height(items)
            used = [False] * len(items)

            for i, row in enumerate(items):
                if used[i]:
                    continue

                row_w = row["bbox"][2] - row["bbox"][0]
                row_frac = row_w / page_w

                if row_frac < 0.55:
                    out.append(row)
                    used[i] = True
                    continue

                combined_text = row["text"]
                combined_bbox = list(row["bbox"])
                combined_conf = row["confidence"]

                for j in range(i + 1, len(items)):
                    if used[j]:
                        continue
                    cont = items[j]
                    cont_w = cont["bbox"][2] - cont["bbox"][0]
                    cont_frac = cont_w / page_w

                    if cont_frac >= 0.55:
                        break

                    y_dist = cont["bbox"][1] - combined_bbox[3]
                    if y_dist > med_h * 2.5 or y_dist < -med_h * 0.5:
                        break

                    cont_x_frac = cont["bbox"][0] / page_w
                    if not (0.15 < cont_x_frac < 0.65):
                        break

                    combined_text = combined_text + " " + cont["text"]
                    combined_bbox[2] = max(combined_bbox[2], cont["bbox"][2])
                    combined_bbox[3] = max(combined_bbox[3], cont["bbox"][3])
                    combined_conf = min(combined_conf, cont["confidence"])
                    used[j] = True

                row = dict(row)
                row["text"] = combined_text
                row["bbox"] = combined_bbox
                row["confidence"] = combined_conf
                out.append(row)
                used[i] = True

            for i, item in enumerate(items):
                if not used[i]:
                    out.append(item)

        return out

    # ==================== HELPERS ====================
    
    def _median_height(self, results: List[Dict]) -> float:
        """Calculate median height of text regions"""
        heights = [r["bbox"][3] - r["bbox"][1]
                   for r in results if r["bbox"][3] > r["bbox"][1]]
        return float(np.median(heights)) if heights else 20.0


# ===========================================================================
# BOUNDING BOX UTILITIES
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
        
        # Filter out overlapping boxes
        ranked = [r for r in ranked 
                  if calculate_iou(best["bbox"], r["bbox"]) < iou_threshold]
    
    return keep


# ===========================================================================
# TEXT EXTRACTION
# ===========================================================================

def extract_text_by_page(results: List[Dict]) -> Dict[int, str]:
    """Extract text grouped by page"""
    pages: Dict[int, List[str]] = {}
    for r in results:
        page = r.get("page", 1)
        if page not in pages:
            pages[page] = []
        pages[page].append(r.get("text", ""))
    
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
    
    logger.info(f"Exported text to {output_path}")


# ===========================================================================
# JSON I/O
# ===========================================================================

def save_results(results: List[Dict], output_path) -> None:
    """Save results as JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean results for JSON
    clean = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if k not in ['_col', 'crop']:
                if isinstance(v, (str, int, float, bool, list)) or v is None:
                    clean_r[k] = v
        clean.append(clean_r)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {output_path}")


# ===========================================================================
# VISUALISATION
# ===========================================================================

_ENGINE_COLORS = {
    "surya": "#00C853",
    "doctr": "#2196F3",
    "trocr+dbnet": "#FF6D00",
    "unknown": "#E91E63",
}


def visualize_results(
    image: Image.Image,
    results: List[Dict],
    output_path: Optional[str] = None,
    show_confidence: bool = True,
    show_engine: bool = False,
    alpha: float = 0.15,
) -> Image.Image:
    """Draw bounding boxes + labels with confidence heat coloring"""
    vis = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(vis)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for r in results:
        bbox = [int(v) for v in r["bbox"]]
        conf = float(r.get("confidence", 1.0))
        engine = r.get("engine", "unknown")
        color_hex = _conf_to_color(conf)
        color_rgb = _hex_to_rgb(color_hex)

        draw_ov.rectangle(bbox, fill=(*color_rgb, int(255 * alpha)))
        draw.rectangle(bbox, outline=color_hex, width=max(1, int(conf * 3)))

        label = r["text"][:35] + ("…" if len(r["text"]) > 35 else "")
        if show_confidence:
            label += f"  {conf:.2f}"
        if show_engine:
            label += f" [{engine}]"

        tx = bbox[0]
        ty = max(0, bbox[1] - 15)
        try:
            tb = draw.textbbox((tx, ty), label, font=font)
            if tb[2] < image.width and tb[3] < image.height:
                draw.rectangle(tb, fill=color_hex)
                draw.text((tx, ty), label, fill="white", font=font)
        except Exception:
            pass

    vis = Image.alpha_composite(vis, overlay).convert("RGB")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        vis.save(output_path)
        logger.info(f"Visualisation → {output_path}")

    return vis


def _conf_to_color(conf: float) -> str:
    """Convert confidence score to color"""
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
    """Convert hex color to RGB tuple"""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))