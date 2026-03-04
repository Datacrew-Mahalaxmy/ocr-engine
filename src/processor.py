"""
processor.py - Main document processing pipeline v5 (with image preprocessing)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image
from tqdm import tqdm

from .detector import TextDetector
from .utils import EnhancedPostProcessor, save_results, visualize_results, export_to_txt
from .preprocess import get_preprocessor

logger = logging.getLogger(__name__)


def _detect_document_type(image: Image.Image) -> str:
    """
    Lightweight heuristic: is this page from a scanned or digital PDF?
    Samples the top-left margin (usually blank) and checks pixel noise.
    Digital = very uniform white (std < 8). Scanned = grainy (std >= 8).
    Returns 'digital' or 'scanned'. Used for adaptive merge threshold only.
    """
    import numpy as np
    arr = np.array(image.convert("L"), dtype=np.float32)
    h, w = arr.shape
    sample = arr[:max(1, h // 12), :max(1, w // 12)]
    std = float(sample.std())
    doc_type = "scanned" if std > 8.0 else "digital"
    logger.debug(f"doc_type detection: margin std={std:.1f} → {doc_type}")
    return doc_type


class DocumentProcessor:
    """
    End-to-end OCR pipeline for any document type with image preprocessing.
    Outputs per word/line: text, bbox [x1,y1,x2,y2], confidence, page, engine.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialising DocumentProcessor on device={self.device}")
        logger.info(f"Engine: {config.get('engine', 'auto')}")

        self.detector       = TextDetector(config)
        self.post_processor = EnhancedPostProcessor(config.get("post_process", {}))
        
        # Initialize preprocessor (will be adapted per document type)
        self.preprocessor = get_preprocessor('default')

        self._setup_output_dirs()
        logger.info("DocumentProcessor ready with image preprocessing")

    # ── PDF ──────────────────────────────────────────────────────────────────

    def process_pdf(
        self,
        pdf_path: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
        return_images: bool = False,
    ) -> Dict:
        images = self._pdf_to_images(pdf_path, start_page, end_page)

        all_results: List[Dict] = []
        all_images: Optional[List[Image.Image]] = [] if return_images else None

        # Detect document type from first page for adaptive preprocessing
        if images:
            doc_type = self._detect_detailed_document_type(images[0])
            self.preprocessor = get_preprocessor(doc_type)
            logger.info(f"Using {doc_type}-specific preprocessing")

        for idx, image in enumerate(tqdm(images, desc="Pages"), start=start_page):
            # Process image with preprocessing
            page_results = self.process_image(image, page_num=idx)
            all_results.extend(page_results)
            if return_images:
                all_images.append(image)

        all_results = self.post_processor.process(all_results)

        output = {
            "metadata": {
                "source":           pdf_path,
                "pages_processed":  len(images),
                "total_regions":    len(all_results),
                "device":           self.device,
                "engine":           type(self.detector.engine).__name__,
            },
            "results": all_results,
        }
        if return_images:
            output["images"] = all_images
        return output

    # ── Single image ─────────────────────────────────────────────────────────

    def process_image(
        self,
        image: Image.Image,
        page_num: int = 1,
    ) -> List[Dict]:
        """
        OCR a single PIL image with preprocessing. Returns RAW (unfiltered) detection dicts.
        Callers must invoke self.post_processor.process() on the results.
        """
        image = image.convert("RGB")
        pw, ph = image.size

        # Lightweight doc-type detection (no side effects — just metadata tag)
        doc_type = _detect_document_type(image)

        # Step 1: Preprocess the image for better OCR
        try:
            processed_image = self.preprocessor.preprocess(image)
            logger.debug(f"Page {page_num}: Applied preprocessing")
        except Exception as e:
            logger.warning(f"Preprocessing failed for page {page_num}: {e}, using original image")
            processed_image = image

        # Step 2: Run OCR on preprocessed image
        detections = self.detector.detect_with_crops(processed_image)

        results = []
        for det in detections:
            conf     = det.get("confidence", 1.0)
            det_conf = det.get("detection_confidence", conf)
            rec_conf = det.get("recognition_confidence", conf)
            results.append({
                "text":                    det["text"],
                "bbox":                    det["bbox"],
                "detection_confidence":    det_conf,
                "recognition_confidence":  rec_conf,
                "confidence":              conf,
                "page":                    page_num,
                "engine":                  det.get("engine", "unknown"),
                "page_width":              pw,
                "page_height":             ph,
                "doc_type":                doc_type,
            })

        logger.debug(f"Page {page_num}: {len(results)} raw regions (doc_type={doc_type})")
        return results

    # ── Batch ────────────────────────────────────────────────────────────────

    def process_batch(
        self,
        pdf_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        all_outputs = []
        for pdf_path in tqdm(pdf_paths, desc="PDFs"):
            try:
                output = self.process_pdf(pdf_path)
                all_outputs.append(output)
                if output_dir:
                    stem = Path(pdf_path).stem
                    save_results(output["results"],
                                 Path(output_dir) / f"{stem}_results.json")
                    export_to_txt(output["results"],
                                  Path(output_dir) / f"{stem}_text.txt")
            except Exception as e:
                logger.error(f"Failed {pdf_path}: {e}", exc_info=True)
                all_outputs.append({"error": str(e), "source": pdf_path})
        return all_outputs

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _pdf_to_images(self, pdf_path, start_page=1, end_page=None):
        from pdf2image import convert_from_path
        # Prefer explicit dpi; fall back to doc-type-aware dpi_settings; then global dpi.
        doc_type = self.config.get("document_type", "scanned")
        dpi_settings = self.config.get("dpi_settings", {})
        dpi = (
            dpi_settings.get(doc_type)
            or self.config.get("dpi", 300)
        )
        images = convert_from_path(
            pdf_path, dpi=dpi,
            first_page=start_page, last_page=end_page,
        )
        logger.info(f"Converted {len(images)} pages at {dpi} dpi (doc_type={doc_type})")
        return images

    def _detect_detailed_document_type(self, image: Image.Image) -> str:
        """
        Detect document type for adaptive preprocessing.
        Uses quick heuristics to choose the best preprocessor.
        """
        import numpy as np
        
        # Convert to grayscale
        gray = np.array(image.convert("L"))
        
        # Check if it's Aadhaar (colored background, specific patterns)
        # Simple heuristic: check for high contrast areas and typical layout
        h, w = gray.shape
        top_region = gray[:h//4, :]
        
        if np.std(top_region) > 50 and np.mean(top_region) < 200:
            # Likely Aadhaar with colored background and text
            return 'aadhaar_card'
        
        # Check for tabular structure (bank statements)
        edges = cv2.Canny(gray, 50, 150)
        horizontal_lines = np.sum(edges, axis=1)
        if np.max(horizontal_lines) > w * 0.3:
            return 'bank_statement'
        
        return 'default'

    def _setup_output_dirs(self):
        out = self.config.get("output", {})
        for key in ("json_output_dir", "image_output_dir"):
            d = out.get(key)
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)