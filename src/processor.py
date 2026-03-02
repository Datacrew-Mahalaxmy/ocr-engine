"""
processor.py - Main document processing pipeline v4 (stable)

Stable base: v3 logic preserved exactly.
v4 additions (safe only):
  - doc_type tag per result (for PostProcessor adaptive merge)
  - simple scanned-vs-digital detection via pixel std (no side effects)
  - no image region suppression (was too aggressive, caused regressions)
  - no auto-DPI (use config dpi directly, default 300)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image
from tqdm import tqdm

from .detector import TextDetector
from .utils import PostProcessor, save_results, visualize_results, export_to_txt

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
    End-to-end OCR pipeline for any document type.
    Outputs per word/line: text, bbox [x1,y1,x2,y2], confidence, page, engine.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialising DocumentProcessor on device={self.device}")
        logger.info(f"Engine: {config.get('engine', 'auto')}")

        self.detector       = TextDetector(config)
        self.post_processor = PostProcessor(config.get("post_process", {}))

        self._setup_output_dirs()
        logger.info("DocumentProcessor ready")

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

        for idx, image in enumerate(tqdm(images, desc="Pages"), start=start_page):
            # process_image returns RAW detections — post-processing applied once below
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
        OCR a single PIL image. Returns RAW (unfiltered) detection dicts.
        Callers must invoke self.post_processor.process() on the results.
        """
        image = image.convert("RGB")
        pw, ph = image.size

        # Lightweight doc-type detection (no side effects — just metadata tag)
        doc_type = _detect_document_type(image)

        detections = self.detector.detect_with_crops(image)

        results = []
        for det in detections:
            conf     = det.get("confidence", 1.0)
            det_conf = det.get("detection_confidence", conf)
            rec_conf = det.get("recognition_confidence", conf)
            results.append({
                "text":                    det["text"],
                "bbox":                    det["bbox"],
                # Separate scores: detection_confidence = how cleanly the region was found,
                # recognition_confidence = how sure the model is about the text string.
                # When the engine provides both (DocTR), they differ meaningfully.
                # When only one is available (Surya, TrOCR+DBNet), both equal that score.
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

    def _setup_output_dirs(self):
        out = self.config.get("output", {})
        for key in ("json_output_dir", "image_output_dir"):
            d = out.get(key)
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)