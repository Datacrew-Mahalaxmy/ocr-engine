"""
detector.py - Deep learning text detection (no traditional OCR)

Wraps the EngineFactory so the rest of the pipeline is engine-agnostic.
"""

import logging
from PIL import Image
from typing import List, Dict, Optional

from .models import EngineFactory

logger = logging.getLogger(__name__)


class TextDetector:
    """
    Engine-agnostic text detector.

    Internally uses whichever OCR engine is configured (Surya / DocTR / TrOCR+DB).
    Returns a list of detection dicts with keys:
        text, bbox, confidence, engine
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("Initialising OCR engine via EngineFactory...")
        self.engine = EngineFactory.create(config)
        logger.info(f"Active engine: {type(self.engine).__name__}")

    # ------------------------------------------------------------------
    # Public API (kept compatible with old processor.py)
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image) -> List[Dict]:
        """Run detection + recognition; return list of result dicts."""
        return self._run(image)

    def detect_with_crops(self, image: Image.Image) -> List[Dict]:
        """
        Same as detect() but also attaches the cropped PIL image
        under the 'crop' key (used by processor when recogniser is separate).
        """
        results = self._run(image)
        for r in results:
            bbox = r["bbox"]
            # Clamp to image bounds
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(image.width,  int(bbox[2]))
            y2 = min(image.height, int(bbox[3]))
            if x2 > x1 and y2 > y1:
                r["crop"] = image.crop([x1, y1, x2, y2])
                r["bbox"] = [x1, y1, x2, y2]
            else:
                r["crop"] = None
        valid_results  = [r for r in results if r.get("crop") is not None]
        dropped        = len(results) - len(valid_results)
        if dropped:
            logger.debug(f"detect_with_crops: dropped {dropped} zero-area bbox(es)")
        return valid_results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, image: Image.Image) -> List[Dict]:
        """Run engine and validate output."""
        image = image.convert("RGB")
        try:
            results = self.engine.run(image)
        except Exception as e:
            logger.error(f"Engine run failed: {e}", exc_info=True)
            return []

        valid = []
        for r in results:
            text = r.get("text", "").strip()
            bbox = r.get("bbox", [])
            conf = float(r.get("confidence", 0.0))

            if not text:
                continue
            if len(bbox) != 4:
                continue
            if conf < self.config.get("min_confidence", 0.0):
                continue

            valid.append(
                {
                    "text": text,
                    "bbox": [float(v) for v in bbox],
                    "confidence": conf,
                    "engine": r.get("engine", "unknown"),
                }
            )
        return valid