"""
recognizer.py

When the active engine is Surya or DocTR, detection and recognition happen
together inside TextDetector.  This module provides a standalone
TrOCR recognizer for:
  - Re-scoring / verifying crops with a second model
  - Handwriting mode when detection comes from a separate source
"""

import logging
from PIL import Image
from typing import List, Tuple, Optional

import torch
from .models import TrOCRRecognizer

logger = logging.getLogger(__name__)


class TextRecognizer:
    """
    Standalone transformer text recognizer (TrOCR).

    Use this when you have pre-cropped images and want recognition only,
    or when you want to run a second-pass recognizer on top of detector crops.
    """

    def __init__(self, config: dict):
        rec_cfg = config.get("recognition", {})
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        variant = rec_cfg.get("variant", "printed")   # printed / handwritten / printed-large
        num_beams = rec_cfg.get("num_beams", 4)
        max_length = rec_cfg.get("max_length", 128)

        self.model = TrOCRRecognizer(
            variant=variant,
            device=device,
            num_beams=num_beams,
            max_length=max_length,
        )
        logger.info("TextRecognizer (TrOCR) ready")

    def recognize(self, crops: List[Image.Image]) -> List[Tuple[str, float]]:
        """Recognize text from a list of PIL crops. Returns (text, confidence)."""
        return self.model.recognize_batch(crops)

    def recognize_single(self, crop: Image.Image) -> Tuple[str, float]:
        return self.model.recognize(crop)