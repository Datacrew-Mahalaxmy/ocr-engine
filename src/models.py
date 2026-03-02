"""
OCR Models - Pure Transformer / Deep Learning (No Traditional OCR)

Engines (in priority order):
  1. Surya  - Pure transformer OCR (DetectionPredictor + RecognitionPredictor)
  2. DocTR  - DBNet detection + PARSeq transformer recognition
  3. TrOCR  - Microsoft TrOCR recognition + DBNet detection
"""

import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine 1: Surya
# ---------------------------------------------------------------------------
class SuryaOCREngine:
    """
    Surya transformer OCR. Tries multiple load strategies:
      Strategy 1 - new predictor API:  surya.detection + surya.recognition
      Strategy 2 - mixed API:          surya.detection + surya.model.recognition (crop-based)
      Strategy 3 - full legacy API:    surya.model.detection.segformer + surya.model.recognition
    """

    def __init__(self, device: Optional[str] = None, langs: Optional[List[str]] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.langs  = langs or ["en"]
        self.api_version = None
        self._load()

    def _load(self):
        # Strategy 1: new predictor API (surya 0.6.x / 0.17+)
        try:
            from surya.detection  import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            logger.info("Loading Surya DetectionPredictor...")
            self.det_predictor = DetectionPredictor()
            logger.info("Loading Surya RecognitionPredictor...")
            self.rec_predictor = RecognitionPredictor()
            self.api_version = "new"
            logger.info("Surya loaded via predictor API")
            return
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {e}")

        # Strategy 2: new detection + old recognition (crop-based)
        try:
            from surya.detection import DetectionPredictor
            from surya.model.recognition.model     import load_model     as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            logger.info("Loading Surya mixed API...")
            self.det_predictor = DetectionPredictor()
            self.rec_model     = load_rec_model()
            self.rec_processor = load_rec_processor()
            self.rec_model.to(self.device)
            self.api_version   = "mixed"
            logger.info("Surya loaded via mixed API")
            return
        except Exception as e:
            logger.debug(f"Strategy 2 failed: {e}")

        # Strategy 3: full legacy API
        try:
            from surya.model.detection.segformer import (
                load_model     as load_det_model,
                load_processor as load_det_processor,
            )
            from surya.model.recognition.model     import load_model     as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.ocr import run_ocr
            logger.info("Loading Surya legacy API...")
            self.det_model     = load_det_model()
            self.det_processor = load_det_processor()
            self.rec_model     = load_rec_model()
            self.rec_processor = load_rec_processor()
            self.det_model.to(self.device)
            self.rec_model.to(self.device)
            self._run_ocr_fn   = run_ocr
            self.api_version   = "old"
            logger.info("Surya loaded via legacy API")
            return
        except Exception as e:
            logger.debug(f"Strategy 3 failed: {e}")

        raise RuntimeError(
            "All surya strategies failed. Use: python main.py <input> --engine doctr"
        )

    def run(self, image: Image.Image) -> List[Dict]:
        image = image.convert("RGB")
        if self.api_version == "new":
            return self._run_new(image)
        elif self.api_version == "mixed":
            return self._run_mixed(image)
        return self._run_old(image)

    def _run_new(self, image: Image.Image) -> List[Dict]:
        det_results = self.det_predictor([image])
        # Newer Surya (>=0.6): rec_predictor accepts (images, [langs]) and uses
        # internal detection; older 0.6.x needs explicit bbox passing.
        # Try the simpler call first, fall back to bbox-passing.
        try:
            rec_results = self.rec_predictor([image], [self.langs])
            return self._parse_text_lines(rec_results[0].text_lines)
        except TypeError:
            pass
        # Fallback: pass detection bboxes explicitly
        det_page = det_results[0]
        try:
            rec_results = self.rec_predictor([image], [det_page.bboxes], [self.langs])
        except TypeError:
            rec_results = self.rec_predictor([image], det_results, [self.langs])
        return self._parse_text_lines(rec_results[0].text_lines)

    def _run_mixed(self, image: Image.Image) -> List[Dict]:
        det_results = self.det_predictor([image])
        det_page    = det_results[0]
        output = []
        for pb in det_page.bboxes:
            pts = getattr(pb, "polygon", None)
            if pts:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            elif hasattr(pb, "bbox"):
                bbox = list(pb.bbox)
            else:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop([x1, y1, x2, y2])
            try:
                inputs = self.rec_processor(images=crop, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.rec_model.generate(**inputs)
                text = self.rec_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                conf = float(getattr(pb, "confidence", 1.0) or 1.0)
                if text:
                    output.append({"text": text, "bbox": [float(v) for v in bbox],
                                   "confidence": conf, "engine": "surya"})
            except Exception as e:
                logger.debug(f"Mixed recognition failed: {e}")
        return output

    def _run_old(self, image: Image.Image) -> List[Dict]:
        results = self._run_ocr_fn(
            [image], [self.langs],
            self.det_model, self.det_processor,
            self.rec_model, self.rec_processor,
        )
        return self._parse_text_lines(results[0].text_lines)

    def _parse_text_lines(self, text_lines) -> List[Dict]:
        output = []
        for line in text_lines:
            text = (getattr(line, "text", "") or "").strip()
            if not text:
                continue
            pts = getattr(line, "polygon", None)
            if pts and len(pts) >= 2:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            elif hasattr(line, "bbox"):
                bbox = list(line.bbox)
            else:
                continue
            conf = getattr(line, "confidence", None)
            output.append({
                "text":       text,
                "bbox":       [float(v) for v in bbox],
                "confidence": float(conf) if conf is not None else 1.0,
                "engine":     "surya",
            })
        return output


# ---------------------------------------------------------------------------
# Engine 2: DocTR
# ---------------------------------------------------------------------------
class DocTROCREngine:
    # DBNet detection threshold defaults.
    # bin_thresh: binarisation threshold for the probability map (0-1).
    #   Lower = more sensitive, finds fainter/smaller text regions.
    #   Too low = noise regions detected as text.
    # box_thresh: minimum DBNet score for a candidate box to be kept (0-1).
    #   Lower = more boxes kept.
    # Recommended ranges:
    #   Clean digital PDF:   bin_thresh=0.3, box_thresh=0.1  (doctr defaults)
    #   Good quality scan:   bin_thresh=0.2, box_thresh=0.05
    #   Dense/faded scan:    bin_thresh=0.15, box_thresh=0.05
    #   Bank statements:     bin_thresh=0.15, box_thresh=0.05
    DEFAULT_BIN_THRESH = 0.15   # more sensitive than doctr default (0.3)
    DEFAULT_BOX_THRESH = 0.05   # more sensitive than doctr default (0.1)

    # Recognition architecture options (in order of accuracy for printed English):
    #   "parseq"   — transformer, best accuracy for printed English (default)
    #   "master"   — transformer, excellent for financial/KYC documents
    #   "vitstr_small" — ViT-based, strong accuracy, heavier than parseq
    #   "crnn_vgg16_bn" — fast CNN baseline (legacy)
    DEFAULT_RECO_ARCH = "parseq"

    def __init__(
        self,
        device: Optional[str] = None,
        assume_straight: bool = True,
        bin_thresh: float = DEFAULT_BIN_THRESH,
        box_thresh: float = DEFAULT_BOX_THRESH,
        reco_arch: str = DEFAULT_RECO_ARCH,
    ):
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.assume_straight = assume_straight
        self.bin_thresh      = bin_thresh
        self.box_thresh      = box_thresh
        self.reco_arch       = reco_arch
        self._load()

    def _load(self):
        try:
            from doctr.models import ocr_predictor
            logger.info(
                f"Loading DocTR (db_resnet50 + {self.reco_arch}, "
                f"bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh})..."
            )
            self.predictor = ocr_predictor(
                det_arch="db_resnet50", reco_arch=self.reco_arch,
                pretrained=True,
                assume_straight_pages=self.assume_straight,
                bin_thresh=self.bin_thresh,
                box_thresh=self.box_thresh,
            )
            if self.device == "cuda":
                self.predictor = self.predictor.cuda()
            logger.info(f"DocTR loaded ({self.reco_arch})")
        except TypeError:
            # Older doctr versions don't support bin_thresh/box_thresh in constructor.
            # Fall back and set them on the predictor object after construction.
            logger.warning("DocTR version doesn't accept bin_thresh in constructor — patching post-init")
            from doctr.models import ocr_predictor
            self.predictor = ocr_predictor(
                det_arch="db_resnet50", reco_arch=self.reco_arch,
                pretrained=True, assume_straight_pages=self.assume_straight,
            )
            # Patch thresholds directly onto the detection predictor
            try:
                self.predictor.det_predictor.model.postprocessor.bin_thresh = self.bin_thresh
                self.predictor.det_predictor.model.postprocessor.box_thresh = self.box_thresh
                logger.info(f"Patched bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}")
            except AttributeError as e:
                logger.warning(f"Could not patch detection thresholds: {e} — using doctr defaults")
            if self.device == "cuda":
                self.predictor = self.predictor.cuda()
            logger.info("DocTR loaded (fallback path)")
        except Exception as e:
            logger.error(f"DocTR load failed: {e}")
            raise

    def run(self, image: Image.Image) -> List[Dict]:
        import numpy as np
        img_array = np.array(image.convert("RGB"))
        doc  = self.predictor([img_array])
        page = doc.pages[0]
        h, w = page.dimensions
        output = []
        for block in page.blocks:
            # Block-level geometry used as a proxy for detection confidence.
            # DocTR does not expose a separate per-word detection score, so we
            # use the block geometry centre as a stable block identifier.
            block_geo = block.geometry  # ((x1,y1),(x2,y2)) normalised
            # Approximate detection confidence from block area (larger = more reliable)
            block_area = (
                (block_geo[1][0] - block_geo[0][0]) *
                (block_geo[1][1] - block_geo[0][1])
            )
            # Heuristic: larger detected blocks tend to have higher detection confidence.
            # Clamp to [0.5, 1.0] so it is always distinguishable from recognition_conf.
            det_conf_approx = float(min(1.0, 0.5 + block_area * 10))

            for line in block.lines:
                for word in line.words:
                    geo  = word.geometry
                    rec_conf = float(word.confidence)  # true recognition confidence
                    output.append({
                        "text":                    word.value,
                        "bbox":                    [geo[0][0]*w, geo[0][1]*h, geo[1][0]*w, geo[1][1]*h],
                        # recognition_confidence: how sure the model is about the text string
                        "recognition_confidence":  rec_conf,
                        # detection_confidence: proxy for how cleanly the text region was detected
                        "detection_confidence":    det_conf_approx,
                        # confidence: primary score used downstream — use recognition
                        "confidence":              rec_conf,
                        "engine":                  "doctr",
                    })
        return output


# ---------------------------------------------------------------------------
# Engine 3: TrOCR
# ---------------------------------------------------------------------------
class TrOCRRecognizer:
    MODEL_MAP = {
        "printed":           "microsoft/trocr-base-printed",
        "handwritten":       "microsoft/trocr-base-handwritten",
        "printed-large":     "microsoft/trocr-large-printed",
        "handwritten-large": "microsoft/trocr-large-handwritten",
    }

    def __init__(self, variant="printed", device=None, num_beams=4, max_length=128):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams  = num_beams
        self.max_length = max_length
        model_name = self.MODEL_MAP.get(variant, variant)
        logger.info(f"Loading TrOCR: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model     = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("TrOCR loaded")

    @torch.no_grad()
    def recognize(self, crop: Image.Image) -> Tuple[str, float]:
        try:
            pv  = self.processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
            out = self.model.generate(pv, num_beams=self.num_beams, max_length=self.max_length,
                                      output_scores=True, return_dict_in_generate=True)
            text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
            # Confidence: geometric-mean of per-token max-softmax probabilities.
            # out.scores is a tuple of (seq_len,) tensors of shape (batch, vocab).
            # We take softmax over vocab for each step, grab the max (chosen token prob),
            # then average in log-space for numerical stability.
            token_probs = [
                torch.softmax(step[0], dim=-1).max().item()
                for step in out.scores
            ]
            conf = float(np.exp(np.mean(np.log(np.clip(token_probs, 1e-9, 1.0)))) if token_probs else 0.5)
            return text, conf
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}")
            return "", 0.0

    @torch.no_grad()
    def recognize_batch(self, crops: List[Image.Image]) -> List[Tuple[str, float]]:
        return [self.recognize(c) if c and c.size[0] >= 8 and c.size[1] >= 8 else ("", 0.0)
                for c in crops]


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------
class EngineFactory:
    @staticmethod
    def create(config: dict):
        engine_name = config.get("engine", "auto")
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if engine_name in ("surya", "auto"):
            try:
                langs = config.get("surya", {}).get("langs", ["en"])
                return SuryaOCREngine(device=device, langs=langs)
            except Exception as e:
                logger.warning(f"Surya unavailable: {e}")
                if engine_name == "surya":
                    raise

        if engine_name in ("doctr", "auto"):
            try:
                doctr_cfg  = config.get("doctr", {})
                straight   = doctr_cfg.get("assume_straight_pages", True)
                bin_thresh = doctr_cfg.get("bin_thresh", DocTROCREngine.DEFAULT_BIN_THRESH)
                box_thresh = doctr_cfg.get("box_thresh", DocTROCREngine.DEFAULT_BOX_THRESH)
                reco_arch  = doctr_cfg.get("reco_arch", DocTROCREngine.DEFAULT_RECO_ARCH)
                return DocTROCREngine(device=device, assume_straight=straight,
                                     bin_thresh=bin_thresh, box_thresh=box_thresh,
                                     reco_arch=reco_arch)
            except Exception as e:
                logger.warning(f"DocTR unavailable: {e}")
                if engine_name == "doctr":
                    raise

        if engine_name in ("trocr", "auto"):
            try:
                # Support both "trocr" and "recognition" config sections for flexibility
                trocr_cfg  = config.get("trocr", config.get("recognition", {}))
                variant    = trocr_cfg.get("variant",    "printed")
                num_beams  = trocr_cfg.get("num_beams",  4)
                max_len    = trocr_cfg.get("max_length", 128)
                return TrOCRWithDBDetector(device=device, variant=variant,
                                           num_beams=num_beams, max_length=max_len)
            except Exception as e:
                logger.error(f"TrOCR+detector unavailable: {e}")
                raise

        raise RuntimeError(f"No OCR engine could be loaded. engine={engine_name}")


# ---------------------------------------------------------------------------
# TrOCR + DBNet combo
# ---------------------------------------------------------------------------
class TrOCRWithDBDetector:
    """DBNet text detection (from DocTR) + TrOCR recognition (from HuggingFace)."""

    def __init__(self, device, variant="printed", num_beams=4, max_length=128):
        # Use DocTR's detection-only model to avoid loading an unnecessary recognizer
        try:
            from doctr.models import db_resnet50
            logger.info("Loading DBNet detector (detection-only)...")
            self.det_model = db_resnet50(pretrained=True)
            if device == "cuda":
                self.det_model = self.det_model.cuda()
            self.det_model.eval()
            self._det_mode = "standalone"
        except Exception:
            # Fallback: use full ocr_predictor but ignore its recognition output
            from doctr.models import ocr_predictor
            logger.info("Loading DBNet via ocr_predictor fallback...")
            self.det_model = ocr_predictor(
                det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
            )
            self._det_mode = "predictor"

        self.device = device
        self.recognizer = TrOCRRecognizer(
            variant=variant, device=device,
            num_beams=num_beams, max_length=max_length,
        )
        logger.info("TrOCR+DBNet ready")

    def run(self, image: Image.Image) -> List[Dict]:
        import numpy as np
        img_array = np.array(image.convert("RGB"))

        if self._det_mode == "standalone":
            # Use DocTR's preprocessing + standalone detection model
            from doctr.models.preprocessor import PreProcessor
            from doctr.io import DocumentFile
            # Run via the predictor wrapper for consistent bbox format
            from doctr.models import ocr_predictor
            # Cache full predictor on first use
            if not hasattr(self, "_full_predictor"):
                self._full_predictor = ocr_predictor(
                    det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
                )
            doc = self._full_predictor([img_array])
        else:
            doc = self.det_model([img_array])

        page = doc.pages[0]
        h, w = page.dimensions
        crops, bboxes = [], []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geo = word.geometry
                    x1, y1 = int(geo[0][0] * w), int(geo[0][1] * h)
                    x2, y2 = int(geo[1][0] * w), int(geo[1][1] * h)
                    if x2 > x1 + 4 and y2 > y1 + 4:
                        crops.append(image.crop([x1, y1, x2, y2]))
                        bboxes.append([float(x1), float(y1), float(x2), float(y2)])

        texts_confs = self.recognizer.recognize_batch(crops)
        output = []
        for (text, conf), bbox in zip(texts_confs, bboxes):
            if text:
                output.append({
                    "text": text, "bbox": bbox,
                    "confidence": conf, "engine": "trocr+dbnet",
                })
        return output