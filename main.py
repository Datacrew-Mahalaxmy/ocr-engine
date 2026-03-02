#!/usr/bin/env python3
"""
Transformer OCR System
Pure transformer / deep-learning OCR — no Tesseract, no PaddleOCR.

Engines available (in auto-priority order):
  1. Surya      - SegFormer detection + Donut recognition (best quality)
  2. DocTR      - DBNet detection + PARSeq transformer recognition
  3. TrOCR+DB   - DBNet detection + Microsoft TrOCR recognition

Outputs per region:
  - text                  : recognised string
  - bbox                  : [x1, y1, x2, y2] pixel coordinates
  - confidence            : float 0–1 (geometric mean of recognition confs)
  - detection_confidence  : how cleanly the region was detected
  - recognition_confidence: how sure the model is about the text string
  - page                  : page number
  - engine                : which model produced this result

Usage examples:
  python main.py document.pdf --engine surya --output results/
  python main.py scan.png --engine doctr --visualize
  python main.py docs/ --engine auto --output batch_results/
  python main.py form.pdf --doc-type scanned --dpi 300
  python main.py bank.pdf --doc-type bank_statement --visualize
  python main.py letter.pdf --doc-type scanned --no-preprocess
"""

import argparse
import logging
import sys
import cv2
import numpy as np
from pathlib import Path

import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — supports both flat layout and src/ sub-package layout.
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent


def _try_add(path: Path) -> bool:
    if (path / "processor.py").exists() or (path / "__init__.py").exists():
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        return True
    return False


if not _try_add(_here):
    _try_add(_here / "src")

try:
    from src.processor import DocumentProcessor
    from src.utils import visualize_results, save_results, export_to_txt
except ImportError:
    from processor import DocumentProcessor          # type: ignore[no-redef]
    from utils import visualize_results, save_results, export_to_txt  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Correct skew in scanned documents."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def enhance_image(image: np.ndarray, doc_type: str = "scanned") -> np.ndarray:
    """
    Denoise + CLAHE contrast enhancement.
    For digital docs: lighter processing (no denoising, gentler CLAHE).
    For scanned/bank docs: full denoising + stronger CLAHE.
    """
    if doc_type == "digital":
        # Digital PDFs are clean — only apply gentle contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.merge((l, a, b))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    else:
        # Scanned / bank statement: full pipeline
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.merge((l, a, b))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image


def upscale_if_needed(image: np.ndarray, min_width: int = 1800) -> np.ndarray:
    """Upscale small images so text is large enough for the detector."""
    h, w = image.shape[:2]
    if w < min_width:
        scale = min_width / w
        image = cv2.resize(
            image, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    return image


def preprocess_image(pil_image: Image.Image, doc_type: str = "scanned") -> Image.Image:
    """
    Full pre-processing pipeline:
      Scanned:        deskew → denoise → CLAHE → upscale
      Bank statement: deskew → denoise → CLAHE → upscale
      Digital:        gentle CLAHE only (no deskew, no denoising)

    Returns a PIL Image ready for OCR.
    """
    image = np.array(pil_image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if doc_type in ("scanned", "bank_statement"):
        image = deskew_image(image)

    image = enhance_image(image, doc_type=doc_type)

    # Only upscale for scanned/bank — digital PDFs are already vector-rendered
    if doc_type != "digital":
        image = upscale_if_needed(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


# ---------------------------------------------------------------------------
# Post-processing — column-aware row reconstruction
# ---------------------------------------------------------------------------

def reconstruct_rows(
    results: list,
    y_tolerance: int = 15,
    max_x_gap_fraction: float = 0.30,
) -> list:
    """
    Group OCR word/line results into reading-order rows by Y-coordinate
    proximity, then sort each row left-to-right.

    v5 KEY FIX — column bleeding prevention:
      Two tokens on the same Y-band but separated by more than
      max_x_gap_fraction of page width are treated as belonging to
      DIFFERENT columns and are kept as separate rows.
      This prevents left-sidebar text (names, addresses) from being
      joined with right-body text (paragraphs) on the same line.

    Returns a new list of result dicts — one per reconstructed row.
    """
    if not results:
        return results

    pages: dict = {}
    for r in results:
        p = r.get("page", 1)
        pages.setdefault(p, []).append(r)

    row_results = []

    for page_num, page_words in sorted(pages.items()):
        # Infer page width from metadata or max bbox
        page_w = (
            page_words[0].get("page_width", 0)
            or max(w["bbox"][2] for w in page_words)
            or 1
        )

        # Sort by top-Y first
        page_words.sort(key=lambda x: x["bbox"][1])

        rows: list = []
        current_row: list = [page_words[0]]

        for word in page_words[1:]:
            current_y = word["bbox"][1]
            row_y     = current_row[0]["bbox"][1]

            # Y-proximity check
            if abs(current_y - row_y) > y_tolerance:
                rows.append(current_row)
                current_row = [word]
                continue

            # X-gap fraction check — column bleeding guard
            # Find the rightmost X of everything currently in current_row
            row_x2    = max(w["bbox"][2] for w in current_row)
            word_x1   = word["bbox"][0]
            x_gap     = word_x1 - row_x2

            if page_w > 0 and x_gap > max_x_gap_fraction * page_w:
                # Large horizontal gap → different column → start new row
                rows.append(current_row)
                current_row = [word]
            else:
                current_row.append(word)

        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda x: x["bbox"][0])

            rec_confs = [w.get("recognition_confidence", w.get("confidence", 1.0)) for w in row]
            det_confs = [w.get("detection_confidence",   w.get("confidence", 1.0)) for w in row]
            x1 = min(w["bbox"][0] for w in row)
            y1 = min(w["bbox"][1] for w in row)
            x2 = max(w["bbox"][2] for w in row)
            y2 = max(w["bbox"][3] for w in row)

            # Geometric mean for recognition confidence — penalises single
            # low-confidence words more than arithmetic mean
            geo_mean_rec = float(np.exp(np.mean(
                np.log(np.clip(rec_confs, 1e-9, 1.0))
            )))
            geo_mean_det = float(np.mean(det_confs))

            row_results.append({
                "text":                   " ".join(w["text"] for w in row),
                "bbox":                   [x1, y1, x2, y2],
                "confidence":             round(geo_mean_rec, 4),
                "detection_confidence":   round(geo_mean_det, 4),
                "recognition_confidence": round(geo_mean_rec, 4),
                "page":                   row[0].get("page", page_num),
                "engine":                 row[0].get("engine", "unknown"),
                "word_count":             len(row),
            })

    return row_results


# ---------------------------------------------------------------------------
# DPI helper
# ---------------------------------------------------------------------------

def get_optimal_dpi(doc_type: str, config: dict) -> int:
    """Return the best DPI for the given document type."""
    dpi_settings = config.get("dpi_settings", {})
    defaults = {
        "digital":        150,
        "scanned":        300,
        "id_card":        400,
        "bank_statement": 300,
    }
    return dpi_settings.get(doc_type, defaults.get(doc_type, 300))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("transformer_ocr.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_config_path(given: str) -> str:
    candidates = [
        Path(given),
        _here / "config.yaml",
        _here / "src" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Config file not found. Tried: {[str(c) for c in candidates]}"
    )


def process_single_image(
    processor: "DocumentProcessor",
    image_path: Path,
    output_dir: Path,
    visualize: bool,
    doc_type: str = "scanned",
    apply_preprocessing: bool = True,
    apply_row_reconstruction: bool = True,
    row_tolerance: int = 15,
    max_x_gap_fraction: float = 0.30,
) -> list:
    """
    OCR one image file:
      1. Pre-process (deskew / denoise / upscale) — doc-type aware
      2. Run OCR
      3. Post-process
      4. Row reconstruction (column-aware)
      5. Save JSON + TXT; optionally save visualisation
    """
    image = Image.open(image_path).convert("RGB")

    if apply_preprocessing:
        image = preprocess_image(image, doc_type=doc_type)

    raw_results = processor.process_image(image)
    results     = processor.post_processor.process(raw_results)

    if apply_row_reconstruction:
        results = reconstruct_rows(
            results,
            y_tolerance=row_tolerance,
            max_x_gap_fraction=max_x_gap_fraction,
        )

    stem = image_path.stem
    save_results(results, output_dir / f"{stem}_results.json")
    export_to_txt(results, output_dir / f"{stem}_text.txt")

    if visualize:
        vis_path = output_dir / f"{stem}_vis.png"
        visualize_results(image, results, str(vis_path))

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transformer OCR — extract text, bbox, confidence from any document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan.pdf --engine surya --output results/
  python main.py photo.png --engine doctr --visualize
  python main.py docs/ --engine auto --output batch_results/
  python main.py form.pdf --doc-type scanned --dpi 300
  python main.py bank.pdf --doc-type bank_statement --visualize
  python main.py letter.pdf --doc-type digital --no-preprocess
        """,
    )

    # Positional
    parser.add_argument("input", help="PDF file, image file, or directory of files")

    # Config / engine
    parser.add_argument(
        "--config", default="src/config.yaml",
        help="Path to YAML config file (default: src/config.yaml or config.yaml)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "surya", "doctr", "trocr"],
        default=None,
        help="OCR engine to use (overrides config). Default: auto",
    )

    # Document characteristics
    parser.add_argument(
        "--doc-type",
        choices=["auto", "digital", "scanned", "id_card", "bank_statement"],
        default=None,
        help=(
            "Document type hint. Controls DPI + pre-processing. "
            "digital=clean PDF (no deskew/denoise), scanned=scan, "
            "bank_statement=financial scan, id_card=high-DPI scan"
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help="DPI for PDF rendering (overrides auto). Recommended: 150 digital, 300 scanned, 400 id_card",
    )

    # Pre-processing
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip all image pre-processing (fastest; use for clean high-res digital PDFs)",
    )

    # Row reconstruction
    parser.add_argument(
        "--no-row-reconstruction", action="store_true",
        help="Skip row grouping (keep raw word-level results)",
    )
    parser.add_argument(
        "--row-tolerance", type=int, default=15,
        help="Y-pixel tolerance for grouping words into the same row (default: 15)",
    )
    parser.add_argument(
        "--max-x-gap-fraction", type=float, default=0.30,
        help=(
            "Maximum X-gap (as fraction of page width) allowed when grouping words "
            "into the same row. Larger gap = two-column layouts treated as one row. "
            "Default 0.30 (30%% of page). Lower to 0.15 for dense multi-column docs."
        ),
    )

    # Hardware
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Compute device (default: cuda if available, else cpu)",
    )

    # Output
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Save annotated images with bounding boxes",
    )

    # PDF page range
    parser.add_argument("--start-page", type=int, default=1,   help="First PDF page (1-indexed)")
    parser.add_argument("--end-page",   type=int, default=None, help="Last PDF page (inclusive)")

    # Verbosity
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("=== Transformer OCR System Starting ===")

    # ── Config ────────────────────────────────────────────────────────────────
    try:
        config_path = _resolve_config_path(args.config)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Using config: {config_path}")
    config = load_config(config_path)

    # ── Resolve document type ─────────────────────────────────────────────────
    doc_type = args.doc_type or config.get("document_type", "scanned")
    if doc_type == "auto":
        doc_type = "scanned"  # safe default

    # ── Resolve DPI ───────────────────────────────────────────────────────────
    if args.dpi:
        effective_dpi = args.dpi
    elif config.get("dpi"):
        effective_dpi = config["dpi"]
    else:
        effective_dpi = get_optimal_dpi(doc_type, config)

    logger.info(f"Document type: {doc_type} | DPI: {effective_dpi}")

    # ── CLI overrides into config ──────────────────────────────────────────────
    if args.device:
        config["device"] = args.device
    if args.engine:
        config["engine"] = args.engine
    config["document_type"] = doc_type
    config["dpi"]           = effective_dpi

    # ── Flags ─────────────────────────────────────────────────────────────────
    apply_preprocessing      = not args.no_preprocess
    apply_row_reconstruction = not args.no_row_reconstruction
    row_tolerance            = args.row_tolerance
    max_x_gap_fraction       = args.max_x_gap_fraction

    # Digital docs get pre-processing disabled by default (they are clean)
    if doc_type == "digital" and not apply_preprocessing:
        logger.info("Pre-processing: DISABLED (digital doc)")
    elif apply_preprocessing:
        logger.info(f"Pre-processing: ENABLED (doc_type={doc_type})")
    else:
        logger.info("Pre-processing: DISABLED (--no-preprocess)")

    if apply_row_reconstruction:
        logger.info(
            f"Row reconstruction: ENABLED "
            f"(y_tolerance={row_tolerance}px, max_x_gap={max_x_gap_fraction:.0%})"
        )
    else:
        logger.info("Row reconstruction: DISABLED (--no-row-reconstruction)")

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialise processor (loads models once) ───────────────────────────────
    logger.info("Loading OCR models…")
    processor = DocumentProcessor(config)
    logger.info("Models loaded — starting OCR")

    # ── Shared processing helper ───────────────────────────────────────────────
    def run_pdf_processing(pdf_path_str, start_page, end_page, stem):
        """Process one PDF: pre-process → OCR → reconstruct → save."""
        out = processor.process_pdf(
            pdf_path_str,
            start_page=start_page,
            end_page=end_page,
            return_images=(args.visualize or apply_preprocessing),
        )
        pdf_results = out["results"]

        if apply_preprocessing and "images" in out:
            logger.info(f"  Pre-processing {len(out['images'])} page(s)…")
            pdf_results = []
            for page_img, page_num in zip(
                out["images"],
                range(start_page, start_page + len(out["images"])),
            ):
                processed_img = preprocess_image(page_img, doc_type=doc_type)
                raw           = processor.process_image(processed_img)
                page_results  = processor.post_processor.process(raw)
                for r in page_results:
                    r["page"] = page_num
                pdf_results.extend(page_results)

        if apply_row_reconstruction:
            pdf_results = reconstruct_rows(
                pdf_results,
                y_tolerance=row_tolerance,
                max_x_gap_fraction=max_x_gap_fraction,
            )

        save_results(pdf_results, output_dir / f"{stem}_results.json")
        export_to_txt(pdf_results, output_dir / f"{stem}_text.txt")

        if args.visualize and "images" in out:
            for page_img, page_num in zip(
                out["images"],
                range(start_page, start_page + len(out["images"])),
            ):
                page_res = [r for r in pdf_results if r.get("page") == page_num]
                vis_path = output_dir / f"{stem}_p{page_num}_vis.png"
                vis_img  = preprocess_image(page_img, doc_type=doc_type) \
                           if apply_preprocessing else page_img
                visualize_results(vis_img, page_res, str(vis_path))

        return pdf_results

    # ── Route: single PDF ──────────────────────────────────────────────────────
    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        logger.info(f"Processing PDF: {input_path}")
        results = run_pdf_processing(
            str(input_path), args.start_page, args.end_page, input_path.stem
        )
        logger.info(f"Found {len(results)} text region(s) — saved to {output_dir}")

    # ── Route: single image ────────────────────────────────────────────────────
    elif input_path.is_file() and input_path.suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".heic",
    }:
        logger.info(f"Processing image: {input_path}")
        results = process_single_image(
            processor, input_path, output_dir, args.visualize,
            doc_type=doc_type,
            apply_preprocessing=apply_preprocessing,
            apply_row_reconstruction=apply_row_reconstruction,
            row_tolerance=row_tolerance,
            max_x_gap_fraction=max_x_gap_fraction,
        )
        logger.info(f"Found {len(results)} text region(s) — saved to {output_dir}")

    # ── Route: directory ───────────────────────────────────────────────────────
    elif input_path.is_dir():
        image_exts  = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        pdf_files   = sorted(input_path.glob("*.pdf"))
        image_files = sorted(
            f for f in input_path.iterdir() if f.suffix.lower() in image_exts
        )

        logger.info(
            f"Found {len(pdf_files)} PDF(s) and {len(image_files)} image(s) in {input_path}"
        )

        all_results = []

        for pdf in pdf_files:
            logger.info(f"  PDF: {pdf.name}")
            try:
                res = run_pdf_processing(
                    str(pdf), args.start_page, args.end_page, pdf.stem
                )
                all_results.extend(res)
            except Exception as exc:
                logger.error(f"  Failed: {pdf.name}: {exc}", exc_info=args.debug)

        for img_file in image_files:
            logger.info(f"  Image: {img_file.name}")
            try:
                res = process_single_image(
                    processor, img_file, output_dir, args.visualize,
                    doc_type=doc_type,
                    apply_preprocessing=apply_preprocessing,
                    apply_row_reconstruction=apply_row_reconstruction,
                    row_tolerance=row_tolerance,
                    max_x_gap_fraction=max_x_gap_fraction,
                )
                all_results.extend(res)
            except Exception as exc:
                logger.error(f"  Failed: {img_file.name}: {exc}", exc_info=args.debug)

        if all_results:
            save_results(all_results, output_dir / "all_results.json")
            export_to_txt(all_results, output_dir / "all_text.txt")

        logger.info(
            f"Total: {len(all_results)} region(s) from "
            f"{len(pdf_files)} PDF(s) + {len(image_files)} image(s)"
        )

    # ── Unknown input ──────────────────────────────────────────────────────────
    else:
        logger.error(
            f"Input not found or unsupported file type: {input_path}\n"
            "Supported: .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp, .heic, or a directory"
        )
        sys.exit(1)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()