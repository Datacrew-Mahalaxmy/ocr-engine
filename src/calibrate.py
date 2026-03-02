#!/usr/bin/env python3
"""
calibrate.py — OCR parameter calibration tool

Runs a sweep of detection thresholds on your actual documents and shows
exactly how many text regions are found at each setting, so you can pick
the right values for your config.yaml without guesswork.

Usage:
    python calibrate.py scan.pdf
    python calibrate.py bank_statement.pdf --page 1
    python calibrate.py document.pdf --engine doctr --sweep-all
    python calibrate.py document.pdf --recommend

The tool is READ-ONLY — it never modifies any files.
After finding good values, paste them into your config.yaml doctr section.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.WARNING)  # suppress engine noise during calibration

# ── Detection threshold sweep ────────────────────────────────────────────────

BIN_THRESH_VALUES  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # DBNet binarization
BOX_THRESH_VALUES  = [0.03, 0.05, 0.07, 0.10]                # DBNet box keep threshold
MERGE_VALUES       = [8, 12, 15, 20, 25, 30]                  # post-process merge gap
MIN_CONF_VALUES    = [0.1, 0.2, 0.3, 0.4, 0.5]               # confidence filter


def pdf_to_image(pdf_path: str, page: int = 1, dpi: int = 300):
    """Convert a single PDF page to a PIL image."""
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page, last_page=page)
    if not images:
        raise ValueError(f"Could not render page {page} from {pdf_path}")
    return images[0]


def run_doctr(image, bin_thresh: float, box_thresh: float) -> List[Dict]:
    """Run DocTR with specific detection thresholds. Returns raw word dicts."""
    import numpy as np
    try:
        from doctr.models import ocr_predictor
    except ImportError:
        print("ERROR: doctr not installed. Install with: pip install python-doctr")
        sys.exit(1)

    try:
        predictor = ocr_predictor(
            det_arch="db_resnet50", reco_arch="parseq",
            pretrained=True,
            bin_thresh=bin_thresh,
            box_thresh=box_thresh,
        )
    except TypeError:
        # Older doctr — patch post-init
        predictor = ocr_predictor(
            det_arch="db_resnet50", reco_arch="parseq", pretrained=True
        )
        try:
            predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
            predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
        except AttributeError:
            pass

    img_array = np.array(image.convert("RGB"))
    doc  = predictor([img_array])
    page = doc.pages[0]
    h, w = page.dimensions

    words = []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                geo = word.geometry
                words.append({
                    "text":       word.value,
                    "bbox":       [geo[0][0]*w, geo[0][1]*h, geo[1][0]*w, geo[1][1]*h],
                    "confidence": float(word.confidence),
                })
    return words


def apply_conf_filter(words: List[Dict], min_conf: float) -> List[Dict]:
    return [w for w in words if w["confidence"] >= min_conf and len(w["text"].strip()) >= 2]


def words_preview(words: List[Dict], n: int = 8) -> str:
    texts = [w["text"][:20] for w in words[:n]]
    suffix = "..." if len(words) > n else ""
    return "  |  ".join(texts) + suffix


# ── Sweep functions ──────────────────────────────────────────────────────────

def sweep_detection(image, verbose: bool = False):
    """Sweep bin_thresh × box_thresh and show token counts."""
    print("\n" + "═"*70)
    print("  DETECTION THRESHOLD SWEEP  (bin_thresh × box_thresh)")
    print("  More tokens = better recall. Stop before noise spikes.")
    print("═"*70)
    print(f"  {'bin_thresh':>10} {'box_thresh':>10} {'tokens':>8}  {'sample text'}")
    print("  " + "-"*66)

    results = {}
    for bt in BIN_THRESH_VALUES:
        for bx in BOX_THRESH_VALUES:
            try:
                words = run_doctr(image, bt, bx)
                n = len(words)
                results[(bt, bx)] = (n, words)
                preview = words_preview(words)
                flag = "  ← try this" if 40 <= n <= 200 else ("  ← may be too few" if n < 20 else "  ← may include noise" if n > 300 else "")
                print(f"  {bt:>10.2f} {bx:>10.2f} {n:>8}  {preview[:40]}{flag}")
                if verbose:
                    for w in words[:5]:
                        print(f"            conf={w['confidence']:.3f} | {w['text'][:60]}")
            except Exception as e:
                print(f"  {bt:>10.2f} {bx:>10.2f} {'ERR':>8}  {e}")
    return results


def sweep_merge(words: List[Dict]):
    """Show how merge_threshold affects final line count."""
    print("\n" + "═"*70)
    print("  MERGE THRESHOLD SWEEP")
    print("  Goal: each logical line of text = 1 merged token.")
    print("  Too low = fragmented words. Too high = separate lines merged.")
    print("═"*70)
    print(f"  {'merge_threshold':>16} {'tokens_in':>10} {'tokens_out':>11}  {'sample'}")
    print("  " + "-"*66)

    # Import PostProcessor
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from utils import PostProcessor
    except ImportError:
        print("  WARNING: could not import utils.PostProcessor — skipping merge sweep")
        return

    for mt in MERGE_VALUES:
        # Add dummy page/page_width data
        enriched = [{**w, "page": 1, "page_width": 2000, "page_height": 3000} for w in words]
        pp = PostProcessor({
            "min_confidence": 0.0,   # already filtered
            "min_text_length": 1,
            "nms_iou_threshold": 0.5,
            "filter_noise": False,
            "column_detection": True,
            "enable_reading_order": True,
            "enable_merge": True,
            "merge_threshold": mt,
            "large_text_merge_boost": 2.5,
            "large_text_height_px": 40,
        })
        try:
            out = pp.process(enriched)
            preview = "  |  ".join([r["text"][:15] for r in out[:5]])
            print(f"  {mt:>16} {len(enriched):>10} {len(out):>11}  {preview}")
        except Exception as e:
            print(f"  {mt:>16} {'ERR':>10} {'ERR':>11}  {e}")


def sweep_confidence(words_all: List[Dict]):
    """Show how confidence threshold affects token count."""
    print("\n" + "═"*70)
    print("  CONFIDENCE THRESHOLD SWEEP")
    print("  Shows how many tokens survive at each min_confidence level.")
    print("  Too high = real text dropped. Too low = noise included.")
    print("═"*70)
    confs = sorted([w["confidence"] for w in words_all])
    if not confs:
        print("  No words to analyse.")
        return

    print(f"\n  Confidence distribution (n={len(confs)}):")
    buckets = {}
    for c in confs:
        b = int(c * 10) / 10
        buckets[b] = buckets.get(b, 0) + 1
    for b in sorted(buckets):
        bar = "█" * min(40, buckets[b])
        pct = buckets[b] / len(confs) * 100
        print(f"    {b:.1f}-{b+0.1:.1f}: {bar} {buckets[b]} ({pct:.0f}%)")

    print(f"\n  {'min_confidence':>16} {'tokens_kept':>12} {'tokens_dropped':>15}")
    print("  " + "-"*46)
    for mc in MIN_CONF_VALUES:
        kept    = sum(1 for c in confs if c >= mc)
        dropped = len(confs) - kept
        bar     = "▓" * int(kept / max(1, len(confs)) * 20)
        print(f"  {mc:>16.1f} {kept:>12} {dropped:>15}   {bar}")


def recommend(image, dpi: int):
    """Run a quick calibration and print recommended config.yaml values."""
    print("\n" + "═"*70)
    print("  QUICK CALIBRATION — finding recommended settings")
    print("═"*70)

    # Quick scan of bin_thresh values with fixed box_thresh=0.05
    print("\n  Testing detection sensitivity...")
    best_bt   = 0.15
    best_n    = 0
    best_words = []

    for bt in [0.30, 0.20, 0.15, 0.10, 0.05]:
        try:
            words = run_doctr(image, bt, 0.05)
            n = len(words)
            print(f"    bin_thresh={bt:.2f}: {n} tokens found")
            # Target: 30-300 tokens is a healthy range for a single page
            if n > best_n and n <= 400:
                best_n = n
                best_bt = bt
                best_words = words
        except Exception as e:
            print(f"    bin_thresh={bt:.2f}: ERROR ({e})")

    # Analyse confidence distribution on best result
    confs = [w["confidence"] for w in best_words]
    if confs:
        mean_conf = sum(confs) / len(confs)
        low_count  = sum(1 for c in confs if c < 0.3)
        # Recommend min_confidence just below the natural confidence floor
        if mean_conf > 0.85:
            rec_conf = 0.2
        elif mean_conf > 0.7:
            rec_conf = 0.15
        else:
            rec_conf = 0.1
    else:
        rec_conf = 0.2

    # Recommend merge_threshold based on median token height
    heights = [w["bbox"][3] - w["bbox"][1] for w in best_words] if best_words else [20]
    med_h = sorted(heights)[len(heights)//2] if heights else 20
    rec_merge = 12 if med_h < 20 else (15 if med_h < 35 else 20)

    print("\n" + "─"*70)
    print("  RECOMMENDED config.yaml settings:")
    print("─"*70)
    print(f"""
  # Copy these into your config.yaml:

  min_confidence: {rec_conf}

  doctr:
    det_arch: "db_resnet50"
    reco_arch: "parseq"
    assume_straight_pages: true
    bin_thresh: {best_bt}       # ← calibrated for this document type
    box_thresh: 0.05            # ← safe value for dense/scanned docs

  post_process:
    min_confidence: {rec_conf}
    min_text_length: 2
    merge_threshold: {rec_merge}  # ← based on median text height of {med_h:.0f}px
    large_text_merge_boost: 2.5
    large_text_height_px: 40

  # Stats: {best_n} tokens found at bin_thresh={best_bt}
  # Median token height: {med_h:.1f}px, Mean confidence: {mean_conf:.3f}
""")

    if best_words:
        print("  Sample extracted text:")
        for w in best_words[:12]:
            print(f"    conf={w['confidence']:.3f} | {w['text'][:60]}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate OCR parameters for a specific document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibrate.py scan.pdf                    # full sweep
  python calibrate.py bank.pdf --recommend        # quick recommendation only
  python calibrate.py form.pdf --page 3           # test page 3
  python calibrate.py doc.pdf --sweep-detection   # only detection sweep
  python calibrate.py doc.pdf --sweep-merge       # only merge sweep
  python calibrate.py doc.pdf --dpi 200           # use 200 dpi
        """
    )
    parser.add_argument("pdf", help="PDF file to calibrate on")
    parser.add_argument("--page",            type=int,   default=1,     help="Page to use (default: 1)")
    parser.add_argument("--dpi",             type=int,   default=300,   help="DPI for rendering (default: 300)")
    parser.add_argument("--recommend",       action="store_true",        help="Quick recommendation only")
    parser.add_argument("--sweep-all",       action="store_true",        help="Full sweep of all parameters")
    parser.add_argument("--sweep-detection", action="store_true",        help="Sweep detection thresholds only")
    parser.add_argument("--sweep-merge",     action="store_true",        help="Sweep merge thresholds only")
    parser.add_argument("--sweep-confidence",action="store_true",        help="Sweep confidence thresholds only")
    parser.add_argument("--verbose",         action="store_true",        help="Show sample tokens in detection sweep")
    parser.add_argument("--bin-thresh",      type=float, default=0.15,   help="Fix bin_thresh for merge/conf sweeps")
    parser.add_argument("--box-thresh",      type=float, default=0.05,   help="Fix box_thresh for merge/conf sweeps")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"ERROR: File not found: {args.pdf}")
        sys.exit(1)

    print(f"\n  Document: {args.pdf}")
    print(f"  Page:     {args.page}")
    print(f"  DPI:      {args.dpi}")

    print("\n  Rendering page...")
    try:
        image = pdf_to_image(args.pdf, page=args.page, dpi=args.dpi)
        print(f"  Page size: {image.size[0]}×{image.size[1]}px")
    except Exception as e:
        print(f"  ERROR rendering PDF: {e}")
        sys.exit(1)

    # Default: run recommend + detection sweep
    run_detect  = args.sweep_all or args.sweep_detection or (not args.recommend and not args.sweep_merge and not args.sweep_confidence)
    run_merge   = args.sweep_all or args.sweep_merge
    run_conf    = args.sweep_all or args.sweep_confidence
    run_rec     = args.recommend or (not run_detect and not run_merge and not run_conf)

    if run_rec:
        recommend(image, args.dpi)

    if run_detect:
        det_results = sweep_detection(image, verbose=args.verbose)
        # Use best result for downstream sweeps
        best = max(det_results.items(), key=lambda x: x[1][0] if x[1][0] <= 400 else 0)
        best_words = best[1][1]
        print(f"\n  → Best detection: bin_thresh={best[0][0]}, box_thresh={best[0][1]} → {best[1][0]} tokens")
    else:
        print(f"\n  Running DocTR with bin_thresh={args.bin_thresh}, box_thresh={args.box_thresh}...")
        best_words = run_doctr(image, args.bin_thresh, args.box_thresh)
        best_words = apply_conf_filter(best_words, 0.1)
        print(f"  Found {len(best_words)} tokens")

    if run_merge and best_words:
        filtered = apply_conf_filter(best_words, 0.2)
        sweep_merge(filtered)

    if run_conf and best_words:
        sweep_confidence(best_words)

    print("\n  Done. Copy the recommended settings to your config.yaml.\n")


if __name__ == "__main__":
    main()