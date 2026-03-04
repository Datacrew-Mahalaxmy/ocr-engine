"""
api.py - FastAPI Document Identifier API with proper parallel processing
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import yaml
import uuid
import json
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import shutil
from .sbert_similarity import compare_json_files, get_comparator
import tempfile
import time
import multiprocessing
from functools import lru_cache
from starlette.exceptions import HTTPException as StarletteHTTPException

os.environ["DOCTR_CACHE_DIR"] = "/tmp"  # Use temp dir

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ocr-api")

# -------------------------------------------------
# App Initialization
# -------------------------------------------------
app = FastAPI(title="Document Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://project-ocr-final.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
MAX_WORKERS = min(4, multiprocessing.cpu_count())  # Don't exceed CPU cores
MAX_FILE_SIZE_MB = 100

# -------------------------------------------------
# Load YAML Config
# -------------------------------------------------
try:
    config_path = Path("src/config.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml")

    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    CONFIG["engine"] = "doctr"
    logger.info(f"Loaded config from {config_path} with engine=doctr")

except Exception as e:
    logger.error(f"Failed to load config: {e}")
    CONFIG = {"engine": "doctr"}

# -------------------------------------------------
# GLOBAL PROCESSOR (SINGLETON) - FIX FOR PARALLEL PROCESSING
# -------------------------------------------------
_global_processor = None
_processor_lock = threading.Lock()

def get_global_processor():
    """Get or create a SINGLE global processor instance (not thread-local)"""
    global _global_processor
    if _global_processor is None:
        with _processor_lock:
            if _global_processor is None:
                try:
                    from src.processor import DocumentProcessor
                    logger.info("🚀 Creating GLOBAL processor instance (shared across threads)")
                    start_time = time.time()
                    _global_processor = DocumentProcessor(CONFIG)
                    logger.info(f"✅ GLOBAL processor created in {time.time() - start_time:.2f}s")
                except Exception as e:
                    logger.error(f"Failed to create global processor: {e}")
                    raise
    return _global_processor

# -------------------------------------------------
# Load Classifier (Singleton)
# -------------------------------------------------
try:
    from custom_classifier import get_classifier
    classifier = get_classifier()
    logger.info("✅ Document classifier loaded")
except Exception as e:
    logger.warning(f"⚠️ Could not load classifier: {e}")
    classifier = None

# -------------------------------------------------
# Parallel Page Processing Function (USES GLOBAL PROCESSOR)
# -------------------------------------------------
def process_single_page(args):
    """
    Process a single page with the SHARED global processor
    """
    page_num, image, temp_dir = args
    
    thread_id = threading.get_ident()
    start_time = time.time()

    try:
        # Use the global processor (shared across threads)
        processor = get_global_processor()

        # Save preview to temp directory
        preview_path = os.path.join(temp_dir, f"page_{page_num}.png")
        image.save(preview_path, "PNG")

        # Process the image
        results = processor.process_image(image)
        results = processor.post_processor.process(results)

        for r in results:
            r["page"] = page_num

        elapsed = time.time() - start_time
        logger.debug(f"Page {page_num} (thread {thread_id}): {len(results)} regions in {elapsed:.2f}s")

        return {
            "page_num": page_num,
            "results": results,
            "preview_path": preview_path,
            "success": True,
            "processing_time": elapsed
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Page {page_num} failed after {elapsed:.2f}s: {e}")

        return {
            "page_num": page_num,
            "results": [],
            "preview_path": None,
            "success": False,
            "error": str(e),
            "processing_time": elapsed
        }

# -------------------------------------------------
# Helper function for secure filename
# -------------------------------------------------
def secure_filename(filename: str) -> str:
    """Secure filename - simplified version"""
    import re
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename

# -------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "doctr",
        "parallel_workers": MAX_WORKERS,
        "time": datetime.utcnow().isoformat()
    }

# -------------------------------------------------
# Upload and Process Endpoint - OPTIMIZED
# -------------------------------------------------
@app.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    engine: str = "doctr"
):
    """
    Upload and process document with optimized parallel page processing
    """
    overall_start = time.time()
    logger.info(f"📤 Received upload request: {file.filename}")

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.filename == "":
        raise HTTPException(status_code=400, detail="Empty filename")

    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)"
        )

    session_id = str(uuid.uuid4())[:8]
    logger.info(f"🔑 Session: {session_id}, File: {file.filename}")

    # Create a temporary directory for this session
    temp_dir = tempfile.mkdtemp()
    filename = secure_filename(file.filename)
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        all_results = []
        preview_images = []
        first_image = None
        page_times = []

        # ---------------- PDF Processing ---------------- #
        if file.filename.lower().endswith('.pdf'):
            logger.info("📄 Processing PDF with optimized parallel threads")

            from pdf2image import convert_from_path

            try:
                # Convert PDF to images from temp file
                convert_start = time.time()
                images = convert_from_path(tmp_path, dpi=400)
                page_count = len(images)
                logger.info(f"   Converted {page_count} pages in {time.time() - convert_start:.2f}s at 400 DPI")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF conversion failed: {str(e)}"
                )

            if images:
                first_image = images[0]

            # WARM UP: Initialize global processor before parallel processing
            logger.info("🔥 Warming up global processor...")
            warmup_start = time.time()
            get_global_processor()
            logger.info(f"✅ Global processor warmed up in {time.time() - warmup_start:.2f}s")

            process_args = [
                (page_num + 1, image, temp_dir)
                for page_num, image in enumerate(images)
            ]

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_page = {
                    executor.submit(process_single_page, args): args[0]
                    for args in process_args
                }

                completed = 0
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]

                    try:
                        result = future.result(timeout=300)
                        page_times.append(result.get('processing_time', 0))

                        if result["success"]:
                            all_results.extend(result["results"])
                            if result["preview_path"]:
                                preview_images.append(result["preview_path"])

                        completed += 1
                        if completed % 5 == 0 or completed == page_count:
                            logger.info(f"Progress: {completed}/{page_count} pages complete")

                    except Exception as e:
                        logger.error(f"Page {page_num} failed: {e}")

            avg_page_time = sum(page_times) / len(page_times) if page_times else 0
            logger.info(f"✅ Processed {page_count} pages with {MAX_WORKERS} threads. "
                       f"Avg page time: {avg_page_time:.2f}s, Total: {time.time() - overall_start:.2f}s")

        # ---------------- Image Processing ---------------- #
        else:
            logger.info("🖼️ Processing image file")
            
            # Open image from temp file
            image = Image.open(tmp_path).convert("RGB")
            first_image = image

            # Save preview to temp directory
            preview_path = os.path.join(temp_dir, "page_1.png")
            image.save(preview_path, "PNG")
            preview_images.append(preview_path)

            processor = get_global_processor()
            results = processor.process_image(image)
            results = processor.post_processor.process(results)
            all_results = results
            logger.info(f"   Found {len(results)} regions in {time.time() - overall_start:.2f}s")

        # ---------------- Save Results to Temp ---------------- #
        text_by_page = {}
        classification = {
            "document_type": "unknown",
            "raw_type": "unknown",
            "confidence": 0.0,
        }

        if all_results:
            # Save JSON to temp file
            json_path = os.path.join(temp_dir, "results.json")

            clean_results = []
            for r in all_results:
                clean_r = {
                    k: v for k, v in r.items()
                    if k not in ["_col", "crop"]
                    and isinstance(v, (str, int, float, bool, list, dict))
                }
                clean_results.append(clean_r)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)

            # Save text file to temp
            txt_path = os.path.join(temp_dir, "text.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                current_page = None
                for r in all_results:
                    page = r.get('page', 1)
                    if page != current_page:
                        if current_page is not None:
                            f.write('\n\n')
                        f.write(f"--- Page {page} ---\n")
                        current_page = page
                    f.write(r.get('text', '') + '\n')

            # Group text by page
            for r in all_results:
                page = r.get("page", 1)
                text_by_page.setdefault(page, []).append(r.get("text", ""))

            for page in text_by_page:
                text_by_page[page] = "\n".join(text_by_page[page])

            # ---------------- Classification ---------------- #
            try:
                from custom_classifier import classify_document
                logger.info("🔍 Running document classification...")
                
                classification = classify_document(all_results, first_image)

                if classification:
                    if 'confidence' in classification:
                        classification["confidence"] = float(
                            classification.get("confidence", 0.0)
                        )
                    if 'scores' in classification:
                        classification['scores'] = {str(k): float(v) for k, v in classification['scores'].items()}
                    
                    logger.info(f"✅ Classification: {classification.get('document_type', 'unknown')} "
                              f"(confidence: {classification.get('confidence', 0):.2f})")

            except Exception as e:
                logger.error(f"Classification failed: {e}")
                classification = {
                    'document_type': 'unknown',
                    'raw_type': 'unknown',
                    'confidence': 0.0,
                    'error': str(e)
                }

        # ---------------- Response ---------------- #
        total_time = time.time() - overall_start
        response = {
            "success": True,
            "session_id": session_id,
            "filename": filename,
            "total_regions": len(all_results),
            "text_by_page": {str(k): v for k, v in text_by_page.items()},
            "preview_images": [],
            "classification": classification,
            "processing_time": round(total_time, 2),
            "pages_processed": len(text_by_page),
            "note": "Files saved to temp directory (will be deleted)"
        }

        logger.info(f"✅ Complete for {session_id} in {total_time:.2f}s")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# -------------------------------------------------
# [Rest of your endpoints remain the same...]
# -------------------------------------------------


# -------------------------------------------------
# NEW ENDPOINT: Compare Textract vs DocTR using SBERT
# -------------------------------------------------
# -------------------------------------------------
# COMPARE WITH TEXTRACT ENDPOINT - FIXED VERSION
# -------------------------------------------------
@app.post("/compare-with-textract")
async def compare_with_textract(
    textract_json: UploadFile = File(..., description="Textract JSON file"),
    doctr_json: UploadFile = File(..., description="DocTR JSON file"),
    model_name: str = Form("all-MiniLM-L6-v2")
):
    """
    Compare Textract JSON with DocTR JSON using Sentence-BERT
    Returns dashboard metrics with detailed comparison
    """
    logger.info(f"📊 Comparing Textract vs DocTR with model: {model_name}")
    logger.info(f"   Textract file: {textract_json.filename}")
    logger.info(f"   DocTR file: {doctr_json.filename}")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save Textract JSON
            textract_path = os.path.join(temp_dir, "textract.json")
            with open(textract_path, "wb") as f:
                shutil.copyfileobj(textract_json.file, f)
            
            # Save DocTR JSON
            doctr_path = os.path.join(temp_dir, "doctr.json")
            with open(doctr_path, "wb") as f:
                shutil.copyfileobj(doctr_json.file, f)
            
            # Compare using SBERT
            logger.info("🔍 Running SBERT comparison...")
            results = compare_json_files(textract_path, doctr_path, model_name)
            
            # Format for dashboard with proper detailed comparison
            dashboard = {
                "document": {
                    "name": doctr_json.filename.replace('.json', '.pdf'),
                    "reference": textract_json.filename,
                    "engine": "DocTR (Live)"
                },
                "metrics": {
                    "overall_accuracy": {
                        "value": results['overall_accuracy'],
                        "status": "PASS" if results['overall_accuracy'] >= 85 else "FAIL"
                    },
                    "semantic_similarity": {
                        "value": results['semantic_similarity'],
                        "display": f"{results['semantic_similarity']}%"
                    },
                    "word_error_rate": {
                        "value": results['word_error_rate'],
                        "display": f"{results['word_error_rate']}%"
                    },
                    "character_error_rate": {
                        "value": results['character_error_rate'],
                        "display": f"{results['character_error_rate']}%"
                    },
                    "processing_time": {
                        "value": results['processing_time'],
                        "display": results['processing_time_display']
                    }
                },
                "detailed_comparison": [],
                "stats": results['stats']
            }
            
            # FIX: Populate detailed_comparison from aligned_pairs
            if 'aligned_pairs' in results and results['aligned_pairs']:
                logger.info(f"   Found {len(results['aligned_pairs'])} aligned pairs")
                
                for pair in results['aligned_pairs']:
                    # Handle matched pairs (both textract and doctr exist)
                    if pair.get('doctr') and pair.get('textract'):
                        dashboard['detailed_comparison'].append({
                            "textract_text": pair['textract']['text'],
                            "doctr_text": pair['doctr']['text'],
                            "similarity_score": pair['similarity_percent'],
                            "match_status": "exact" if pair['similarity'] > 0.95 else 
                                          "similar" if pair['similarity'] > 0.7 else "different"
                        })
                    # Handle missing DocTR matches (textract only)
                    elif pair.get('textract') and not pair.get('doctr'):
                        dashboard['detailed_comparison'].append({
                            "textract_text": pair['textract']['text'],
                            "doctr_text": "(missing)",
                            "similarity_score": 0,
                            "match_status": "missing"
                        })
                    # Handle missing Textract matches (doctr only) - optional
                    elif pair.get('doctr') and not pair.get('textract'):
                        dashboard['detailed_comparison'].append({
                            "textract_text": "(missing)",
                            "doctr_text": pair['doctr']['text'],
                            "similarity_score": 0,
                            "match_status": "missing"
                        })
            
            # Add summary stats for debugging
            dashboard['summary'] = {
                'total_pairs': len(dashboard['detailed_comparison']),
                'matched_pairs': results['stats'].get('matched_pairs', 0),
                'unmatched_textract': results['stats'].get('unmatched_textract', 0),
                'textract_blocks': results['stats'].get('textract_blocks', 0),
                'doctr_blocks': results['stats'].get('doctr_blocks', 0)
            }
            
            logger.info(f"✅ Comparison complete. Accuracy: {results['overall_accuracy']}%")
            logger.info(f"   Detailed comparison has {len(dashboard['detailed_comparison'])} items")
            
            return JSONResponse(content=dashboard)
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# Get Preview Image Endpoint - MODIFIED FOR VERCEL
# -------------------------------------------------
@app.get("/preview/{session_id}/{page}")
async def get_preview(session_id: str, page: int):
    """Get preview image - Note: This won't work in Vercel without persistent storage"""
    raise HTTPException(
        status_code=501,
        detail="Preview images not available in serverless deployment. Text results are returned in the response."
    )


# -------------------------------------------------
# Download Results Endpoint - MODIFIED FOR VERCEL
# -------------------------------------------------
@app.get("/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """Download result files - Note: This won't work in Vercel without persistent storage"""
    raise HTTPException(
        status_code=501,
        detail="File download not available in serverless deployment. Text results are returned in the response."
    )


# -------------------------------------------------
# List Sessions Endpoint - MODIFIED FOR VERCEL
# -------------------------------------------------
@app.get("/sessions")
async def list_sessions():
    """List all processing sessions - Not available in serverless"""
    raise HTTPException(
        status_code=501,
        detail="Session listing not available in serverless deployment"
    )


# -------------------------------------------------
# Supported Types Endpoint
# -------------------------------------------------
@app.get("/supported_types")
async def get_supported_types():
    """Get list of supported document types"""
    try:
        from custom_classifier import get_classifier
        classifier = get_classifier()
        types = list(classifier.DOCUMENT_SIGNATURES.keys())
        return JSONResponse(content={
            'count': len(types),
            'types': types
        })
    except Exception as e:
        logger.error(f"❌ Failed to get supported types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Error Handlers
# -------------------------------------------------
@app.exception_handler(413)
async def request_entity_too_large_handler(request: Request, exc):
    return JSONResponse(
        status_code=413,
        content={"error": f"File too large (max {MAX_FILE_SIZE_MB}MB)"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# -------------------------------------------------
# Startup Event
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 Starting Document Identifier API (FastAPI) for Vercel")
    logger.info("⚙️  Engine: doctr")
    logger.info(f"⚡ Parallel workers: {MAX_WORKERS}")
    logger.info("📁 Using temporary files (serverless mode)")
    logger.info("=" * 50)


# -------------------------------------------------
# For Vercel serverless deployment
# -------------------------------------------------
app = app  # Vercel looks for 'app' variable


# For running locally
if __name__ == "__main__":
    import uvicorn
    logger.info("Running locally on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)