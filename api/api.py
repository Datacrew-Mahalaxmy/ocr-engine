"""
api.py - FastAPI Document Identifier API
Uses FastAPI instead of Flask
Modified for Vercel serverless deployment (uses temp files)
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
MAX_WORKERS = 1
MAX_FILE_SIZE_MB = 100

# For Vercel, we use temp directories instead of persistent folders
# UPLOAD_FOLDER and PROCESSED_FOLDER are not used with temp files

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
# Thread-Local Processor
# -------------------------------------------------
thread_local = threading.local()


def get_processor():
    """Get or create thread-local processor instance"""
    if not hasattr(thread_local, "processor"):
        try:
            from src.processor import DocumentProcessor

            logger.debug(
                f"Creating processor for thread {threading.get_ident()}"
            )
            thread_local.processor = DocumentProcessor(CONFIG)

        except Exception as e:
            logger.error(f"Failed to create processor: {e}")
            raise

    return thread_local.processor


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
# Parallel Page Processing Function
# -------------------------------------------------
def process_single_page(args):
    """
    Process a single page with its own thread-local processor
    Args:
        args: (page_num, image, temp_dir)
    """
    page_num, image, temp_dir = args

    try:
        processor = get_processor()

        # Save preview to temp directory
        preview_path = os.path.join(temp_dir, f"page_{page_num}.png")
        image.save(preview_path, "PNG")

        results = processor.process_image(image)
        results = processor.post_processor.process(results)

        for r in results:
            r["page"] = page_num

        logger.debug(f"Page {page_num}: {len(results)} regions")

        return {
            "page_num": page_num,
            "results": results,
            "preview_path": preview_path,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Page {page_num} failed: {e}")

        return {
            "page_num": page_num,
            "results": [],
            "preview_path": None,
            "success": False,
            "error": str(e),
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
# Upload and Process Endpoint - MODIFIED FOR VERCEL
# -------------------------------------------------
@app.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    engine: str = "doctr"
):
    """
    Upload and process document with parallel page processing
    Modified for Vercel - uses temporary files instead of persistent storage
    """
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

        # ---------------- PDF Processing ---------------- #
        if file.filename.lower().endswith('.pdf'):
            logger.info("📄 Processing PDF with parallel threads")

            from pdf2image import convert_from_path

            try:
                # Convert PDF to images from temp file
                images = convert_from_path(tmp_path, dpi=400)
                page_count = len(images)
                logger.info(f"   Converted {page_count} pages at 400 DPI")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF conversion failed: {str(e)}"
                )

            if images:
                first_image = images[0]

            process_args = [
                (page_num + 1, image, temp_dir)
                for page_num, image in enumerate(images)
            ]

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

                        if result["success"]:
                            all_results.extend(result["results"])
                            if result["preview_path"]:
                                preview_images.append(result["preview_path"])

                        completed += 1
                        logger.debug(f"Progress: {completed}/{page_count} pages complete")

                    except Exception as e:
                        logger.error(f"Page {page_num} failed: {e}")

            logger.info(f"✅ Processed {page_count} pages with {MAX_WORKERS} threads")

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

            processor = get_processor()
            results = processor.process_image(image)
            results = processor.post_processor.process(results)
            all_results = results
            logger.info(f"   Found {len(results)} regions")

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
        response = {
            "success": True,
            "session_id": session_id,
            "filename": filename,
            "total_regions": len(all_results),
            "text_by_page": {str(k): v for k, v in text_by_page.items()},
            "preview_images": [],  # Can't return file paths in serverless
            "classification": classification,
            "note": "Files saved to temp directory (will be deleted)"
        }

        logger.info(f"✅ Complete for {session_id}")
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
# NEW ENDPOINT: Compare Textract vs DocTR using SBERT
# -------------------------------------------------
@app.post("/compare-with-textract")
async def compare_with_textract(
    textract_json: UploadFile = File(..., description="Textract JSON file"),
    doctr_json: UploadFile = File(..., description="DocTR JSON file"),
    model_name: str = Form("all-MiniLM-L6-v2")
):
    """
    Compare Textract JSON with DocTR JSON using Sentence-BERT
    Returns dashboard metrics with semantic similarity scores
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
            
            # Format for dashboard (matching your image)
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
            
            # Format detailed comparison for display
            for pair in results['aligned_pairs']:
                if pair['doctr']:
                    dashboard['detailed_comparison'].append({
                        "textract": pair['textract']['text'],
                        "doctr": pair['doctr']['text'],
                        "similarity": pair['similarity_percent'],
                        "match_status": "exact" if pair['similarity'] > 0.95 else 
                                      "similar" if pair['similarity'] > 0.7 else "different"
                    })
                else:
                    dashboard['detailed_comparison'].append({
                        "textract": pair['textract']['text'],
                        "doctr": "(missing)",
                        "similarity": 0,
                        "match_status": "missing"
                    })
            
            logger.info(f"✅ Comparison complete. Accuracy: {results['overall_accuracy']}%")
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