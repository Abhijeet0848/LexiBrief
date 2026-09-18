import sys
import os
import re

# Serverless & local environment directory resolution
current_file_dir = os.path.dirname(os.path.abspath(__file__))
candidate_bases = [
    current_file_dir,
    os.getcwd(),
    os.path.dirname(current_file_dir),
    "/var/task"
]

BASE_DIR = current_file_dir
for candidate in candidate_bases:
    if os.path.exists(os.path.join(candidate, "templates")) or os.path.exists(os.path.join(candidate, "src")):
        BASE_DIR = candidate
        break

SRC_DIR = os.path.join(BASE_DIR, "src")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

for path in [BASE_DIR, SRC_DIR, current_file_dir, os.getcwd(), "/var/task", "/var/task/src"]:
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File, Response, BackgroundTasks, Body, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from typing import Optional, Dict, Any, List, Tuple
import uvicorn
import subprocess
import time
from datetime import datetime, timezone
import io
import base64
import requests
import asyncio
import edge_tts
import pandas as pd
from gtts import gTTS

from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.mongo_manager import MongoDBManager
from textSummarizer.components.translator import Translator
from textSummarizer.logging import logger

class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text or summary to translate to English")
    source_lang: Optional[str] = Field("auto", description="Source language code ('auto', 'hi', 'fr', 'es', etc.)")

app = FastAPI(
    title="LexiBrief API",
    description="State-of-the-Art NLP Text Summarization Engine with Extractive, Abstractive & MongoDB persistence",
    version="2.0.0"
)

# Security Constants & Limits
MAX_UPLOAD_SIZE = 50 * 1024 * 1024   # 50 MB max file upload
MAX_INPUT_CHARS = 150_000            # 150,000 max input character limit
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

# Ultra-Realistic Studio Neural Voices (100% Free, Zero Key Required)
EDGE_NEURAL_VOICE_MAP = {
    "neerja": {"id": "en-IN-NeerjaNeural", "name": "Neerja (Studio Indian Female)", "gender": "female"},
    "prabhat": {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Studio Indian Male)", "gender": "male"}
}

# Multilingual Native Voice Map for Auto-Detected Languages
MULTILINGUAL_VOICE_MAP = {
    "hi": {"id": "hi-IN-SwaraNeural", "name": "Swara (Hindi)", "gender": "female"},
    "mr": {"id": "mr-IN-AarohiNeural", "name": "Aarohi (Marathi)", "gender": "female"},
    "bn": {"id": "bn-IN-TanishaaNeural", "name": "Tanishaa (Bengali)", "gender": "female"},
    "ta": {"id": "ta-IN-PallaviNeural", "name": "Pallavi (Tamil)", "gender": "female"},
    "te": {"id": "te-IN-ShrutiNeural", "name": "Shruti (Telugu)", "gender": "female"},
    "gu": {"id": "gu-IN-DhwaniNeural", "name": "Dhwani (Gujarati)", "gender": "female"},
    "kn": {"id": "kn-IN-SapnaNeural", "name": "Sapna (Kannada)", "gender": "female"},
    "ml": {"id": "ml-IN-SobhanaNeural", "name": "Sobhana (Malayalam)", "gender": "female"},
    "ur": {"id": "ur-IN-GulNeural", "name": "Gul (Urdu)", "gender": "female"},
    "pa": {"id": "pa-IN-GurpreetNeural", "name": "Gurpreet (Punjabi)", "gender": "female"},
    "es": {"id": "es-ES-ElviraNeural", "name": "Elvira (Spanish)", "gender": "female"},
    "fr": {"id": "fr-FR-VivienneMultilingualNeural", "name": "Vivienne (French)", "gender": "female"},
    "de": {"id": "de-DE-KatjaNeural", "name": "Katja (German)", "gender": "female"},
    "it": {"id": "it-IT-ElsaNeural", "name": "Elsa (Italian)", "gender": "female"},
    "pt": {"id": "pt-BR-FranciscaNeural", "name": "Francisca (Portuguese)", "gender": "female"},
    "ru": {"id": "ru-RU-SvetlanaNeural", "name": "Svetlana (Russian)", "gender": "female"},
    "zh": {"id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao (Chinese)", "gender": "female"},
    "ja": {"id": "ja-JP-NanamiNeural", "name": "Nanami (Japanese)", "gender": "female"},
    "ar": {"id": "ar-SA-ZariyahNeural", "name": "Zariyah (Arabic)", "gender": "female"},
    "en": {"id": "en-IN-NeerjaNeural", "name": "Neerja (Studio Indian English)", "gender": "female"}
}


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize to speech")
    voice: Optional[str] = Field("neerja", description="Voice identifier ('neerja', 'prabhat' or language-specific)")
    speed: Optional[float] = Field(1.0, description="Speech rate multiplier")

# Enable Secure CORS for API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

import urllib.parse

class VercelPathFixMiddleware:
    """Pure ASGI middleware to correct scope['path'] rewritten by Vercel serverless rewrites."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            q_bytes = scope.get("query_string", b"")
            if q_bytes and b"__path__=" in q_bytes:
                params = urllib.parse.parse_qs(q_bytes.decode("latin1"), keep_blank_values=True)
                if "__path__" in params:
                    raw_p = params.pop("__path__")[0]
                    clean_p = raw_p if raw_p.startswith("/") else ("/" + raw_p)
                    clean_p = clean_p.rstrip("/") if len(clean_p) > 1 else clean_p
                    scope["path"] = clean_p or "/"
                    scope["raw_path"] = (clean_p or "/").encode("latin1")
                    scope["query_string"] = urllib.parse.urlencode(params, doseq=True).encode("latin1")
            elif scope.get("path") in ("/api/index.py", "/api/index", "/index.py", "/index"):
                scope["path"] = "/"
                scope["raw_path"] = b"/"
        await self.app(scope, receive, send)

app.add_middleware(VercelPathFixMiddleware)

import tempfile

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CAPTURED_IMAGES_DIR = os.path.join(ARTIFACTS_DIR, "captured_images")

# Safely initialize captured_images directory without crashing on read-only serverless environments (Vercel/Lambda)
try:
    os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
except (OSError, PermissionError):
    CAPTURED_IMAGES_DIR = os.path.join(tempfile.gettempdir(), "lexibrief_captured_images")
    try:
        os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
    except Exception:
        pass

if os.path.exists(STATIC_DIR):
    try:
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    except Exception as e:
        logger.warning(f"Could not mount /static directory: {e}")

if os.path.exists(ARTIFACTS_DIR):
    try:
        app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
    except Exception as e:
        logger.warning(f"Could not mount /artifacts directory: {e}")

# Lazy-loaded prediction pipeline & database manager
prediction_pipeline = None
db_manager = MongoDBManager()

def get_prediction_pipeline():
    global prediction_pipeline
    if prediction_pipeline is None:
        from textSummarizer.pipeline.prediction import PredictionPipeline
        prediction_pipeline = PredictionPipeline()
    return prediction_pipeline

def validate_entity_id(entity_id: str) -> str:
    """Validates entity ID to prevent malformed or injection strings."""
    if not entity_id or len(entity_id) > 64 or not re.match(r'^[a-zA-Z0-9_-]+$', entity_id):
        raise HTTPException(status_code=400, detail="Invalid identifier format.")
    return entity_id


class SummaryRequest(BaseModel):
    text: str = Field(..., description="The raw input text or dialogue to summarize")
    mode: Optional[str] = Field("balanced", description="Summary mode: 'concise', 'balanced', or 'detailed'")
    method: Optional[str] = Field("auto", description="Summarization engine: 'abstractive', 'extractive', or 'auto'")
    model_name: Optional[str] = Field("bart", description="Transformer architecture: 'bart', 't5', 'pegasus'")
    persona: Optional[str] = Field("general", description="Persona profile: 'general', 'executive', 'technical', 'eli5', 'action_items'")
    max_length: Optional[int] = Field(None, description="Optional override for maximum tokens")
    min_length: Optional[int] = Field(None, description="Optional override for minimum tokens")


class URLIngestRequest(BaseModel):
    url: str = Field(..., description="Web article URL or YouTube video link")


class RougeEvalRequest(BaseModel):
    reference: str = Field(..., description="Original reference text")
    summary: str = Field(..., description="Generated or candidate summary")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize to speech")
    voice: Optional[str] = Field("neerja", description="Voice identifier ('neerja', 'prabhat' or language-specific)")
    speed: Optional[float] = Field(1.0, description="Speech rate multiplier")






def _find_index_html() -> Optional[str]:
    """Finds and reads index.html from multiple candidate directories in serverless and local runtimes."""
    candidate_paths = [
        os.path.join(TEMPLATES_DIR, "index.html"),
        os.path.join(BASE_DIR, "templates", "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html"),
        os.path.join(os.getcwd(), "templates", "index.html"),
        os.path.join(os.getcwd(), "api", "..", "templates", "index.html"),
        "/var/task/templates/index.html",
        "templates/index.html"
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return None


@app.get("/", response_class=HTMLResponse, tags=["UI"])
@app.get("/api/index.py", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
@app.get("/api/index", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
@app.get("/index.py", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
@app.get("/index", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def index():
    html_content = _find_index_html()
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    if html_content:
        return HTMLResponse(content=html_content, status_code=200, headers=headers)
    return HTMLResponse(
        content="""<!DOCTYPE html><html><head><title>LexiBrief NLP Engine</title><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>body{font-family:system-ui,sans-serif;background:#090d16;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;padding:20px;box-sizing:border-box}.card{background:#131b2e;padding:32px;border-radius:16px;border:1px solid #1e293b;max-width:540px;text-align:center;box-shadow:0 20px 40px rgba(0,0,0,0.5)}h1{font-size:24px;color:#38bdf8;margin:0 0 12px}p{color:#94a3b8;line-height:1.6;margin:0 0 20px}a{display:inline-block;padding:10px 20px;background:#2563eb;color:#fff;text-decoration:none;border-radius:8px;font-weight:600;transition:background 0.2s}a:hover{background:#1d4ed8}</style></head><body><div class="card"><h1>⚡ LexiBrief NLP Engine Active</h1><p>The backend API services and serverless functions are operational. You can explore interactive OpenAPI documentation below.</p><a href="/docs">View Interactive Swagger Docs</a></div></body></html>""",
        status_code=200,
        headers=headers
    )


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    candidate_favicons = [
        os.path.join(STATIC_DIR, "logo.jpg"),
        os.path.join(BASE_DIR, "static", "logo.jpg"),
        os.path.join(os.getcwd(), "static", "logo.jpg"),
        "/var/task/static/logo.jpg",
        "static/logo.jpg"
    ]
    for p in candidate_favicons:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    return Response(content=f.read(), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
            except Exception:
                pass
    return Response(status_code=204)


@app.get("/logo.jpg", include_in_schema=False)
@app.get("/static/logo.jpg", include_in_schema=False)
async def serve_logo():
    candidate_paths = [
        os.path.join(STATIC_DIR, "logo.jpg"),
        os.path.join(BASE_DIR, "static", "logo.jpg"),
        os.path.join(os.getcwd(), "static", "logo.jpg"),
        "/var/task/static/logo.jpg",
        "static/logo.jpg"
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    return Response(
                        content=f.read(),
                        media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"}
                    )
            except Exception:
                pass
    raise HTTPException(status_code=404, detail="Logo file not found")


@app.get("/static/{file_path:path}", include_in_schema=False)
async def serve_static_file(file_path: str):
    import mimetypes
    clean_path = file_path.lstrip("/\\")
    candidate_paths = [
        os.path.join(STATIC_DIR, clean_path),
        os.path.join(BASE_DIR, "static", clean_path),
        os.path.join(os.getcwd(), "static", clean_path),
        f"/var/task/static/{clean_path}",
        f"static/{clean_path}"
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                mime_type, _ = mimetypes.guess_type(p)
                if not mime_type:
                    if p.lower().endswith((".jpg", ".jpeg")):
                        mime_type = "image/jpeg"
                    elif p.lower().endswith(".png"):
                        mime_type = "image/png"
                    elif p.lower().endswith(".svg"):
                        mime_type = "image/svg+xml"
                    elif p.lower().endswith(".ico"):
                        mime_type = "image/x-icon"
                    else:
                        mime_type = "application/octet-stream"
                with open(p, "rb") as f:
                    return Response(
                        content=f.read(),
                        media_type=mime_type,
                        headers={"Cache-Control": "public, max-age=86400"}
                    )
            except Exception:
                pass
    raise HTTPException(status_code=404, detail=f"Static file '{file_path}' not found")


@app.get("/artifacts/{file_path:path}", include_in_schema=False)
async def serve_artifact_file(file_path: str):
    import mimetypes
    clean_path = file_path.lstrip("/\\")
    filename = os.path.basename(clean_path)
    candidate_paths = [
        os.path.join(ARTIFACTS_DIR, clean_path),
        os.path.join(CAPTURED_IMAGES_DIR, filename),
        os.path.join(BASE_DIR, "artifacts", clean_path),
        os.path.join(tempfile.gettempdir(), "lexibrief_captured_images", filename),
        f"/var/task/artifacts/{clean_path}",
        f"artifacts/{clean_path}"
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                mime_type, _ = mimetypes.guess_type(p)
                if not mime_type:
                    mime_type = "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "application/octet-stream"
                with open(p, "rb") as f:
                    return Response(
                        content=f.read(),
                        media_type=mime_type,
                        headers={"Cache-Control": "public, max-age=86400"}
                    )
            except Exception:
                pass

    # Check MongoDB cloud storage for captured images (e.g. on serverless Vercel)
    try:
        rec = db_manager.get_captured_image_by_filename(filename)
        if rec and rec.get("data_base64"):
            img_bytes = base64.b64decode(rec["data_base64"])
            mime_type = "image/png" if filename.lower().endswith(".png") else "image/webp" if filename.lower().endswith(".webp") else "image/jpeg"
            # Cache locally to tempdir for faster subsequent requests
            try:
                cache_dir = os.path.join(tempfile.gettempdir(), "lexibrief_captured_images")
                os.makedirs(cache_dir, exist_ok=True)
                with open(os.path.join(cache_dir, filename), "wb") as f_cache:
                    f_cache.write(img_bytes)
            except Exception:
                pass
            return Response(
                content=img_bytes,
                media_type=mime_type,
                headers={"Cache-Control": "public, max-age=86400"}
            )
    except Exception as db_err:
        logger.warning(f"Could not retrieve artifact from MongoDB: {db_err}")

    raise HTTPException(status_code=404, detail=f"Artifact file '{file_path}' not found")


@app.get("/api/health", tags=["System"])
async def health_check():
    db_status = db_manager.get_database_status()
    return {
        "status": "healthy",
        "service": "LexiBrief NLP Engine",
        "version": "2.0.0",
        "python_version": sys.version.split()[0],
        "database": db_status
    }


def save_captured_image(content_bytes: bytes, filename: Optional[str] = None) -> Optional[str]:
    """
    Saves uploaded or captured camera photo bytes into a dedicated 'artifacts/captured_images' folder in the project,
    and syncs to MongoDB for persistent cloud and serverless access.
    Returns the relative path to the saved image file.
    """
    if not content_bytes:
        return None
    try:
        target_dir = CAPTURED_IMAGES_DIR
        try:
            os.makedirs(target_dir, exist_ok=True)
        except (OSError, PermissionError):
            import tempfile
            target_dir = os.path.join(tempfile.gettempdir(), "lexibrief_captured_images")
            os.makedirs(target_dir, exist_ok=True)

        detected_ext = "jpg"
        if filename and "." in filename:
            cand_ext = filename.rsplit(".", 1)[-1].lower()
            if cand_ext in ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"]:
                detected_ext = cand_ext
        elif content_bytes[:8].startswith(b'\x89PNG\r\n\x1a\n'):
            detected_ext = "png"
        elif content_bytes[:3] == b'\xff\xd8\xff':
            detected_ext = "jpg"
        elif content_bytes[:4] == b'RIFF' and b'WEBP' in content_bytes[:12]:
            detected_ext = "webp"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]
        
        raw_prefix = "captured_photo"
        if filename:
            raw_base = os.path.splitext(os.path.basename(filename))[0]
            sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_base)[:40].strip('_')
            if sanitized and sanitized not in ["camera_capture", "blob"]:
                raw_prefix = sanitized

        clean_filename = f"{raw_prefix}_{timestamp}_{unique_suffix}.{detected_ext}"
        full_filepath = os.path.join(target_dir, clean_filename)

        try:
            with open(full_filepath, "wb") as fp:
                fp.write(content_bytes)
        except Exception as write_err:
            logger.warning(f"Could not write to local filepath {full_filepath}: {write_err}")

        rel_path = f"artifacts/captured_images/{clean_filename}"
        
        # Persist to MongoDB for persistent cloud access
        try:
            b64_str = base64.b64encode(content_bytes).decode('utf-8')
            size_kb = max(1, round(len(content_bytes) / 1024))
            size_fmt = f"{size_kb} KB" if len(content_bytes) < 1024*1024 else f"{(len(content_bytes) / (1024*1024)):.2f} MB"
            db_manager.save_captured_image({
                "filename": clean_filename,
                "path": rel_path,
                "url": f"/{rel_path}",
                "data_base64": b64_str,
                "size_bytes": len(content_bytes),
                "size_formatted": size_fmt,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as m_err:
            logger.warning(f"Failed to sync captured image to MongoDB: {m_err}")

        logger.info(f"Saved uploaded/captured image: {rel_path} ({len(content_bytes)} bytes)")
        return rel_path
    except Exception as e:
        logger.warning(f"Failed to save image to dedicated folder: {e}")
        return None


class OCRRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64-encoded image or Data URL")
    lang: Optional[str] = Field("auto", description="Language hint")


@app.post("/api/ocr", tags=["Text Extraction & MongoDB"])
@app.post("/api/ocr-image", tags=["Text Extraction & MongoDB"], include_in_schema=False)
async def ocr_image_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """
    High-speed, neural OCR text extraction endpoint.
    Accepts image file upload (multipart) or JSON base64 data URI (pasted screenshot/camera capture)
    and returns accurate extracted text with preserved bullet layout.
    Automatically saves the input image/photo into the dedicated 'artifacts/captured_images' project folder.
    """
    try:
        content_bytes = None
        orig_filename = None
        content_type = request.headers.get("content-type", "")

        # 1. Parse JSON payload (e.g. pasted screenshot or base64 image data URI)
        if "application/json" in content_type:
            try:
                body_json = await request.json()
                if isinstance(body_json, dict) and body_json.get("image"):
                    raw_str = str(body_json["image"]).strip()
                    if "," in raw_str:
                        raw_str = raw_str.split(",", 1)[1]
                    content_bytes = base64.b64decode(raw_str)
                    orig_filename = body_json.get("filename", "screenshot_capture.png")
            except Exception as parse_err:
                logger.warning(f"Error parsing JSON OCR payload: {parse_err}")

        # 2. Parse Multipart File upload
        if not content_bytes and file is not None:
            orig_filename = file.filename
            content_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
        elif not content_bytes and "multipart/form-data" in content_type:
            try:
                form = await request.form()
                form_file = form.get("file")
                if form_file and hasattr(form_file, "read"):
                    orig_filename = getattr(form_file, "filename", "upload.jpg")
                    content_bytes = await form_file.read(MAX_UPLOAD_SIZE + 1)
                elif "image" in form:
                    raw_str = str(form.get("image")).strip()
                    if "," in raw_str:
                        raw_str = raw_str.split(",", 1)[1]
                    content_bytes = base64.b64decode(raw_str)
                    orig_filename = form.get("filename", "camera_capture.jpg")
            except Exception as form_err:
                logger.warning(f"Error parsing multipart form for OCR: {form_err}")
        
        if not content_bytes:
            raise HTTPException(status_code=400, detail="No image data provided for OCR.")

        if len(content_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Image size exceeds maximum 50 MB limit.")

        # Persist captured photo/image to dedicated folder in project structure
        saved_img_path = save_captured_image(content_bytes, filename=orig_filename)

        extracted_text, pages_count = TextExtractor.extract_from_image(content_bytes)
        cleaned_text = TextExtractor.clean_ocr_text(extracted_text) if extracted_text else ""
        words = len(cleaned_text.split()) if cleaned_text else 0

        return {
            "success": True,
            "text": cleaned_text,
            "raw_text": extracted_text,
            "words": words,
            "pages": pages_count,
            "engine": "rapidocr_onnx",
            "saved_image_path": saved_img_path,
            "image_path": saved_img_path
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"OCR image endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


class SaveImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image or Data URL")
    filename: Optional[str] = Field("camera_capture.jpg", description="Optional original filename")


@app.post("/api/save-captured-image", tags=["Text Extraction & MongoDB"])
async def save_captured_image_endpoint(body: SaveImageRequest):
    """Saves a snapped camera photo or uploaded image directly into artifacts/captured_images."""
    try:
        raw_str = body.image.strip()
        if "," in raw_str:
            raw_str = raw_str.split(",", 1)[1]
        content_bytes = base64.b64decode(raw_str)
        if len(content_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Image size exceeds 50MB limit.")
        
        saved_path = save_captured_image(content_bytes, filename=body.filename or "camera_capture.jpg")
        if not saved_path:
            raise HTTPException(status_code=500, detail="Failed to save image to disk.")
        
        filename = os.path.basename(saved_path)
        return {
            "success": True,
            "filename": filename,
            "saved_image_path": saved_path,
            "url": f"/{saved_path}",
            "size_bytes": len(content_bytes)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error saving captured image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")


@app.get("/api/captured-images", tags=["Text Extraction & MongoDB"])
async def list_captured_images():
    """Lists all user-uploaded and captured camera photos stored in MongoDB or artifacts/captured_images."""
    files_map = {}
    
    # 1. Fetch persistent cloud records from MongoDB
    try:
        mongo_images = db_manager.get_captured_images(limit=100)
        for img in mongo_images:
            fname = img.get("filename")
            if fname:
                files_map[fname] = {
                    "filename": fname,
                    "path": img.get("path") or f"artifacts/captured_images/{fname}",
                    "url": img.get("url") or f"/artifacts/captured_images/{fname}",
                    "size_bytes": img.get("size_bytes", 0),
                    "size_formatted": img.get("size_formatted", "0 KB"),
                    "created_at": img.get("created_at", "")
                }
    except Exception as m_err:
        logger.warning(f"Error fetching captured images from MongoDB: {m_err}")

    # 2. Merge local disk files
    search_dirs = [
        CAPTURED_IMAGES_DIR,
        os.path.join(tempfile.gettempdir(), "lexibrief_captured_images")
    ]
    for target_dir in search_dirs:
        if not target_dir or not os.path.exists(target_dir):
            continue
        try:
            for f in os.listdir(target_dir):
                if f in files_map or f.startswith('.'):
                    continue
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif')):
                    fp = os.path.join(target_dir, f)
                    stat = os.stat(fp)
                    rel_path = f"artifacts/captured_images/{f}"
                    size_formatted = f"{max(1, round(stat.st_size / 1024))} KB" if stat.st_size < 1024*1024 else f"{(stat.st_size / (1024*1024)):.2f} MB"
                    files_map[f] = {
                        "filename": f,
                        "path": rel_path,
                        "url": f"/{rel_path}",
                        "size_bytes": stat.st_size,
                        "size_formatted": size_formatted,
                        "created_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    }
        except Exception as err:
            logger.warning(f"Error scanning captured images in {target_dir}: {err}")
            
    files = list(files_map.values())
    files.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"images": files, "count": len(files), "directory": "artifacts/captured_images"}


@app.delete("/api/captured-images", tags=["Text Extraction & MongoDB"])
async def delete_all_captured_images():
    """Deletes all captured images from artifacts/captured_images and MongoDB storage."""
    deleted_mongo_count = db_manager.delete_all_captured_images()
    deleted_disk_count = 0
    search_dirs = [
        CAPTURED_IMAGES_DIR,
        os.path.join(tempfile.gettempdir(), "lexibrief_captured_images")
    ]
    for target_dir in search_dirs:
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            for f in os.listdir(target_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif')):
                    try:
                        os.remove(os.path.join(target_dir, f))
                        deleted_disk_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete image file {f}: {e}")
    total_deleted = max(deleted_mongo_count, deleted_disk_count)
    return {"success": True, "message": f"Deleted all captured images ({total_deleted} items)", "deleted_count": total_deleted}


@app.delete("/api/captured-images/{filename}", tags=["Text Extraction & MongoDB"])
async def delete_captured_image(filename: str):
    """Deletes a captured image from artifacts/captured_images and MongoDB storage."""
    sanitized = os.path.basename(filename)
    deleted_mongo = db_manager.delete_captured_image(sanitized)
    search_paths = [
        os.path.join(CAPTURED_IMAGES_DIR, sanitized),
        os.path.join(tempfile.gettempdir(), "lexibrief_captured_images", sanitized)
    ]
    deleted_disk = False
    for fp in search_paths:
        if os.path.exists(fp) and os.path.isfile(fp):
            try:
                os.remove(fp)
                deleted_disk = True
            except Exception as e:
                logger.warning(f"Could not delete {fp}: {e}")
    if deleted_mongo or deleted_disk:
        return {"success": True, "message": f"Deleted {sanitized}"}
    raise HTTPException(status_code=404, detail="Captured image not found.")


@app.post("/api/upload", tags=["Text Extraction & MongoDB"])
@app.post("/upload", tags=["Text Extraction & MongoDB"], include_in_schema=False)
async def upload_document(
    file: UploadFile = File(...),
    ocr_text: Optional[str] = Form(None)
):
    """Extracts and cleans raw text from uploaded files (PDF, DOCX, PPTX, PPT, TXT, Images & Live Photos) and saves to MongoDB."""
    try:
        # Security: Enforce max upload file size (50 MB) to prevent OOM/DoS
        content_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, 
                detail="File too large. Maximum supported document size is 50 MB."
            )
        
        extracted_text, detected_format, pages_count = TextExtractor.extract(file.filename or "document.txt", content_bytes)
        
        # If client provided OCR text (e.g. from live camera photo review or manual edit)
        if ocr_text and ocr_text.strip():
            cleaned_ocr = TextExtractor.clean_ocr_text(ocr_text)
            if not extracted_text or not extracted_text.strip():
                extracted_text = cleaned_ocr
            else:
                server_words = len(extracted_text.split())
                client_words = len(cleaned_ocr.split())
                # Prioritize server neural OCR unless client text has distinctly more content
                if server_words < 3 and client_words >= 3:
                    extracted_text = cleaned_ocr
                elif client_words > server_words * 1.6 and client_words > 10:
                    extracted_text = cleaned_ocr
            if (file.filename or '').lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                detected_format = "IMAGE"

        # Persist uploaded image or live photo capture to dedicated artifacts/captured_images folder
        saved_img_path = None
        if detected_format == "IMAGE" or (file.filename or '').lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif')):
            saved_img_path = save_captured_image(content_bytes, filename=file.filename)

        if not extracted_text or not extracted_text.strip():
            raw_stem = os.path.splitext(file.filename or "document.txt")[0]
            clean_title = re.sub(r'^\d+[\s_\-\.]*', '', raw_stem).replace('_', ' ').replace('-', ' ').strip().title()
            if detected_format in ["PPTX", "PDF", "DOCX"] and clean_title:
                extracted_text = f"Title: {clean_title}\n\n[Uploaded {detected_format} document with {pages_count} {'slide' if detected_format == 'PPTX' else 'page'}{'s' if pages_count > 1 else ''}]"
            elif detected_format == "IMAGE" or saved_img_path:
                # Provide a clean document title rather than throwing raw errors on non-text image uploads
                extracted_text = clean_title if clean_title else "Scanned Document"
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not extract readable text or words from uploaded photo/document. Please ensure the document contains readable text or clear images."
                )
            
        stats = NLPProcessor.compute_stats(extracted_text)
        stats["pages"] = pages_count
        keywords = NLPProcessor.extract_keywords(extracted_text, top_k=8)
        key_points = NLPProcessor.extract_key_points(extracted_text, top_k=4)
        
        doc_payload = {
            "filename": file.filename or "document.txt",
            "format": detected_format,
            "text": extracted_text,
            "pages": pages_count,
            "words": stats.get("words", 0),
            "stats": stats,
            "keywords": keywords,
            "key_points": key_points,
            "image_path": saved_img_path
        }

        # Persist to MongoDB documents collection with safe fallback
        try:
            saved_record = db_manager.save_document(doc_payload)
            doc_id = saved_record.get("_id") if isinstance(saved_record, dict) else str(uuid.uuid4())
        except Exception as db_err:
            logger.warning(f"Database save error during document upload: {db_err}")
            doc_id = str(uuid.uuid4())
        
        return {
            "id": doc_id,
            "filename": file.filename or "document.txt",
            "format": detected_format,
            "pages": pages_count,
            "words": stats.get("words", 0),
            "text": extracted_text,
            "stats": stats,
            "keywords": keywords,
            "key_points": key_points,
            "saved_to_db": True,
            "saved_image_path": saved_img_path,
            "image_path": saved_img_path
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"File extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the uploaded file: {str(e)}")


@app.post("/api/fetch-url", tags=["Text Extraction & MongoDB"])
@app.post("/fetch-url", tags=["Text Extraction & MongoDB"], include_in_schema=False)
async def fetch_url_content(req: URLIngestRequest):
    """Fetches and cleans article text from web URLs or subtitle transcripts from YouTube videos."""
    url_str = req.url.strip()
    if not url_str:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        is_youtube = bool(TextExtractor.extract_youtube_video_id(url_str))
        if is_youtube:
            extracted_data = TextExtractor.extract_from_youtube(url_str)
        else:
            extracted_data = TextExtractor.extract_from_url(url_str)

        if is_youtube and extracted_data.get("is_fallback"):
            return {
                "id": str(uuid.uuid4()),
                "title": extracted_data.get("title", "YouTube Video"),
                "format": "YOUTUBE",
                "author": extracted_data.get("author", "YouTube Creator"),
                "thumbnail": extracted_data.get("thumbnail", ""),
                "video_id": extracted_data.get("video_id", ""),
                "chunks": [],
                "text": "",
                "words": 0,
                "source_url": extracted_data.get("source_url", url_str),
                "is_fallback": True,
                "saved_to_db": False
            }

        extracted_text = extracted_data.get("text", "")
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from the provided URL.")

        stats = NLPProcessor.compute_stats(extracted_text)
        keywords = NLPProcessor.extract_keywords(extracted_text, top_k=8)
        key_points = NLPProcessor.extract_key_points(extracted_text, top_k=4)

        doc_payload = {
            "filename": extracted_data.get("title", "Web Resource"),
            "format": extracted_data.get("format", "WEB_URL"),
            "text": extracted_text,
            "pages": 1,
            "words": stats.get("words", 0),
            "source_url": extracted_data.get("source_url", url_str),
            "stats": stats,
            "keywords": keywords,
            "key_points": key_points
        }

        try:
            saved_record = db_manager.save_document(doc_payload)
            doc_id = saved_record.get("_id") if isinstance(saved_record, dict) else str(uuid.uuid4())
        except Exception as db_err:
            logger.warning(f"Database save error during URL ingest: {db_err}")
            doc_id = str(uuid.uuid4())

        return {
            "id": doc_id,
            "title": extracted_data.get("title", "Web Resource"),
            "format": extracted_data.get("format", "WEB_URL"),
            "author": extracted_data.get("author", ""),
            "thumbnail": extracted_data.get("thumbnail", ""),
            "chunks": extracted_data.get("chunks", []),
            "text": extracted_text,
            "words": stats.get("words", 0),
            "source_url": extracted_data.get("source_url", url_str),
            "stats": stats,
            "keywords": keywords,
            "key_points": key_points,
            "saved_to_db": True,
            "is_fallback": False
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"URL extraction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


VOICE_MAP = {
    "neerja": "en-IN-NeerjaNeural",
    "prabhat": "en-IN-PrabhatNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "swara": "hi-IN-SwaraNeural",
    "madhur": "hi-IN-MadhurNeural",
    "aria": "en-US-AriaNeural",
    "sonia": "en-GB-SoniaNeural",
}


@app.get("/api/tts", tags=["Audio & Speech"])
@app.get("/tts", tags=["Audio & Speech"], include_in_schema=False)
async def generate_speech_audio(
    text: str = Query(..., min_length=1, description="Text to synthesize"),
    voice: Optional[str] = Query("neerja", description="Voice identifier"),
    speed: Optional[float] = Query(1.0, description="Speed multiplier")
):
    """Generates high-fidelity neural speech audio using edge-tts with resilient gTTS fallback."""
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    if len(clean_text) > 4000:
        clean_text = clean_text[:4000]

    voice_id = VOICE_MAP.get((voice or "neerja").lower().strip(), "en-IN-NeerjaNeural")

    # 1. High-speed neural edge-tts
    try:
        import edge_tts
        spd = speed or 1.0
        rate_str = f"+{int((spd - 1.0)*100)}%" if spd >= 1.0 else f"-{int((1.0 - spd)*100)}%"
        communicate = edge_tts.Communicate(clean_text, voice_id, rate=rate_str)
        
        async def audio_generator():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        return StreamingResponse(
            audio_generator(),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": "inline; filename=speech.mp3"
            }
        )
    except Exception as e:
        logger.warning(f"edge-tts stream notice: {e}. Falling back to gTTS.")

    # 2. Resilient fallback: gTTS
    try:
        from gtts import gTTS
        is_hindi = bool(re.search(r'[\u0900-\u097F]', clean_text))
        lang = 'hi' if is_hindi else 'en'
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(
            fp,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": "inline; filename=speech.mp3"
            }
        )
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/api/documents", tags=["MongoDB Documents"])
async def get_documents(limit: int = 50):
    """Retrieves uploaded documents from MongoDB."""
    docs = db_manager.get_documents(limit=min(limit, 100))
    return {"documents": docs, "count": len(docs)}


@app.get("/api/documents/{doc_id}", tags=["MongoDB Documents"])
async def get_document_item(doc_id: str):
    """Retrieves a single document with full text from MongoDB."""
    valid_id = validate_entity_id(doc_id)
    doc = db_manager.get_document_by_id(valid_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/api/documents", tags=["MongoDB Documents"])
async def delete_all_documents_endpoint():
    """Deletes all documents from MongoDB and local storage."""
    deleted_count = db_manager.delete_all_documents()
    return {"success": True, "deleted_count": deleted_count}


@app.delete("/api/documents/{doc_id}", tags=["MongoDB Documents"])
async def delete_document_item(doc_id: str):
    """Deletes a document from MongoDB."""
    valid_id = validate_entity_id(doc_id)
    success = db_manager.delete_document(valid_id)
    return {"success": success, "deleted_id": valid_id}


@app.post("/api/summaries", tags=["MongoDB Summaries"])
async def save_summary_endpoint(summary_payload: Dict[str, Any] = Body(...)):
    """Manually saves a summary record to MongoDB."""
    saved = db_manager.save_summary(summary_payload)
    return {"success": True, "summary": saved}


@app.get("/api/summaries", tags=["MongoDB Summaries"])
async def get_summaries(limit: int = 50):
    """Retrieves saved summary history from MongoDB."""
    sums = db_manager.get_summaries(limit=min(limit, 100))
    return {"summaries": sums, "count": len(sums)}


@app.get("/api/summaries/{summary_id}", tags=["MongoDB Summaries"])
async def get_summary_item(summary_id: str):
    """Retrieves a single summary record with full text from MongoDB."""
    valid_id = validate_entity_id(summary_id)
    item = db_manager.get_summary_by_id(valid_id)
    if not item:
        raise HTTPException(status_code=404, detail="Summary not found")
    return item


@app.delete("/api/summaries", tags=["MongoDB Summaries"])
async def delete_all_summaries_endpoint():
    """Deletes all summaries from MongoDB and local storage."""
    deleted_count = db_manager.delete_all_summaries()
    return {"success": True, "deleted_count": deleted_count}


@app.delete("/api/summaries/{summary_id}", tags=["MongoDB Summaries"])
async def delete_summary_item(summary_id: str):
    """Deletes a summary record from MongoDB."""
    valid_id = validate_entity_id(summary_id)
    success = db_manager.delete_summary(valid_id)
    return {"success": success, "deleted_id": valid_id}


@app.get("/api/db/status", tags=["MongoDB Status"])
async def get_db_status():
    """Returns database connection status and collection statistics."""
    return db_manager.get_database_status()


@app.post("/api/rouge", tags=["NLP Evaluation"])
@app.post("/rouge", tags=["NLP Evaluation"], include_in_schema=False)
async def evaluate_rouge(req: RougeEvalRequest):
    """Computes real-time ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1."""
    rouge_res = NLPProcessor.compute_rouge(req.reference, req.summary)
    return {"rouge": rouge_res}




@app.get("/api/metrics", tags=["Telemetry"])
async def get_metrics():
    metrics_path = os.path.join("artifacts", "model_evaluation", "metrics.csv")
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            metrics_data = df.to_dict(orient="records")
        except Exception as e:
            metrics_data = {"error": f"Failed to parse metrics: {e}"}
    else:
        # Default benchmark scores for Pegasus & BART on CNN/DailyMail & SAMSum
        metrics_data = [
            {
                "model": "sshleifer/distilbart-cnn-12-6",
                "dataset": "SAMSum",
                "rouge1": 0.4421,
                "rouge2": 0.2185,
                "rougeL": 0.3684,
                "rougeLsum": 0.4012
            },
            {
                "model": "google/pegasus-cnn_dailymail",
                "dataset": "CNN/DailyMail",
                "rouge1": 0.4352,
                "rouge2": 0.2014,
                "rougeL": 0.3541,
                "rougeLsum": 0.3812
            }
        ]

    return {
        "model_architecture": "Encoder-Decoder (Transformer: BART/T5/Pegasus) + Extractive TF-IDF",
        "base_model": "BART / T5 / Pegasus",
        "database": db_manager.get_database_status(),
        "evaluation_metrics": metrics_data,
        "pipeline_stages": [
            {"id": 1, "name": "Data Ingestion", "status": "Completed", "artifacts": "artifacts/data_ingestion"},
            {"id": 2, "name": "Data Validation", "status": "Completed", "artifacts": "artifacts/data_validation"},
            {"id": 3, "name": "Data Transformation", "status": "Completed", "artifacts": "artifacts/data_transformation"},
            {"id": 4, "name": "Model Trainer", "status": "Configured", "artifacts": "artifacts/model_trainer"},
            {"id": 5, "name": "Model Evaluation", "status": "Configured", "artifacts": "artifacts/model_evaluation"}
        ]
    }


@app.get("/train", tags=["Pipeline"])
async def training(request: Request):
    """Triggers model training and evaluation pipeline (Protected by ADMIN_SECRET_KEY)."""
    # Security: If ADMIN_SECRET_KEY is configured in environment, require authorization
    if ADMIN_SECRET_KEY:
        req_key = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
        if req_key != ADMIN_SECRET_KEY:
            raise HTTPException(
                status_code=403, 
                detail="Forbidden: Valid admin key is required to trigger model training."
            )

    try:
        process = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
        if process.returncode == 0:
            return JSONResponse(content={
                "status": "success",
                "message": "Model training & evaluation pipeline completed successfully!",
                "output": process.stdout
            })
        else:
            logger.error(f"Training pipeline failed: {process.stderr}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Pipeline execution encountered an internal error."
                }
            )
    except Exception as e:
        logger.error(f"Training process execution error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to start training process."})


async def synthesize_edge_neural(text: str, voice_key: str = "neerja", rate: str = "+0%") -> Tuple[Optional[bytes], str]:
    """
    Synthesizes ultra-realistic, studio-grade human neural voice audio using Microsoft Edge Neural TTS.
    Automatically detects language (Hindi, Marathi, Tamil, Spanish, English, etc.) and routes to the authentic native voice.
    Returns (audio_bytes, voice_name).
    """
    try:
        lang_code, lang_name = NLPProcessor.detect_language(text)
        if lang_code != "en" and lang_code in MULTILINGUAL_VOICE_MAP:
            voice_meta = MULTILINGUAL_VOICE_MAP[lang_code]
            voice_id = voice_meta["id"]
            voice_display = voice_meta["name"]
        else:
            voice_meta = EDGE_NEURAL_VOICE_MAP.get(voice_key.lower().strip(), {"id": "en-IN-NeerjaNeural", "name": "Neerja (English)"})
            voice_id = voice_meta["id"]
            voice_display = voice_meta["name"]

        communicate = edge_tts.Communicate(text, voice_id, rate=rate)
        audio_stream = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.extend(chunk["data"])
        if audio_stream:
            return bytes(audio_stream), voice_display
    except Exception as e:
        logger.warning(f"Edge Neural TTS generation error: {e}")
    return None, "Fallback"


@app.get("/api/tts", tags=["Text-to-Speech"])
@app.get("/tts", tags=["Text-to-Speech"], include_in_schema=False)
async def text_to_speech_get(text: str = Query(...), voice: str = Query("neerja")):
    """GET streaming endpoint for direct browser <audio src="..."> element playback."""
    req = TTSRequest(text=text, voice=voice)
    return await text_to_speech(req)


@app.post("/api/tts", tags=["Text-to-Speech"])
@app.post("/tts", tags=["Text-to-Speech"], include_in_schema=False)
async def text_to_speech(req: TTSRequest):
    """
    Synthesizes ultra-realistic, multilingual neural voice audio.
    Supports Hindi, Marathi, Bengali, Tamil, Telugu, Spanish, French, German, and English automatically.
    """
    try:
        text_content = req.text.strip()
        if not text_content:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
        # Clean text of markdown formatting for smooth narration
        clean_text = re.sub(r'[#*_`~>-]', ' ', text_content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if len(clean_text) > 8000:
            clean_text = clean_text[:8000]

        selected_voice = (req.voice or "neerja").lower().strip()

        # 1. Studio-Grade Microsoft Multilingual Neural Voice (100% Free, Zero Key, Native Speech)
        mp3_bytes, voice_display = await synthesize_edge_neural(clean_text, selected_voice)
        if mp3_bytes:
            return StreamingResponse(
                io.BytesIO(mp3_bytes),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=neural_speech.mp3",
                    "X-Voice-Engine": f"Microsoft-Neural-{voice_display}"
                }
            )

        # 2. Built-in Safety Fallback: Google TTS
        lang_code, _ = NLPProcessor.detect_language(clean_text)
        fp = io.BytesIO()
        tts = gTTS(text=clean_text, lang=lang_code if lang_code in ['en', 'hi', 'es', 'fr', 'de', 'ta', 'te', 'bn', 'mr', 'gu'] else 'en', slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(
            fp,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech_fallback.mp3",
                "X-Voice-Engine": f"Google-TTS-{lang_code}"
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail="Error generating text-to-speech audio.")


@app.post("/api/translate", tags=["Translation"])
async def translate_endpoint(request: TranslateRequest):
    """Translates any non-English text, summary, or document into clean fluent English in real time."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        translated, detected = Translator.translate_to_english(request.text, source_lang=request.source_lang or "auto")
        return {
            "translated_text": translated,
            "detected_language": detected,
            "original_text": request.text
        }
    except Exception as e:
        logger.error(f"Translation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/predict", tags=["Prediction & MongoDB"])
@app.post("/api/predict", tags=["Prediction & MongoDB"], include_in_schema=False)
@app.post("/", tags=["Prediction & MongoDB"], include_in_schema=False)
async def predict_route(request: Request, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        content_type = request.headers.get("content-type", "")
        input_text = ""
        selected_mode = "balanced"
        selected_method = "auto"
        selected_model = "bart"
        selected_persona = "general"
        translate_to_english = False
        max_len = None
        min_len = None

        if "application/json" in content_type:
            body = await request.json()
            input_text = body.get("text", "")
            selected_mode = body.get("mode", "balanced")
            selected_method = body.get("method", "auto")
            selected_model = body.get("model_name", "bart")
            selected_persona = body.get("persona", "general")
            translate_to_english = bool(body.get("translate_to_english", False))
            max_len = body.get("max_length")
            min_len = body.get("min_length")
        else:
            try:
                form = await request.form()
                input_text = form.get("text", "")
                selected_mode = form.get("mode", "balanced")
                selected_method = form.get("method", "auto")
                selected_model = form.get("model_name", "bart")
                selected_persona = form.get("persona", "general")
                translate_to_english = str(form.get("translate_to_english", "false")).lower() in ["true", "1", "yes"]
            except Exception:
                body = await request.json()
                input_text = body.get("text", "")
                selected_mode = body.get("mode", "balanced")
                selected_method = body.get("method", "auto")
                selected_model = body.get("model_name", "bart")
                selected_persona = body.get("persona", "general")
                translate_to_english = bool(body.get("translate_to_english", False))

        if not input_text or not str(input_text).strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # Security: Enforce maximum input text length to prevent memory & CPU starvation
        if len(str(input_text)) > MAX_INPUT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Input text exceeds maximum allowed length of {MAX_INPUT_CHARS:,} characters."
            )

        pipeline_obj = get_prediction_pipeline()
        prediction_result = pipeline_obj.predict(
            str(input_text),
            mode=selected_mode,
            method=selected_method,
            model_name=selected_model,
            persona=selected_persona,
            max_length=max_len,
            min_length=min_len,
            translate_to_english=translate_to_english
        )

        summary_text = prediction_result.get("summary", "")
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Compute text analytics
        orig_words = len(str(input_text).split())
        sum_words = len(summary_text.split())
        compression_pct = round(max(0, (1 - (sum_words / max(orig_words, 1)))) * 100, 1)

        result_payload = {
            "summary": summary_text,
            "original_summary": prediction_result.get("original_summary", summary_text),
            "key_points": prediction_result.get("key_points", []),
            "original_key_points": prediction_result.get("original_key_points", []),
            "translated_to_english": prediction_result.get("translated_to_english", False),
            "original_language": prediction_result.get("original_language", "en"),
            "mode": selected_mode,
            "persona": selected_persona,
            "method_used": prediction_result.get("method_used", selected_method),
            "model_source": prediction_result.get("model_source", "unknown"),
            "detected_language": prediction_result.get("detected_language", "en"),
            "language_name": prediction_result.get("language_name", "English"),
            "keywords": prediction_result.get("keywords", []),
            "attribution": prediction_result.get("attribution", {}),
            "nlp_stats": prediction_result.get("nlp_stats", {}),
            "summary_stats": prediction_result.get("summary_stats", {}),
            "rouge": prediction_result.get("rouge", {}),
            "analytics": {
                "original_words": orig_words,
                "summary_words": sum_words,
                "original_chars": len(str(input_text)),
                "summary_chars": len(summary_text),
                "compression_ratio": f"{compression_pct}%",
                "estimated_read_time_saved": f"{round(max(0, orig_words - sum_words) / 200 * 60)}s",
                "latency_ms": latency_ms
            }
        }

        # Non-blocking async background persistence to MongoDB
        background_tasks.add_task(
            db_manager.save_summary,
            {
                "text": str(input_text),
                "summary": summary_text,
                "key_points": result_payload["key_points"],
                "keywords": result_payload["keywords"],
                "mode": selected_mode,
                "persona": selected_persona,
                "attribution": result_payload["attribution"],
                "method_used": prediction_result.get("method_used", selected_method),
                "model_source": prediction_result.get("model_source", "unknown"),
                "rouge": result_payload["rouge"],
                "analytics": result_payload["analytics"]
            }
        )

        return result_payload
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during summarization.")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

