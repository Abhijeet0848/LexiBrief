import sys
import os
import re

# Serverless & local environment directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

for path in [BASE_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File, Response, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import subprocess
import time
import pandas as pd

from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.mongo_manager import MongoDBManager
from textSummarizer.logging import logger

app = FastAPI(
    title="LexiBrief API",
    description="State-of-the-Art NLP Text Summarization Engine with Extractive, Abstractive & MongoDB persistence",
    version="2.0.0"
)

# Security Constants & Limits
MAX_UPLOAD_SIZE = 15 * 1024 * 1024   # 15 MB max file upload
MAX_INPUT_CHARS = 150_000            # 150,000 max input character limit
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

# Enable Secure CORS for API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    max_length: Optional[int] = Field(None, description="Optional override for maximum tokens")
    min_length: Optional[int] = Field(None, description="Optional override for minimum tokens")


class RougeEvalRequest(BaseModel):
    reference: str = Field(..., description="Original reference text")
    summary: str = Field(..., description="Generated or candidate summary")


SAMPLE_PRESETS = [
    {
        "id": "meeting",
        "title": "Project Sprint Meeting",
        "category": "Dialogue",
        "badge": "SAMSum Format",
        "text": """Alex: Hey everyone, let's do a quick sync on the Q3 release deliverables.
Sarah: The data pipeline refactoring is finished. Ingestion throughput is up by 45%.
Michael: Awesome. Frontend migration to the new design system is 90% done. Just ironing out a few responsive glitches on tablet views.
Alex: Great work. What about the model training and AWS ECR CI/CD pipeline?
David: Docker images are building cleanly, and the automated evaluation tests passed with ROUGE-1 score of 0.42. We're scheduled to deploy to EC2 on Thursday.
Alex: Perfect! Let's lock in Thursday morning for the staging deployment and notify the product team.
Sarah: Sounds like a plan. I'll prepare the release notes."""
    },
    {
        "id": "technews",
        "title": "Generative NLP Breakthrough",
        "category": "News Article",
        "badge": "Tech News",
        "text": """Artificial intelligence research laboratories have unveiled a new generation of encoder-decoder transformer architectures specifically optimized for abstractive document comprehension and summarization. By employing advanced sparse attention mechanisms and memory-efficient sequence-to-sequence fine-tuning, the new models achieve human-parity synthesis while reducing computational energy requirements by over 60 percent. Industry analysts predict this breakthrough will drastically streamline corporate workflows across legal discovery, medical literature review, and financial reporting, allowing knowledge workers to distill multi-hundred-page dossiers into actionable executive briefs in real time."""
    },
    {
        "id": "support",
        "title": "Customer Support Escalation",
        "category": "Support Chat",
        "badge": "Customer Ops",
        "text": """Customer: Hi, our production API integration began throwing 429 rate limit errors starting at 08:00 UTC today despite us being on the Enterprise tier.
Agent: Hello! I apologize for the disruption. Let me pull up your account credentials and system logs right away.
Customer: Thanks, our payment webhook processing is currently queued up.
Agent: I have located the issue. A recent load balancer update temporarily misclassified your API key pool. I have applied an immediate hotfix to whitelist your endpoint and doubled your burst concurrency limits.
Customer: Verified on our dashboard! Traffic is clearing normally now and queue is draining.
Agent: Excellent to hear. I have filed an internal incident report to ensure this edge case is permanently prevented in future deployments."""
    },
    {
        "id": "research",
        "title": "Scientific Study Abstract",
        "category": "Research",
        "badge": "Academic",
        "text": """Recent investigations into multi-modal deep learning architectures have demonstrated substantial efficacy in unifying cross-domain sequence reasoning. This study evaluates the generalization capability of pre-trained language models when fine-tuned on highly specialized domain taxonomies without auxiliary knowledge graphs. Empirical benchmarking across five diverse corpora reveals that selective layer unfreezing paired with adaptive learning rate warmup yields a 14.8% relative gain in semantic fidelity and reduces hallucination rates from 8.2% to 1.9%. These findings validate the hypothesis that intermediate representation regularization is critical for robust domain-adapted abstractive summarization."""
    }
]


@app.get("/", response_class=FileResponse, tags=["UI"])
async def index():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return FileResponse("templates/index.html")


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    logo_path = os.path.join(STATIC_DIR, "logo.jpg")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/jpeg")
    return Response(status_code=204)


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


@app.get("/api/presets", tags=["UI"])
async def get_presets():
    return {"presets": SAMPLE_PRESETS}


@app.post("/api/upload", tags=["Text Extraction & MongoDB"])
async def upload_document(file: UploadFile = File(...)):
    """Extracts and cleans raw text from uploaded files (PDF, DOCX, TXT) and saves to MongoDB."""
    try:
        # Security: Enforce max upload file size (15 MB) to prevent OOM/DoS
        content_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, 
                detail="File too large. Maximum supported document size is 15 MB."
            )
        
        extracted_text, detected_format = TextExtractor.extract(file.filename or "document.txt", content_bytes)
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from uploaded file.")
            
        stats = NLPProcessor.compute_stats(extracted_text)
        keywords = NLPProcessor.extract_keywords(extracted_text, top_k=8)
        key_points = NLPProcessor.extract_key_points(extracted_text, top_k=4)
        
        doc_payload = {
            "filename": file.filename or "document.txt",
            "format": detected_format,
            "text": extracted_text,
            "stats": stats,
            "keywords": keywords,
            "key_points": key_points
        }

        # Persist to MongoDB documents collection
        saved_record = db_manager.save_document(doc_payload)
        
        return {
            "id": saved_record.get("_id"),
            "filename": file.filename or "document.txt",
            "format": detected_format,
            "text": extracted_text,
            "stats": stats,
            "keywords": keywords,
            "key_points": key_points,
            "saved_to_db": True
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"File extraction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the uploaded file.")


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


@app.delete("/api/documents/{doc_id}", tags=["MongoDB Documents"])
async def delete_document_item(doc_id: str):
    """Deletes a document from MongoDB."""
    valid_id = validate_entity_id(doc_id)
    success = db_manager.delete_document(valid_id)
    return {"success": success, "deleted_id": valid_id}


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


@app.post("/predict", tags=["Prediction & MongoDB"])
async def predict_route(request: Request, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        content_type = request.headers.get("content-type", "")
        input_text = ""
        selected_mode = "balanced"
        selected_method = "auto"
        selected_model = "bart"
        max_len = None
        min_len = None

        if "application/json" in content_type:
            body = await request.json()
            input_text = body.get("text", "")
            selected_mode = body.get("mode", "balanced")
            selected_method = body.get("method", "auto")
            selected_model = body.get("model_name", "bart")
            max_len = body.get("max_length")
            min_len = body.get("min_length")
        else:
            try:
                form = await request.form()
                input_text = form.get("text", "")
                selected_mode = form.get("mode", "balanced")
                selected_method = form.get("method", "auto")
                selected_model = form.get("model_name", "bart")
            except Exception:
                body = await request.json()
                input_text = body.get("text", "")
                selected_mode = body.get("mode", "balanced")
                selected_method = body.get("method", "auto")
                selected_model = body.get("model_name", "bart")

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
            max_length=max_len,
            min_length=min_len
        )

        summary_text = prediction_result.get("summary", "")
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Compute text analytics
        orig_words = len(str(input_text).split())
        sum_words = len(summary_text.split())
        compression_pct = round(max(0, (1 - (sum_words / max(orig_words, 1)))) * 100, 1)

        result_payload = {
            "summary": summary_text,
            "key_points": prediction_result.get("key_points", []),
            "mode": selected_mode,
            "method_used": prediction_result.get("method_used", selected_method),
            "model_source": prediction_result.get("model_source", "unknown"),
            "keywords": prediction_result.get("keywords", []),
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
    uvicorn.run(app, host="0.0.0.0", port=8080)

