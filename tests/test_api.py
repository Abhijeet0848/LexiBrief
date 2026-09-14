import sys
import os
import io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app import app
from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.extractive_summarizer import ExtractiveSummarizer

client = TestClient(app)

def test_index_route():
    """Verify that the UI landing page loads with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "LexiBrief" in response.text

def test_health_check_endpoint():
    """Verify health check endpoint returns healthy status and service info."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "LexiBrief NLP Engine"

def test_presets_endpoint():
    """Verify that sample presets endpoint returns valid test cases."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 4

def test_metrics_endpoint():
    """Verify that pipeline architecture and ROUGE metrics endpoint returns properly."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_stages" in data
    assert "evaluation_metrics" in data

def test_text_extractor_and_nlp_processor():
    """Verify text extractor and NLP preprocessing components."""
    sample_raw = "  This is an AI summary test!   It extracts key concepts and tokenizes words properly.  "
    cleaned = TextExtractor.clean_text(sample_raw)
    assert cleaned.startswith("This is an AI summary test!")
    
    tokens = NLPProcessor.tokenize_words(cleaned)
    assert "summary" in tokens
    
    sentences = NLPProcessor.split_sentences(cleaned)
    assert len(sentences) >= 2
    
    keywords = NLPProcessor.extract_keywords(cleaned, top_k=3)
    assert len(keywords) > 0

def test_extractive_summarizer():
    """Verify TF-IDF extractive summarization."""
    text = (
        "Machine learning models require robust preprocessing pipelines. "
        "Data ingestion loads the corpus from various raw formats. "
        "Data validation verifies schema and column consistency. "
        "Feature transformation prepares tokenized tensors for neural sequence models. "
        "Evaluation computes standard benchmark metrics like ROUGE and BLEU."
    )
    summary = ExtractiveSummarizer.summarize(text, ratio=0.4)
    assert len(summary) > 0
    assert len(summary) < len(text)

def test_predict_endpoint_modes_and_methods():
    """Verify that text prediction route handles input with extractive and abstractive methods."""
    sample_text = (
        "Artificial intelligence research laboratories have unveiled a new generation "
        "of transformer architectures optimized for abstractive document summarization. "
        "These models drastically reduce energy consumption while maintaining high semantic accuracy. "
        "Enterprise workflows can now process long documentation in seconds."
    )
    # Extractive method
    resp_ext = client.post("/predict", json={"text": sample_text, "mode": "concise", "method": "extractive"})
    assert resp_ext.status_code == 200
    data_ext = resp_ext.json()
    assert "summary" in data_ext
    assert "keywords" in data_ext
    assert "nlp_stats" in data_ext

    # Auto method
    resp_auto = client.post("/predict", json={"text": sample_text, "mode": "balanced", "method": "auto"})
    assert resp_auto.status_code == 200
    assert "summary" in resp_auto.json()

def test_file_upload_endpoint():
    """Verify document upload endpoint with a mock text file."""
    file_content = b"LexiBrief provides high-performance abstractive and extractive text summarization."
    response = client.post(
        "/api/upload",
        files={"file": ("sample_report.txt", io.BytesIO(file_content), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample_report.txt"
    assert data["format"] == "TXT"
    assert "LexiBrief" in data["text"]
    assert "stats" in data
    assert "keywords" in data
