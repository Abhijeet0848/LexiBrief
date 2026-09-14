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
    """Verify that the React UI landing page loads with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "LexiBrief" in response.text
    assert "React" in response.text or "root" in response.text

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
    sample_raw = "  This is an NLP summary test! It extracts key concepts, computes ROUGE metrics, and extracts key takeaways.  "
    cleaned = TextExtractor.clean_text(sample_raw)
    assert cleaned.startswith("This is an NLP summary test!")
    
    tokens = NLPProcessor.tokenize_words(cleaned)
    assert "summary" in tokens
    
    sentences = NLPProcessor.split_sentences(cleaned)
    assert len(sentences) >= 1
    
    keywords = NLPProcessor.extract_keywords(cleaned, top_k=3)
    assert len(keywords) > 0

    key_points = NLPProcessor.extract_key_points(cleaned, top_k=2)
    assert len(key_points) > 0

    stats = NLPProcessor.compute_stats(cleaned)
    assert "words" in stats
    assert "readability_score" in stats

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
    """Verify that text prediction route handles input with extractive, abstractive, key points, and ROUGE scores."""
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
    assert "key_points" in data_ext
    assert "nlp_stats" in data_ext
    assert "rouge" in data_ext
    assert "rouge1" in data_ext["rouge"]

    # Auto method with model name
    resp_auto = client.post("/predict", json={"text": sample_text, "mode": "balanced", "method": "auto", "model_name": "bart"})
    assert resp_auto.status_code == 200
    assert "summary" in resp_auto.json()
    assert "analytics" in resp_auto.json()

def test_file_upload_endpoint():
    """Verify document upload endpoint with a mock text file."""
    file_content = b"LexiBrief provides high-performance abstractive and extractive text summarization with key points and ROUGE evaluation."
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
    assert "key_points" in data

def test_rouge_evaluation_endpoint():
    """Verify standalone ROUGE scoring endpoint."""
    reference = "Artificial intelligence accelerates document analysis and natural language processing."
    summary = "Artificial intelligence accelerates document analysis."
    response = client.post("/api/rouge", json={"reference": reference, "summary": summary})
    assert response.status_code == 200
    data = response.json()
    assert "rouge" in data
    assert data["rouge"]["rouge1"]["f1"] > 0.5
    assert data["rouge"]["rouge2"]["f1"] > 0.0
    assert data["rouge"]["rougeL"]["f1"] > 0.5

def test_mongodb_crud_endpoints():
    """Verify MongoDB document and summary CRUD endpoints."""
    # Summaries list
    res_sums = client.get("/api/summaries")
    assert res_sums.status_code == 200
    sums_data = res_sums.json()
    assert "summaries" in sums_data

    # Documents list
    res_docs = client.get("/api/documents")
    assert res_docs.status_code == 200
    docs_data = res_docs.json()
    assert "documents" in docs_data

    # DB status
    res_status = client.get("/api/db/status")
    assert res_status.status_code == 200
    assert "collections" in res_status.json()


def test_security_input_validation_and_id_sanitization():
    """Verify security controls: input length cap and invalid ID rejection."""
    # Test 1: Excessive input text (>150,000 characters)
    huge_text = "A" * 150_001
    res_huge = client.post("/predict", json={"text": huge_text})
    assert res_huge.status_code == 400
    assert "exceeds maximum allowed length" in res_huge.json()["detail"]

    # Test 2: Invalid entity identifier (injection string)
    res_bad_id = client.get("/api/documents/../../etc/passwd$#@")
    assert res_bad_id.status_code in [400, 404]

    # Test 3: Large file upload rejection (> 15MB)
    large_payload = b"X" * (15 * 1024 * 1024 + 10)
    res_large_file = client.post(
        "/api/upload",
        files={"file": ("large_bomb.txt", io.BytesIO(large_payload), "text/plain")}
    )
    assert res_large_file.status_code == 413
    assert "File too large" in res_large_file.json()["detail"]

