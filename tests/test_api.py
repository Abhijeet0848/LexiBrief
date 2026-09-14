import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app import app

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

def test_predict_endpoint():
    """Verify that text prediction route handles input and returns summary analytics."""
    sample_text = (
        "Artificial intelligence research laboratories have unveiled a new generation "
        "of transformer architectures optimized for abstractive document summarization. "
        "These models drastically reduce energy consumption while maintaining high semantic accuracy."
    )
    response = client.post("/predict", json={"text": sample_text, "mode": "balanced"})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert "analytics" in data
    assert "compression_ratio" in data["analytics"]
