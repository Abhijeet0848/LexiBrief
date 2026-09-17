import sys
import os
import io
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in [BASE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

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


def test_tts_endpoint():
    """Verify that neural text-to-speech audio streaming endpoint returns valid audio."""
    response = client.get("/api/tts?text=LexiBrief+audio+test&voice=neerja")
    assert response.status_code == 200
    assert "audio" in response.headers.get("content-type", "")

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
    """Verify MongoDB document and summary CRUD endpoints and delete-all operations."""
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

    # Delete all documents
    res_del_docs = client.delete("/api/documents")
    assert res_del_docs.status_code == 200
    assert res_del_docs.json()["success"] is True

    # Delete all summaries
    res_del_sums = client.delete("/api/summaries")
    assert res_del_sums.status_code == 200
    assert res_del_sums.json()["success"] is True

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

    # Test 3: Large file upload rejection (> 50MB)
    large_payload = b"X" * (50 * 1024 * 1024 + 10)
    res_large_file = client.post(
        "/api/upload",
        files={"file": ("large_bomb.txt", io.BytesIO(large_payload), "text/plain")}
    )
    assert res_large_file.status_code == 413
    assert "File too large" in res_large_file.json()["detail"]


def test_multilingual_summarization_and_tts():
    """Verify end-to-end language detection, extractive summarization, and TTS in Hindi."""
    hindi_input = (
        "कर्मचारी चयन आयोग भारत सरकार के विभिन्न मंत्रालयों में भर्ती आयोजित करता है। "
        "यह परीक्षा विभिन्न सरकारी विभागों में योग्य उम्मीदवारों के चयन के लिए अत्यंत महत्वपूर्ण है। "
        "सभी उम्मीदवारों को निर्धारित समय सीमा के भीतर अपना आवेदन जमा करना आवश्यक है।"
    )
    res_pred = client.post("/predict", json={"text": hindi_input, "mode": "balanced", "method": "auto"})
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert data["detected_language"] == "hi"
    assert data["language_name"] == "Hindi"
    assert len(data["summary"]) > 0

    # TTS audio generation
    res_tts = client.post("/api/tts", json={"text": data["summary"]})
    assert res_tts.status_code == 200
    assert len(res_tts.content) > 500


def test_sentence_attribution():
    """Verify sentence-level source attribution mapping and confidence scoring."""
    source_text = (
        "Quarterly financial revenue surged by 34% to $12.4 billion this year. "
        "Cloud infrastructure adoption contributed significantly to our overall operating margins. "
        "The board of directors approved a new dividend payout schedule for all shareholders."
    )
    res = client.post("/predict", json={"text": source_text, "mode": "balanced", "method": "extractive"})
    assert res.status_code == 200
    data = res.json()
    assert "attribution" in data
    attr = data["attribution"]
    assert "attribution_map" in attr
    assert "source_sentences" in attr
    assert len(attr["attribution_map"]) > 0
    first_map = attr["attribution_map"][0]
    assert "summary_sentence" in first_map
    assert "source_sentence" in first_map
    assert "confidence" in first_map
    assert first_map["confidence"] > 0


def test_persona_summarization():
    """Verify persona-based summarization (Executive, Technical, Action Items)."""
    meeting_text = (
        "Alex: We need to increase Q3 SaaS revenue by 20% to reach our $5M ARR target. "
        "David: The Kubernetes cluster deployment latency has dropped to 12ms after introducing Redis caching. "
        "Sarah: I will prepare the final API migration documentation and deploy the Docker containers by Thursday."
    )

    # Executive Persona
    res_exec = client.post("/predict", json={"text": meeting_text, "persona": "executive", "method": "extractive"})
    assert res_exec.status_code == 200
    data_exec = res_exec.json()
    assert data_exec["persona"] == "executive"
    assert len(data_exec["summary"]) > 0

    # Action Items Persona
    res_actions = client.post("/predict", json={"text": meeting_text, "persona": "action_items", "method": "extractive"})
    assert res_actions.status_code == 200
    data_actions = res_actions.json()
    assert data_actions["persona"] == "action_items"
    assert "•" in data_actions["summary"] or len(data_actions["summary"]) > 0


def test_url_and_youtube_extraction():
    """Verify URL and YouTube parsing utilities."""
    # YouTube ID parsing
    yt_id = TextExtractor.extract_youtube_video_id("https://www.youtube.com/watch?v=kCc8FmEb1nY")
    assert yt_id == "kCc8FmEb1nY"
    yt_short_id = TextExtractor.extract_youtube_video_id("https://youtu.be/kCc8FmEb1nY")
    assert yt_short_id == "kCc8FmEb1nY"

    # Ingest URL endpoint validation
    res_empty = client.post("/api/fetch-url", json={"url": ""})
    assert res_empty.status_code == 400


def test_translation_endpoint():
    """Verify neural translation endpoint."""
    hindi_text = "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग आधुनिक दुनिया को बदल रहे हैं।"
    res = client.post("/api/translate", json={"text": hindi_text, "source_lang": "auto", "target_lang": "en"})
    assert res.status_code == 200
    data = res.json()
    assert "translated_text" in data
    assert len(data["translated_text"]) > 0
    assert any(w in data["translated_text"].lower() for w in ["artificial", "intelligence", "machine", "learning", "modern", "world"])


def test_multilingual_summarize_with_translation():
    """Verify predict endpoint with translate_to_english toggle for non-English passage."""
    hindi_passage = (
        "भारत एक महान और विविधतापूर्ण देश है। यहाँ कई भाषाएँ और संस्कृतियाँ एक साथ फलती-फूलती हैं। "
        "आधुनिक तकनीकी विकास ने देश के हर नागरिक के जीवन को सुगम बना दिया है। "
        "डिजिटल शिक्षा और इंटरनेट ने युवाओं को नए अवसर प्रदान किए हैं।"
    )
    res = client.post("/predict", json={
        "text": hindi_passage,
        "mode": "balanced",
        "method": "auto",
        "translate_to_english": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("translated_to_english") is True
    assert data.get("original_language") == "hi"


def test_upload_pptx_endpoint():
    """Verify document upload endpoint with PowerPoint PPTX file."""
    from tests.test_pptx_extractor import create_sample_pptx
    pptx_bytes = create_sample_pptx()
    response = client.post(
        "/api/upload",
        files={"file": ("project_roadmap.pptx", pptx_bytes, "application/vnd.openxmlformats-officedocument.presentationml.presentation")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "PPTX"
    assert data["pages"] == 3
    assert data["words"] > 0
    assert "LexiBrief NLP Architecture" in data["text"]


def test_multilingual_ocr_cleansing():
    """Verify that OCR cleansing preserves Hindi, Indic, and non-Latin scripts and Indic punctuation."""
    hindi_raw_ocr = "तोता हमारे देश का एक लोकप्रिय पक्षी है। |~^ तोता मुझे बहुत अच्छा लगता है।\nतोते की बोली मीठी होती है।"
    cleaned = TextExtractor.clean_ocr_text(hindi_raw_ocr)
    assert "तोता हमारे देश का एक लोकप्रिय पक्षी है।" in cleaned
    assert "तोते की बोली मीठी होती है।" in cleaned
    assert "|" not in cleaned
    assert "~" not in cleaned

