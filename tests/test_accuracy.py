import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in [BASE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.extractive_summarizer import ExtractiveSummarizer
from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.pipeline.prediction import PredictionPipeline


@pytest.fixture
def benchmark_test_cases():
    return [
        {
            "domain": "Technical / NLP",
            "source": (
                "Natural Language Processing has undergone a major revolution with the advent of deep transformer "
                "architectures like BERT, BART, and T5. These models utilize self-attention mechanisms to understand "
                "contextual representations and synthesize abstractive summaries. Traditional extractive methods "
                "relying on TF-IDF or text rank still provide rapid, compute-efficient baselines for large corpora."
            ),
            "reference": (
                "Deep transformers like BERT, BART, and T5 revolutionized NLP using self-attention for abstractive "
                "summarization, while extractive TF-IDF methods offer fast baselines for large corpora."
            )
        },
        {
            "domain": "News Article",
            "source": (
                "Renewable energy capacity expanded by over fifty percent globally last year, marking the fastest "
                "growth rate in two decades. Solar photovoltaic installations accounted for three quarters of the "
                "additions, driven by declining module manufacturing costs and supportive government policies across "
                "Europe and North America."
            ),
            "reference": (
                "Global renewable energy capacity expanded by over fifty percent last year, led predominantly by "
                "solar installations due to lower manufacturing costs and government incentives."
            )
        },
        {
            "domain": "Conversational Dialogue",
            "source": (
                "Alice: Hey Bob, did you review the project proposal for the client presentation tomorrow?\n"
                "Bob: Yes Alice, I checked the architecture slides and added performance benchmark metrics.\n"
                "Alice: Great! Let's schedule a dry run at 10 AM before the client joins at noon.\n"
                "Bob: Sounds good, I'll update the team calendar and share the revised deck."
            ),
            "reference": (
                "Bob reviewed the project proposal and added benchmark metrics. Alice and Bob agreed to hold a dry "
                "run tomorrow at 10 AM before the client presentation."
            )
        }
    ]


def test_rouge_metric_accuracy_and_bounds():
    """Verify that ROUGE scores adhere strictly to mathematical bounds [0, 1] and identity properties."""
    text = "Machine learning models optimize loss functions across training epochs."
    
    # 1. Identity test: Exact match should yield 100% (1.0) for Precision, Recall, and F1
    scores_identity = NLPProcessor.compute_rouge(reference=text, candidate=text)
    assert scores_identity["rouge1"]["f1"] == 1.0
    assert scores_identity["rouge1"]["precision"] == 1.0
    assert scores_identity["rouge1"]["recall"] == 1.0
    assert scores_identity["rouge2"]["f1"] == 1.0
    assert scores_identity["rougeL"]["f1"] == 1.0

    # 2. Disjoint test: Completely unrelated text should yield 0.0
    unrelated = "Zebra xylophone umbrella quantum jellyfish."
    scores_disjoint = NLPProcessor.compute_rouge(reference=text, candidate=unrelated)
    assert scores_disjoint["rouge1"]["f1"] == 0.0
    assert scores_disjoint["rouge2"]["f1"] == 0.0
    assert scores_disjoint["rougeL"]["f1"] == 0.0


def test_extractive_summary_accuracy(benchmark_test_cases):
    """Verify extractive summarization accuracy, compression ratio, and keyword coverage."""
    for case in benchmark_test_cases:
        summary = ExtractiveSummarizer.summarize(case["source"], ratio=0.5)
        
        # Summary must be non-empty and shorter than or equal to source
        assert len(summary) > 0
        assert len(summary.split()) <= len(case["source"].split())
        
        # ROUGE overlap against reference ground truth
        scores = NLPProcessor.compute_rouge(case["reference"], summary)
        assert scores["rouge1"]["f1"] > 0.20, f"Failed ROUGE-1 threshold for {case['domain']}"
        assert scores["rougeL"]["f1"] > 0.15, f"Failed ROUGE-L threshold for {case['domain']}"


def test_nlp_readability_and_keyword_accuracy():
    """Verify NLP statistical metrics, sentence splitting, and readability scoring accuracy."""
    doc = (
        "Artificial intelligence platforms automate document summarization. "
        "LexiBrief leverages state-of-the-art transformer pipelines. "
        "Users can inspect real-time precision and recall metrics."
    )
    stats = NLPProcessor.compute_stats(doc)
    assert stats["words"] > 0
    assert stats["sentences"] == 3
    assert stats["characters"] > 100
    assert 0 <= stats["readability_score"] <= 100
    
    keywords = NLPProcessor.extract_keywords(doc, top_k=3)
    assert len(keywords) == 3
    extracted_kw_names = [k["keyword"].lower() for k in keywords]
    assert any(any(sub in k for sub in ["precision", "metrics", "users", "recall", "intelligence", "artificial", "summarization", "document", "lexibrief", "transformer"]) for k in extracted_kw_names)


def test_end_to_end_prediction_accuracy(benchmark_test_cases):
    """Verify end-to-end summarizer accuracy with mode adjustments and ROUGE output."""
    pipeline = PredictionPipeline()
    
    for case in benchmark_test_cases:
        result = pipeline.predict(
            text=case["source"],
            mode="balanced",
            method="extractive"
        )
        
        assert "summary" in result
        assert len(result["summary"]) > 0
        assert "rouge" in result
        assert "keywords" in result
        assert "key_points" in result
        assert len(result["key_points"]) > 0
        assert result["nlp_stats"]["words"] >= result["summary_stats"]["words"]


def test_multilingual_indic_keyword_extraction():
    """Verify that multilingual Indic/Hindi text extracts intact, authentic words instead of broken character fragments."""
    hindi_doc = (
        "दर्शनशास्त्र ज्ञान, वास्तविकता और अस्तित्व की प्रकृति से संबंधित मौलिक प्रश्नों का अध्ययन है। "
        "जब हम ज्ञान की सीमाओं पर विचार करते हैं, तो हमें मस्तिष्क और चेतना के कई अनसुलझे प्रश्न मिलते हैं। "
        "इस सिद्धांत में कई महत्वपूर्ण प्रश्न शामिल हैं।"
    )
    keywords = NLPProcessor.extract_keywords(hindi_doc, top_k=6)
    kw_words = [k["keyword"] for k in keywords]
    
    # Must extract intact Devanagari words like 'प्रश्न', 'ज्ञान', 'मस्तिष्क', 'सिद्धांत'
    assert any("प्रश्न" in w for w in kw_words)
    assert any("ज्ञान" in w for w in kw_words)
    assert any("मस्तिष्क" in w or "सिद्धांत" in w or "सीमाओं" in w or "वास्तविकता" in w for w in kw_words)
    
    # Ensure no single-character fragment or broken matras like 'रक', 'बलकत', 'यर'
    for kw in kw_words:
        assert len(kw) >= 2
        # No dangling or broken combining virama at word start
        assert not kw.startswith('\u094d')


def test_named_entities_and_facts_extraction():
    """Verify extraction of named entities (NER) and verified quantitative facts."""
    sample_text = (
        "Google Cloud announced a $10 billion investment in AI infrastructure in September 2024. "
        "Dr. Andrew Ng stated that enterprise adoption of foundation models grew by 45% this quarter. "
        "The project reached 100,000 active developers across Europe and India."
    )
    entities = NLPProcessor.extract_named_entities(sample_text)
    facts = NLPProcessor.extract_important_facts(sample_text)

    # Entities check
    assert len(entities) >= 3
    ent_texts = [e["text"] for e in entities]
    assert any("$10 billion" in t or "10 billion" in t for t in ent_texts)
    assert any("45%" in t for t in ent_texts)
    assert any("Google" in t or "Dr. Andrew Ng" in t or "Europe" in t for t in ent_texts)

    # Facts check
    assert len(facts) >= 2
    assert any("45%" in f or "$10 billion" in f or "investment" in f for f in facts)


def test_section_detection():
    """Verify section-aware parsing across markdown headers and dividers."""
    structured_doc = (
        "# Introduction\n"
        "LexiBrief is an AI document summarization platform.\n\n"
        "## Performance & Latency\n"
        "Inference latency is under 50ms per batch.\n\n"
        "## Conclusion\n"
        "The system scales seamlessly for enterprise workloads."
    )
    sections = NLPProcessor.detect_sections(structured_doc)
    assert len(sections) == 3
    titles = [s["title"] for s in sections]
    assert "Introduction" in titles
    assert "Performance & Latency" in titles
    assert "Conclusion" in titles


def test_ocr_multi_column_and_caption_layout_reconstruction():
    """Verify that multi-column and side-caption images do not interleave captions into body paragraphs."""
    import numpy as np
    
    # Realistic layout coordinates mimicking an image with a left caption and right body text
    mock_ocr_boxes = [
        # Right column body lines (top)
        ([[320, 40], [950, 40], [950, 65], [320, 65]], "The Noemvriana was an armed confrontation in Athens on", 0.99),
        ([[320, 75], [950, 75], [950, 100], [320, 100]], "1 December 1916 between the Kingdom of Greece and the", 0.99),
        ([[320, 110], [950, 110], [950, 135], [320, 135]], "Allied Powers and their supporters. The crisis arose from", 0.99),
        ([[320, 145], [950, 145], [950, 170], [320, 170]], "disputes over Greek neutrality during the First World War. The", 0.99),
        ([[320, 180], [950, 180], [950, 205], [320, 205]], "Allies feared a secret alliance between King Constantine I and", 0.99),
        ([[320, 215], [950, 215], [950, 240], [320, 240]], "the Central Powers that could endanger their army", 0.99),
        ([[320, 250], [950, 250], [950, 275], [320, 275]], "bivouacking in Thessaloniki. The establishment of Eleftherios", 0.99),
        ([[320, 285], [950, 285], [950, 310], [320, 310]], "Venizelos's Allied-backed provisional government in", 0.99),
        
        # Left column caption (under image on the left)
        ([[10, 180], [220, 180], [220, 205], [10, 205]], "Naval bombardment of", 0.99),
        ([[10, 215], [180, 215], [180, 240], [10, 240]], "Athens during the", 0.99),
        ([[10, 250], [130, 250], [130, 275], [10, 275]], "Noemvriana", 0.99),
        
        # Full-width bottom lines
        ([[10, 320], [950, 320], [950, 345], [10, 345]], "Thessaloniki to create an army in assistance to the Allies divided Greece. Failed", 0.99),
        ([[10, 355], [950, 355], [950, 380], [10, 380]], "negotiations prompted the Allies to land in Athens to compel the surrender of war materiel.", 0.99),
    ]
    
    extracted = TextExtractor._reconstruct_ocr_boxes(mock_ocr_boxes)
    cleaned = TextExtractor.clean_ocr_text(extracted)
    
    # 1. Main body sentences must be coherent and not spliced with caption words
    assert "Naval bombardment of Allies feared" not in cleaned
    assert "Athens during the the Central Powers" not in cleaned
    assert "Noemvriana bivouacking in Thessaloniki" not in cleaned
    
    # 2. Main continuous article flow must be preserved intact
    assert "Allies feared a secret alliance between King Constantine I and" in cleaned
    assert "the Central Powers that could endanger their army" in cleaned
    
    # 3. Caption must be extracted cleanly as an isolated section
    assert "Naval bombardment of" in cleaned
    assert "Athens during the" in cleaned
    assert "Noemvriana" in cleaned


