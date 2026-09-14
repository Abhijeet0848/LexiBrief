import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.extractive_summarizer import ExtractiveSummarizer
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
    assert any(any(sub in k for sub in ["intelligence", "artificial", "summarization", "document", "lexibrief", "transformer"]) for k in extracted_kw_names)


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
