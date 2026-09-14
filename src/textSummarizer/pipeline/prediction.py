from typing import Optional, Dict, Any, List
from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.extractive_summarizer import ExtractiveSummarizer
from textSummarizer.components.abstractive_summarizer import AbstractiveSummarizer
from textSummarizer.logging import logger

class PredictionPipeline:
    """Unified NLP Summarization Pipeline coordinating Extractive and Abstractive Transformer engines."""

    def __init__(self):
        self.abstractive_engine = AbstractiveSummarizer()
        self.extractive_engine = ExtractiveSummarizer()

    def predict(
        self,
        text: str,
        mode: str = "balanced",
        method: str = "auto",
        model_name: str = "bart",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end NLP summarization, keyword extraction, key-points extraction,
        and ROUGE scoring.
        
        Args:
            text: Raw input text.
            mode: 'concise', 'balanced', or 'detailed'.
            method: 'abstractive', 'extractive', or 'auto'.
            model_name: 'bart', 't5', 'pegasus', etc.
            max_length: Optional token limit.
            min_length: Optional min token limit.
        """
        cleaned_text = TextExtractor.clean_text(text)
        if not cleaned_text:
            return {
                "summary": "",
                "key_points": [],
                "method_used": "none",
                "model_source": "none",
                "mode": mode,
                "keywords": [],
                "nlp_stats": {},
                "rouge": {
                    "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                }
            }

        # 1. Compute input text NLP statistics & salient domain keywords
        nlp_stats = NLPProcessor.compute_stats(cleaned_text)
        keywords = NLPProcessor.extract_keywords(cleaned_text, top_k=8)

        # 2. Extract structured key takeaways (action points)
        key_points = NLPProcessor.extract_key_points(cleaned_text, top_k=4)

        # 3. Mode configuration mapping
        mode_configs = {
            "concise": {"max_length": max_length or 64, "min_length": min_length or 20, "ratio": 0.25},
            "balanced": {"max_length": max_length or 128, "min_length": min_length or 40, "ratio": 0.40},
            "detailed": {"max_length": max_length or 256, "min_length": min_length or 80, "ratio": 0.60}
        }
        cfg = mode_configs.get(mode, mode_configs["balanced"])

        summary = ""
        engine_used = ""
        model_source = ""

        # 4. Engine selection
        if method == "extractive":
            summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"])
            engine_used = "Extractive (TF-IDF Saliency)"
            model_source = "extractive_tfidf_engine"
        elif method == "abstractive":
            try:
                summary = self.abstractive_engine.summarize(
                    cleaned_text,
                    model_choice=model_name,
                    max_length=cfg["max_length"],
                    min_length=cfg["min_length"]
                )
                engine_used = f"Abstractive ({model_name.upper()} Transformer)"
                model_source = self.abstractive_engine.model_identifier
            except Exception as e:
                logger.warning(f"Abstractive engine error: {e}. Falling back to Extractive TF-IDF.")
                summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"])
                engine_used = "Extractive Fallback (TF-IDF)"
                model_source = "extractive_fallback_tfidf"
        else:  # "auto" or "hybrid"
            try:
                summary = self.abstractive_engine.summarize(
                    cleaned_text,
                    model_choice=model_name,
                    max_length=cfg["max_length"],
                    min_length=cfg["min_length"]
                )
                engine_used = f"Abstractive ({model_name.upper()} Transformer)"
                model_source = self.abstractive_engine.model_identifier
            except Exception as e:
                logger.info(f"Auto-selected Extractive NLP engine (Transformer note: {e})")
                summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"])
                engine_used = "Extractive (TF-IDF / Saliency)"
                model_source = "extractive_nlp_engine"

        # 5. Compute real-time ROUGE evaluation scores against original source text
        rouge_scores = NLPProcessor.compute_rouge(cleaned_text, summary)

        # 6. Compute summary-specific NLP statistics
        summary_stats = NLPProcessor.compute_stats(summary)

        return {
            "summary": summary,
            "key_points": key_points,
            "method_used": engine_used,
            "model_source": model_source,
            "mode": mode,
            "keywords": keywords,
            "nlp_stats": nlp_stats,
            "summary_stats": summary_stats,
            "rouge": rouge_scores
        }