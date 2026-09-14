import hashlib
from typing import Optional, Dict, Any, List
from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.extractive_summarizer import ExtractiveSummarizer
from textSummarizer.components.abstractive_summarizer import AbstractiveSummarizer
from textSummarizer.logging import logger

class PredictionPipeline:
    """Unified NLP Summarization Pipeline coordinating Extractive and Abstractive Transformer engines with LRU caching."""

    def __init__(self):
        self.abstractive_engine = AbstractiveSummarizer()
        self.extractive_engine = ExtractiveSummarizer()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_cache_entries = 128

    def _make_cache_key(self, text: str, mode: str, method: str, model_name: str, max_len: Optional[int], min_len: Optional[int]) -> str:
        key_raw = f"{text}|{mode}|{method}|{model_name}|{max_len}|{min_len}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

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
        Executes high-speed end-to-end NLP summarization, keyword extraction, key-points extraction,
        and ROUGE scoring in an optimized single-pass workflow.
        """
        cleaned_text = TextExtractor.clean_text(text)
        if not cleaned_text or not any(c.isalnum() for c in cleaned_text):
            return {
                "summary": "",
                "key_points": [],
                "method_used": "none",
                "model_source": "none",
                "mode": mode,
                "keywords": [],
                "nlp_stats": {},
                "summary_stats": {},
                "rouge": {
                    "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                }
            }

        # Check in-memory prediction cache
        cache_key = self._make_cache_key(cleaned_text, mode, method, model_name, max_length, min_length)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Single-pass language detection, tokenization, and sentence splitting
        lang_code, lang_name = NLPProcessor.detect_language(cleaned_text)
        pre_words = NLPProcessor.tokenize_words(cleaned_text)
        pre_sentences = NLPProcessor.split_sentences(cleaned_text)

        # 2. Compute input text NLP statistics & salient domain keywords with precomputed tokens
        nlp_stats = NLPProcessor.compute_stats(cleaned_text, precomputed_words=pre_words, precomputed_sentences=pre_sentences)
        nlp_stats["language"] = lang_code
        nlp_stats["language_name"] = lang_name
        keywords = NLPProcessor.extract_keywords(cleaned_text, top_k=8, precomputed_tokens=pre_words)

        # 3. Extract structured key takeaways (action points) with precomputed sentences
        key_points = NLPProcessor.extract_key_points(cleaned_text, top_k=4, precomputed_sentences=pre_sentences)

        # 4. Mode configuration mapping
        mode_configs = {
            "concise": {"max_length": max_length or 64, "min_length": min_length or 20, "ratio": 0.25},
            "balanced": {"max_length": max_length or 128, "min_length": min_length or 40, "ratio": 0.40},
            "detailed": {"max_length": max_length or 256, "min_length": min_length or 80, "ratio": 0.60}
        }
        cfg = mode_configs.get(mode, mode_configs["balanced"])

        summary = ""
        engine_used = ""
        model_source = ""

        # 5. Engine selection (For non-English text, use high-speed multilingual extractive saliency)
        if method == "extractive" or (method == "auto" and lang_code != "en"):
            summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"], precomputed_sentences=pre_sentences)
            engine_used = f"Multilingual Extractive ({lang_name})" if lang_code != "en" else "Extractive (TF-IDF Saliency)"
            model_source = f"multilingual_{lang_code}_engine" if lang_code != "en" else "extractive_tfidf_engine"
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
                summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"], precomputed_sentences=pre_sentences)
                engine_used = "Extractive Fallback (TF-IDF)"
                model_source = "extractive_fallback_tfidf"
        else:  # "auto" for English
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
                summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"], precomputed_sentences=pre_sentences)
                engine_used = "Extractive (TF-IDF / Saliency)"
                model_source = "extractive_nlp_engine"

        # 6. Compute real-time ROUGE evaluation scores against original source text
        rouge_scores = NLPProcessor.compute_rouge(cleaned_text, summary)

        # 7. Compute summary-specific NLP statistics
        summary_stats = NLPProcessor.compute_stats(summary)
        summary_stats["language"] = lang_code
        summary_stats["language_name"] = lang_name

        result = {
            "summary": summary,
            "key_points": key_points,
            "method_used": engine_used,
            "model_source": model_source,
            "mode": mode,
            "detected_language": lang_code,
            "language_name": lang_name,
            "keywords": keywords,
            "nlp_stats": nlp_stats,
            "summary_stats": summary_stats,
            "rouge": rouge_scores
        }

        # Store in LRU cache
        if len(self._cache) >= self._max_cache_entries:
            # Pop the oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result

        return result