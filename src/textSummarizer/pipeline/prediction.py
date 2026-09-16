import hashlib
from typing import Optional, Dict, Any, List
from textSummarizer.components.text_extractor import TextExtractor
from textSummarizer.components.nlp_processor import NLPProcessor
from textSummarizer.components.extractive_summarizer import ExtractiveSummarizer
from textSummarizer.components.abstractive_summarizer import AbstractiveSummarizer
from textSummarizer.components.translator import Translator
from textSummarizer.logging import logger

class PredictionPipeline:
    """Unified NLP Summarization Pipeline coordinating Extractive, Abstractive Transformer, and Neural Translation engines with LRU caching, Persona routing, and Explainable Attribution."""

    def __init__(self):
        self.abstractive_engine = AbstractiveSummarizer()
        self.extractive_engine = ExtractiveSummarizer()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_cache_entries = 128

    def _make_cache_key(self, text: str, mode: str, method: str, model_name: str, persona: str, max_len: Optional[int], min_len: Optional[int], translate_to_english: bool = False) -> str:
        key_raw = f"{text}|{mode}|{method}|{model_name}|{persona}|{max_len}|{min_len}|{translate_to_english}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def predict(
        self,
        text: str,
        mode: str = "balanced",
        method: str = "auto",
        model_name: str = "bart",
        persona: str = "general",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        translate_to_english: bool = False
    ) -> Dict[str, Any]:
        """
        Executes high-speed end-to-end NLP summarization, persona conditioning, keyword extraction,
        explainable sentence attribution mapping, and ROUGE scoring in an optimized single-pass workflow.
        """
        cleaned_text = TextExtractor.clean_text(text)
        if not cleaned_text or not any(c.isalnum() for c in cleaned_text):
            return {
                "summary": "",
                "key_points": [],
                "method_used": "none",
                "model_source": "none",
                "mode": mode,
                "persona": persona,
                "keywords": [],
                "attribution": {"source_sentences": [], "summary_sentences": [], "attribution_map": []},
                "nlp_stats": {},
                "summary_stats": {},
                "rouge": {
                    "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                }
            }

        persona_clean = (persona or "general").lower().strip()

        # Check in-memory prediction cache
        cache_key = self._make_cache_key(cleaned_text, mode, method, model_name, persona_clean, max_length, min_length, translate_to_english)
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

        # 5. Engine selection (For non-English text or extractive/persona requests)
        if method == "extractive" or (method == "auto" and lang_code != "en") or persona_clean in ["executive", "technical", "eli5", "action_items"]:
            summary = self.extractive_engine.summarize(
                cleaned_text,
                ratio=cfg["ratio"],
                precomputed_sentences=pre_sentences,
                persona=persona_clean
            )
            engine_used = f"Extractive Persona ({persona_clean.capitalize()})" if persona_clean != "general" else (f"Multilingual Extractive ({lang_name})" if lang_code != "en" else "Extractive (TF-IDF Saliency)")
            model_source = f"extractive_persona_{persona_clean}" if persona_clean != "general" else (f"multilingual_{lang_code}_engine" if lang_code != "en" else "extractive_tfidf_engine")
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
                summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"], precomputed_sentences=pre_sentences, persona=persona_clean)
                engine_used = "Extractive Fallback (TF-IDF)"
                model_source = "extractive_fallback_tfidf"
        else:  # "auto" for English general
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
                summary = self.extractive_engine.summarize(cleaned_text, ratio=cfg["ratio"], precomputed_sentences=pre_sentences, persona=persona_clean)
                engine_used = "Extractive (TF-IDF / Saliency)"
                model_source = "extractive_nlp_engine"

        # 6. Optional Neural Translation to English (if requested and source is multilingual)
        original_summary = summary
        original_key_points = key_points
        translated_flag = False

        if translate_to_english and lang_code != "en":
            try:
                translated_summary, _ = Translator.translate_to_english(summary, source_lang=lang_code)
                translated_kp = [Translator.translate_to_english(kp, source_lang=lang_code)[0] for kp in key_points]
                if translated_summary and translated_summary.strip():
                    summary = translated_summary
                    key_points = translated_kp
                    engine_used = f"{engine_used} + English Translation"
                    translated_flag = True
            except Exception as trans_err:
                logger.warning(f"Translation to English error: {trans_err}")

        # 7. Compute real-time ROUGE evaluation scores against original source text
        rouge_scores = NLPProcessor.compute_rouge(cleaned_text, original_summary)

        # 8. Compute Explainable Sentence Attribution Mapping
        attribution_data = NLPProcessor.compute_attribution(cleaned_text, summary if not translated_flag else original_summary)

        # 9. Compute summary-specific NLP statistics
        summary_stats = NLPProcessor.compute_stats(summary)
        summary_stats["language"] = "en" if translated_flag else lang_code
        summary_stats["language_name"] = "English" if translated_flag else lang_name

        result = {
            "summary": summary,
            "original_summary": original_summary,
            "key_points": key_points,
            "original_key_points": original_key_points,
            "translated_to_english": translated_flag,
            "original_language": lang_code,
            "method_used": engine_used,
            "model_source": model_source,
            "mode": mode,
            "persona": persona_clean,
            "detected_language": lang_code,
            "language_name": lang_name,
            "keywords": keywords,
            "attribution": attribution_data,
            "nlp_stats": nlp_stats,
            "summary_stats": summary_stats,
            "rouge": rouge_scores
        }

        # Store in LRU cache
        if len(self._cache) >= self._max_cache_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result

        return result