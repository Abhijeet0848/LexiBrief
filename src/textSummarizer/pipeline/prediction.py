import os
import re
import math
from collections import Counter
from typing import Optional, Dict, Any
from textSummarizer.logging import logger

class PredictionPipeline:
    def __init__(self):
        self._tokenizer = None
        self._pipe = None
        self._model_source = "uninitialized"

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        # 1. Try local trained model first
        try:
            from textSummarizer.config.configuration import ConfigurationManager
            from transformers import AutoTokenizer, pipeline
            config = ConfigurationManager().get_model_evaluation_config()
            
            if os.path.exists(config.model_path) and os.path.exists(config.tokenizer_path):
                logger.info(f"Loading local trained model from: {config.model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
                self._pipe = pipeline("summarization", model=config.model_path, tokenizer=self._tokenizer)
                self._model_source = "local_trained_pegasus"
                return self._pipe
        except Exception as e:
            logger.warning(f"Could not load local trained model: {e}")

        # 2. Try HuggingFace hub model
        try:
            from transformers import AutoTokenizer, pipeline
            hub_model = "google/pegasus-cnn_dailymail"
            logger.info(f"Loading pre-trained Hugging Face model: {hub_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(hub_model)
            self._pipe = pipeline("summarization", model=hub_model, tokenizer=self._tokenizer)
            self._model_source = "hf_pegasus_cnn"
            return self._pipe
        except Exception as e:
            logger.warning(f"Transformer pipeline unavailable ({e}). Using NLP extractive engine fallback.")
            self._pipe = "extractive_fallback"
            self._model_source = "extractive_nlp_engine"
            return self._pipe

    def _extractive_fallback_summarize(self, text: str, target_ratio: float = 0.35) -> str:
        """High-grade statistical extractive summarizer for resilience against missing weights/DLL restrictions."""
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
        if not sentences:
            return text
        if len(sentences) <= 2:
            return text

        # Word frequencies
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {
            'the', 'and', 'is', 'in', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as',
            'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
            'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said',
            'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will'
        }
        filtered_words = [w for w in words if w not in stopwords]
        freq = Counter(filtered_words)
        max_f = max(freq.values()) if freq else 1
        word_scores = {w: count / max_f for w, count in freq.items()}

        # Score sentences
        sentence_scores = []
        for i, s in enumerate(sentences):
            s_words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            score = sum(word_scores.get(w, 0) for w in s_words)
            length_norm = max(len(s_words), 1)
            # Boost early sentences (lead bias)
            position_bias = 1.25 if i == 0 else (1.1 if i == 1 else 1.0)
            sentence_scores.append((i, (score / length_norm) * position_bias, s))

        # Select top sentences
        target_count = max(1, math.ceil(len(sentences) * target_ratio))
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:target_count]
        # Sort back to original chronological order
        ordered_summary = sorted(top_sentences, key=lambda x: x[0])
        return " ".join([s[2] for s in ordered_summary])

    def predict(self, text: str, mode: str = "balanced", max_length: Optional[int] = None, min_length: Optional[int] = None) -> Dict[str, Any]:
        """Summarizes the input text and returns summary with metadata."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return {"summary": "", "model_source": "none"}

        pipe = self._get_pipeline()

        # Length configuration based on mode
        length_configs = {
            "concise": {"max_length": max_length or 64, "min_length": min_length or 20, "ratio": 0.25},
            "balanced": {"max_length": max_length or 128, "min_length": min_length or 40, "ratio": 0.40},
            "detailed": {"max_length": max_length or 256, "min_length": min_length or 80, "ratio": 0.60}
        }
        cfg = length_configs.get(mode, length_configs["balanced"])

        if pipe == "extractive_fallback" or not callable(getattr(pipe, "__call__", None)):
            summary = self._extractive_fallback_summarize(cleaned_text, target_ratio=cfg["ratio"])
        else:
            try:
                gen_kwargs = {
                    "length_penalty": 0.8,
                    "num_beams": 4,
                    "max_length": cfg["max_length"],
                    "min_length": cfg["min_length"],
                    "truncation": True
                }
                output = pipe(cleaned_text, **gen_kwargs)
                summary = output[0]["summary_text"]
            except Exception as e:
                logger.warning(f"Error during transformer generation ({e}), falling back to extractive engine.")
                summary = self._extractive_fallback_summarize(cleaned_text, target_ratio=cfg["ratio"])
                self._model_source = "extractive_nlp_engine (fallback)"

        return {
            "summary": summary,
            "model_source": self._model_source,
            "mode": mode
        }