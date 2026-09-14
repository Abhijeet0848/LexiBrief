import os
from typing import Optional, Dict, Any
from textSummarizer.logging import logger

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class AbstractiveSummarizer:
    """High-Performance Abstractive Sequence-to-Sequence Summarization Engine using Transformers (BART / T5 / Pegasus)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_name = None
        self.model_identifier = "uninitialized"

    def load_model(self, model_choice: str = "bart"):
        """
        Lazy loader for transformer seq2seq summarization models with local caching.
        Supports 'bart', 'bart-large', 't5', 'pegasus', or local trained models.
        """
        model_map = {
            "bart": "sshleifer/distilbart-cnn-12-6",
            "bart-large": "facebook/bart-large-cnn",
            "t5": "t5-small",
            "pegasus": "google/pegasus-cnn_dailymail"
        }
        
        target_model = model_map.get(model_choice.lower(), model_choice)

        if self._model is not None and self._current_model_name == target_model:
            return self._model, self._tokenizer

        # 1. Local fine-tuned model
        try:
            from textSummarizer.config.configuration import ConfigurationManager
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            config = ConfigurationManager().get_model_evaluation_config()

            if os.path.exists(config.model_path) and os.path.exists(config.tokenizer_path) and model_choice == "local":
                logger.info(f"Loading local fine-tuned model from {config.model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
                self.model_identifier = "local_trained_model"
                self._current_model_name = "local"
                return self._model, self._tokenizer
        except Exception as e:
            logger.info(f"Local model not loaded: {e}")

        # 2. Pretrained Seq2Seq Hugging Face Model
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Loading Seq2Seq transformer summarizer: {target_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(target_model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(target_model)
            self.model_identifier = target_model
            self._current_model_name = target_model
            return self._model, self._tokenizer
        except Exception as e:
            logger.warning(f"Transformer model ({target_model}) initialization notice: {e}")
            # Fallback to distilbart if pegasus or others fail
            if target_model != "sshleifer/distilbart-cnn-12-6":
                try:
                    fallback_id = "sshleifer/distilbart-cnn-12-6"
                    logger.info(f"Attempting fallback transformer: {fallback_id}")
                    self._tokenizer = AutoTokenizer.from_pretrained(fallback_id)
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(fallback_id)
                    self.model_identifier = fallback_id
                    self._current_model_name = fallback_id
                    return self._model, self._tokenizer
                except Exception as ex:
                    logger.warning(f"Fallback transformer also unavailable: {ex}")
            self._model = None
            self._tokenizer = None
            self.model_identifier = "unavailable"
            return None, None

    def summarize(self, text: str, model_choice: str = "bart", max_length: int = 128, min_length: int = 30) -> str:
        """Generates abstractive summary using direct AutoModelForSeq2SeqLM inference."""
        # Short-circuit for very short inputs to avoid forced hallucination / repetition
        words = text.split()
        if len(words) <= 10:
            return text.strip()

        model, tokenizer = self.load_model(model_choice=model_choice)
        if model is None or tokenizer is None:
            raise RuntimeError("Transformer model could not be initialized into memory.")

        # T5 models require a task prefix for summarization
        input_text = text
        if "t5" in model_choice.lower() and not input_text.strip().lower().startswith("summarize:"):
            input_text = f"summarize: {input_text}"

        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        input_token_count = inputs["input_ids"].shape[1]
        
        # Adaptively bound lengths to prevent over-generation
        eff_min_length = min(min_length, max(5, input_token_count // 2))
        eff_max_length = max(eff_min_length + 10, min(max_length, input_token_count + 15))

        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 2,               # Optimized for fast generation
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "max_length": eff_max_length,
            "min_length": eff_min_length
        }

        if _TORCH_AVAILABLE:
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
        else:
            outputs = model.generate(**inputs, **gen_kwargs)

        raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean special sentence markers like <n> used by Pegasus
        clean_summary = raw_summary.replace("<n>", " ").strip()
        import re
        clean_summary = re.sub(r'\s+', ' ', clean_summary)
        return clean_summary
