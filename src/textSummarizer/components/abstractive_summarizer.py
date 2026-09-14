import os
from typing import Optional, Dict, Any
from textSummarizer.logging import logger

class AbstractiveSummarizer:
    """Abstractive Sequence-to-Sequence Summarization Engine using Transformers (BART / T5 / Pegasus)."""

    def __init__(self):
        self._pipe = None
        self._tokenizer = None
        self._current_model_name = None
        self.model_identifier = "uninitialized"

    def load_pipeline(self, model_choice: str = "bart"):
        """
        Lazy loader for transformer summarization pipelines.
        Supports 'bart', 't5', 'pegasus', or local trained models.
        """
        model_map = {
            "bart": "sshleifer/distilbart-cnn-12-6",
            "bart-large": "facebook/bart-large-cnn",
            "t5": "t5-small",
            "pegasus": "google/pegasus-cnn_dailymail"
        }
        
        target_model = model_map.get(model_choice.lower(), model_choice)

        if self._pipe is not None and self._current_model_name == target_model:
            return self._pipe

        # 1. Local fine-tuned model
        try:
            from textSummarizer.config.configuration import ConfigurationManager
            from transformers import AutoTokenizer, pipeline
            config = ConfigurationManager().get_model_evaluation_config()

            if os.path.exists(config.model_path) and os.path.exists(config.tokenizer_path) and model_choice == "local":
                logger.info(f"Loading local fine-tuned model from {config.model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
                self._pipe = pipeline("summarization", model=config.model_path, tokenizer=self._tokenizer)
                self.model_identifier = "local_trained_model"
                self._current_model_name = "local"
                return self._pipe
        except Exception as e:
            logger.info(f"Local model not loaded: {e}")

        # 2. Hugging Face Pretrained Hub
        try:
            from transformers import AutoTokenizer, pipeline
            logger.info(f"Loading HuggingFace transformer summarizer: {target_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(target_model)
            self._pipe = pipeline("summarization", model=target_model, tokenizer=self._tokenizer)
            self.model_identifier = target_model
            self._current_model_name = target_model
            return self._pipe
        except Exception as e:
            logger.warning(f"Transformer pipeline ({target_model}) initialization notice: {e}")
            # Fallback to distilbart if pegasus or others fail
            if target_model != "sshleifer/distilbart-cnn-12-6":
                try:
                    fallback_id = "sshleifer/distilbart-cnn-12-6"
                    logger.info(f"Attempting fallback transformer: {fallback_id}")
                    self._tokenizer = AutoTokenizer.from_pretrained(fallback_id)
                    self._pipe = pipeline("summarization", model=fallback_id, tokenizer=self._tokenizer)
                    self.model_identifier = fallback_id
                    self._current_model_name = fallback_id
                    return self._pipe
                except Exception as ex:
                    logger.warning(f"Fallback transformer also unavailable: {ex}")
            self._pipe = None
            self.model_identifier = "unavailable"
            return None

    def summarize(self, text: str, model_choice: str = "bart", max_length: int = 128, min_length: int = 30) -> str:
        """Generates abstractive summary using transformer beam search."""
        pipe = self.load_pipeline(model_choice=model_choice)
        if pipe is None:
            raise RuntimeError("Transformer model could not be initialized into memory.")

        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 4,
            "max_length": max_length,
            "min_length": min_length,
            "truncation": True
        }
        output = pipe(text, **gen_kwargs)
        return output[0]["summary_text"]
