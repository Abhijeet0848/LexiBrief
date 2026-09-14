import os
from typing import Optional, Dict, Any
from textSummarizer.logging import logger

class AbstractiveSummarizer:
    """Abstractive Sequence-to-Sequence Summarization Engine using Google Pegasus."""

    def __init__(self):
        self._pipe = None
        self._tokenizer = None
        self.model_identifier = "uninitialized"

    def load_pipeline(self):
        """Lazy loader for the transformer summarization pipeline."""
        if self._pipe is not None:
            return self._pipe

        # 1. Local fine-tuned model
        try:
            from textSummarizer.config.configuration import ConfigurationManager
            from transformers import AutoTokenizer, pipeline
            config = ConfigurationManager().get_model_evaluation_config()

            if os.path.exists(config.model_path) and os.path.exists(config.tokenizer_path):
                logger.info(f"Loading local fine-tuned Pegasus from {config.model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
                self._pipe = pipeline("summarization", model=config.model_path, tokenizer=self._tokenizer)
                self.model_identifier = "local_trained_pegasus"
                return self._pipe
        except Exception as e:
            logger.info(f"Local model not loaded: {e}")

        # 2. Hugging Face Pretrained Hub
        try:
            from transformers import AutoTokenizer, pipeline
            hub_id = "google/pegasus-cnn_dailymail"
            logger.info(f"Loading pre-trained HuggingFace Pegasus: {hub_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(hub_id)
            self._pipe = pipeline("summarization", model=hub_id, tokenizer=self._tokenizer)
            self.model_identifier = "hf_pegasus_cnn_dailymail"
            return self._pipe
        except Exception as e:
            logger.warning(f"Transformer pipeline initialization exception: {e}")
            self._pipe = None
            self.model_identifier = "unavailable"
            return None

    def summarize(self, text: str, max_length: int = 128, min_length: int = 30) -> str:
        """Generates abstractive summary using transformer beam search."""
        pipe = self.load_pipeline()
        if pipe is None:
            raise RuntimeError("Transformer model could not be loaded into memory.")

        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 4,
            "max_length": max_length,
            "min_length": min_length,
            "truncation": True
        }
        output = pipe(text, **gen_kwargs)
        return output[0]["summary_text"]
