import math
from collections import Counter
from typing import List, Dict, Tuple
from textSummarizer.components.nlp_processor import NLPProcessor

class ExtractiveSummarizer:
    """Extractive Summarization Engine utilizing TF-IDF and sentence saliency scoring."""

    @staticmethod
    def _compute_tf_idf_scores(sentences: List[str]) -> Dict[str, float]:
        """Calculates TF-IDF term weights across sentences in the document."""
        num_sentences = len(sentences)
        if num_sentences == 0:
            return {}

        # Tokenize each sentence
        sentence_tokens = [NLPProcessor.tokenize_words(s) for s in sentences]
        
        # Document Frequency (DF) across sentences
        df: Counter = Counter()
        for tokens in sentence_tokens:
            unique_words = set(w for w in tokens if w not in NLPProcessor.STOPWORDS and len(w) >= 3)
            for w in unique_words:
                df[w] += 1

        # Global TF
        all_words = [w for tokens in sentence_tokens for w in tokens if w not in NLPProcessor.STOPWORDS and len(w) >= 3]
        tf: Counter = Counter(all_words)
        max_tf = max(tf.values()) if tf else 1

        # Compute TF-IDF
        weights = {}
        for w, count in tf.items():
            norm_tf = count / max_tf
            idf = math.log((num_sentences + 1) / (df[w] + 1)) + 1.0
            weights[w] = norm_tf * idf

        return weights

    @classmethod
    def summarize(cls, text: str, ratio: float = 0.35, min_sentences: int = 1, max_sentences: int = 10) -> str:
        """Extracts top representative sentences using semantic saliency, position bias, and length penalties."""
        sentences = NLPProcessor.split_sentences(text)
        if not sentences or len(sentences) <= 2:
            return text.strip()

        word_weights = cls._compute_tf_idf_scores(sentences)
        
        scored_sentences: List[Tuple[int, float, str]] = []
        for idx, sentence in enumerate(sentences):
            tokens = [w for w in NLPProcessor.tokenize_words(sentence) if w in word_weights]
            if not tokens:
                continue

            # Base score from keyword weights
            raw_score = sum(word_weights.get(w, 0.0) for w in tokens)
            
            # Length normalization (penalize overly short or overly verbose fragments)
            token_count = len(tokens)
            length_norm = math.sqrt(token_count) if token_count > 0 else 1.0
            
            # Position bias (lead sentences convey introductory thesis)
            position_multiplier = 1.30 if idx == 0 else (1.15 if idx == 1 else (1.10 if idx == len(sentences) - 1 else 1.0))

            final_score = (raw_score / length_norm) * position_multiplier
            scored_sentences.append((idx, final_score, sentence))

        if not scored_sentences:
            return text.strip()

        # Determine target number of sentences based on ratio
        target_count = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
        
        # Pick highest scoring sentences
        top_ranked = sorted(scored_sentences, key=lambda item: item[1], reverse=True)[:target_count]
        
        # Restore chronological order for narrative flow
        chronological = sorted(top_ranked, key=lambda item: item[0])
        return " ".join(item[2] for item in chronological)
