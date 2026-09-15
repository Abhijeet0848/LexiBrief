import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
from textSummarizer.components.nlp_processor import NLPProcessor

class ExtractiveSummarizer:
    """High-performance Extractive Summarization Engine utilizing TF-IDF and sentence saliency scoring."""

    @staticmethod
    def _compute_tf_idf_scores(sentence_tokens: List[List[str]]) -> Dict[str, float]:
        """Calculates TF-IDF term weights across sentences in a rapid single-pass."""
        num_sentences = len(sentence_tokens)
        if num_sentences == 0:
            return {}

        stopwords = NLPProcessor.STOPWORDS
        df: Counter = Counter()
        tf: Counter = Counter()

        for tokens in sentence_tokens:
            filtered_unique = set(w for w in tokens if w not in stopwords and len(w) >= 3)
            for w in filtered_unique:
                df[w] += 1
            for w in tokens:
                if w not in stopwords and len(w) >= 3:
                    tf[w] += 1

        if not tf:
            return {}

        max_tf = max(tf.values())
        weights = {}
        for w, count in tf.items():
            norm_tf = count / max_tf
            idf = math.log((num_sentences + 1) / (df[w] + 1)) + 1.0
            weights[w] = norm_tf * idf

        return weights

    @classmethod
    def summarize(
        cls,
        text: str,
        ratio: float = 0.35,
        min_sentences: int = 1,
        max_sentences: int = 10,
        precomputed_sentences: Optional[List[str]] = None,
        persona: str = "general"
    ) -> str:
        """Extracts top representative sentences using semantic saliency, persona bias, position bias, and length penalties."""
        sentences = precomputed_sentences if precomputed_sentences is not None else NLPProcessor.split_sentences(text)
        if not sentences or len(sentences) <= 2:
            return text.strip()

        # Tokenize each sentence once
        sentence_tokens = [NLPProcessor.tokenize_words(s) for s in sentences]
        word_weights = cls._compute_tf_idf_scores(sentence_tokens)
        if not word_weights:
            return text.strip()

        num_sentences = len(sentences)
        scored_sentences: List[Tuple[int, float, str]] = []
        
        persona_key = (persona or "general").lower().strip()
        persona_kw = NLPProcessor.PERSONA_KEYWORDS.get(persona_key, set())

        for idx, sentence in enumerate(sentences):
            tokens = [w for w in sentence_tokens[idx] if w in word_weights]
            if not tokens:
                continue

            # Base score from keyword weights
            raw_score = sum(word_weights[w] for w in tokens)
            
            # Persona-specific weighting boosts
            persona_boost = 1.0
            if persona_kw:
                matching_persona_words = sum(1 for w in sentence_tokens[idx] if w in persona_kw)
                if matching_persona_words > 0:
                    persona_boost += (0.25 * matching_persona_words)
            
            # Numeric & metric boosts for executive persona
            if persona_key == "executive":
                if any(c.isdigit() or c in "$%€£" for c in sentence):
                    persona_boost += 0.35
            # Readability / simplicity boost for ELI5
            elif persona_key == "eli5":
                if len(tokens) <= 15:
                    persona_boost += 0.30
            # Action item marker boost
            elif persona_key == "action_items":
                lower_s = sentence.lower()
                if any(w in lower_s for w in ["will", "should", "must", "todo", "action", "next step", "schedule", "deploy", "plan"]):
                    persona_boost += 0.45

            # Length normalization (penalize overly short or overly verbose fragments)
            token_count = len(tokens)
            length_norm = math.sqrt(token_count) if token_count > 0 else 1.0
            
            # Gentle position bias that respects semantic content saliency
            position_multiplier = 1.05 if idx == 0 else 1.0

            final_score = (raw_score / length_norm) * position_multiplier * persona_boost
            scored_sentences.append((idx, final_score, sentence))

        if not scored_sentences:
            return text.strip()

        # Determine target number of sentences based on ratio
        target_count = max(min_sentences, min(max_sentences, math.ceil(num_sentences * ratio)))
        
        # Pick highest scoring sentences
        top_ranked = sorted(scored_sentences, key=lambda item: item[1], reverse=True)[:target_count]
        
        # Restore chronological order for narrative flow
        chronological = sorted(top_ranked, key=lambda item: item[0])

        if persona_key == "action_items":
            return "\n".join(f"• {item[2].strip()}" for item in chronological)
        
        return " ".join(item[2] for item in chronological)

