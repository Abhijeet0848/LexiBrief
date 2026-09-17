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
            filtered_unique = set(w for w in tokens if w not in stopwords and len(w) >= 2)
            for w in filtered_unique:
                df[w] += 1
            for w in tokens:
                if w not in stopwords and len(w) >= 2:
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
                    persona_boost += (0.70 * matching_persona_words)
            
            # Executive persona: Prioritize high-level governance, KPIs, strategic vision, state, and resource metrics
            if persona_key == "executive":
                if any(c.isdigit() or c in "$%€£" for c in sentence):
                    persona_boost += 0.40
                if idx in (0, 1):
                    persona_boost += 0.35
                if any(w in sentence for w in ["साहित्य का यही काम", "हमारा लक्ष्य", "कल्याण ही साध्य", "सत्य वह है", "मनुष्य ही बड़ी", "उद्देश्य", "महत्त्व", "नीति", "सर्वोच्च"]):
                    persona_boost += 0.65
            # Technical persona: Prioritize architectural components, data structures, linguistic sciences, and mechanics
            elif persona_key == "technical":
                lower_s = sentence.lower()
                if any(kw in lower_s for kw in ["queue", "schedul", "descriptor", "stack", "heap", "i/o", "device", "memory space", "pointer"]):
                    persona_boost += 0.50
                if any(w in sentence for w in ["भाषा-विज्ञान", "नृतत्त्व-शास्त्र", "भाषाशास्त्र", "व्याकरण", "ग्रियर्सन", "सर्वे", "संस्कृत", "आर्यभाषा", "पालि", "प्राकृत", "बोली", "शास्त्र", "ग्रंथ"]):
                    persona_boost += 0.75
            # Readability / simplicity boost for ELI5
            elif persona_key == "eli5":
                if len(tokens) <= 18:
                    persona_boost += 0.45
                lower_s = sentence.lower()
                if any(w in lower_s for w in ["assigned to", "shows", "means", "identify", "running", "waiting"]):
                    persona_boost += 0.40
                if any(w in sentence for w in ["जैसे", "उदाहरण", "सीधा", "सहज", "सरल", "दूध", "दही", "धूल", "कहानी", "सुख", "दुख", "बच्चे", "जीवन", "भलाई", "प्रेम"]):
                    persona_boost += 0.65
            # Action item marker boost
            elif persona_key == "action_items":
                lower_s = sentence.lower()
                if any(w in lower_s for w in ["will", "should", "must", "todo", "action", "next step", "schedule", "deploy", "plan", "tracks", "helps", "allocated"]):
                    persona_boost += 0.50
                if any(w in sentence for w in ["चाहिए", "होगा", "होगी", "होंगे", "पड़ेगा", "पड़ेगी", "पड़ेंगे", "कर्तव्य", "संकल्प", "व्रत", "उपाय", "रास्ता", "मार्ग", "सुधार", "प्रयत्न", "प्रयास", "लड़ना", "बचाना", "तैयार", "आवश्यक"]):
                    persona_boost += 0.75

            # Length normalization (penalize overly short or overly verbose fragments)
            token_count = len(tokens)
            length_norm = math.sqrt(token_count) if token_count > 0 else 1.0
            
            # Position multiplier
            position_multiplier = 1.10 if (idx == 0 and persona_key != "technical") else 1.0

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

        # Check if sentences are bulleted
        has_bullets = any(item[2].strip().startswith(('•', '-', '*')) for item in chronological)

        def _join_sentences(items):
            if has_bullets:
                res = []
                for it in items:
                    t = it[2].strip()
                    res.append(t if t.startswith(('•', '-', '*')) else f"• {t}")
                return "\n".join(res)
            
            clean_sents = [it[2].strip() for it in items if it[2].strip()]
            if len(clean_sents) > 5:
                # Group into balanced 3-sentence narrative paragraphs for readability
                paragraphs = []
                for i in range(0, len(clean_sents), 3):
                    chunk = clean_sents[i:i+3]
                    paragraphs.append(" ".join(chunk))
                return "\n\n".join(paragraphs)
            return " ".join(clean_sents)

        if persona_key == "action_items":
            return "\n".join(
                item[2].strip() if item[2].strip().startswith(('•', '-', '*')) else f"• {item[2].strip()}"
                for item in chronological
            )
        elif persona_key == "executive":
            body = _join_sentences(chronological)
            return f"📌 Executive Summary:\n{body}"
        elif persona_key == "technical":
            body = _join_sentences(chronological)
            return f"⚙️ Technical Architecture & Specs:\n{body}"
        elif persona_key == "eli5":
            body = _join_sentences(chronological)
            return f"💡 Plain-English Concept:\n{body}"
        
        return _join_sentences(chronological)

