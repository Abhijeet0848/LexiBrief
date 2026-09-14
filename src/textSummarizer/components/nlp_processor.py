import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

# Precompiled regular expressions for maximum performance
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_RE_WORDS = re.compile(r'\b[A-Za-z0-9_-]{2,}\b')
_RE_VOWELS = re.compile(r'[aeiouy]')


class NLPProcessor:
    """High-performance NLP processing engine: tokenization, sentence splitting, stats, keywords, key-points, and ROUGE scoring."""

    STOPWORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
        'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
        'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot', 'could', 'couldn\'t',
        'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
        'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
        'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s',
        'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll',
        'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
        'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out',
        'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
        'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
        'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll',
        'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
        'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t',
        'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s',
        'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
        'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
    }

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Splits text into discrete sentences with boundary preservation using precompiled regex."""
        if not text:
            return []
        raw_sentences = _RE_SENTENCE_SPLIT.split(text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
        return sentences if sentences else [text.strip()]

    @classmethod
    def tokenize_words(cls, text: str) -> List[str]:
        """Extracts lowercase alphabetic words of length >= 2 using precompiled regex."""
        if not text:
            return []
        return _RE_WORDS.findall(text.lower())

    @classmethod
    def extract_keywords(cls, text: str, top_k: int = 8, precomputed_tokens: List[str] = None) -> List[Dict[str, Any]]:
        """Extracts salient domain keywords with rapid single-pass frequency counting."""
        tokens = precomputed_tokens if precomputed_tokens is not None else cls.tokenize_words(text)
        stopwords = cls.STOPWORDS
        filtered = [t for t in tokens if t not in stopwords and not t.isdigit() and len(t) >= 3]
        if not filtered:
            return []

        counts = Counter(filtered)
        top_items = counts.most_common(top_k)
        if not top_items:
            return []

        max_freq = top_items[0][1]
        return [
            {
                "keyword": word,
                "count": count,
                "importance": round(count / max_freq, 2)
            }
            for word, count in top_items
        ]

    @classmethod
    def extract_key_points(cls, text: str, top_k: int = 4, precomputed_sentences: List[str] = None) -> List[str]:
        """Extracts bulleted key takeaways and essential declarative clauses."""
        sentences = precomputed_sentences if precomputed_sentences is not None else cls.split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= top_k:
            return sentences

        # Precompute sentence tokens and keyword dictionary in single pass
        sent_tokens = [cls.tokenize_words(s) for s in sentences]
        all_words = [w for toks in sent_tokens for w in toks if w not in cls.STOPWORDS and not w.isdigit() and len(w) >= 3]
        
        if not all_words:
            return sentences[:top_k]

        word_counts = Counter(all_words)
        top_kw = dict(word_counts.most_common(15))

        scored = []
        num_sentences = len(sentences)
        for idx, sentence in enumerate(sentences):
            words = sent_tokens[idx]
            token_count = len(words)
            if token_count < 4:
                continue
            kw_score = sum(top_kw.get(w, 0) for w in words)
            pos_weight = 1.3 if idx == 0 else (1.1 if idx == num_sentences - 1 else 1.0)
            score = (kw_score / math.sqrt(token_count)) * pos_weight
            scored.append((idx, score, sentence))

        if not scored:
            return sentences[:top_k]

        top_sentences = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        top_sentences.sort(key=lambda x: x[0])  # Preserve narrative sequence
        return [s[2] for s in top_sentences]

    @classmethod
    def compute_stats(cls, text: str, precomputed_words: List[str] = None, precomputed_sentences: List[str] = None) -> Dict[str, Any]:
        """Computes comprehensive NLP metrics and Flesch readability score in a high-speed single pass."""
        words = precomputed_words if precomputed_words is not None else cls.tokenize_words(text)
        sentences = precomputed_sentences if precomputed_sentences is not None else cls.split_sentences(text)
        chars = len(text)
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_sentence_len = round(word_count / sentence_count, 1)
        reading_time_sec = round((word_count / 200) * 60, 1)  # 200 wpm baseline

        # Optimized syllable estimation using single pass
        syllable_count = 0
        for w in words:
            vowels = len(_RE_VOWELS.findall(w))
            syllable_count += max(1, vowels)

        if word_count > 0:
            asl = word_count / sentence_count
            asw = syllable_count / word_count
            flesch = round(206.835 - (1.015 * asl) - (84.6 * asw), 1)
            flesch = max(0.0, min(100.0, flesch))
        else:
            flesch = 100.0

        return {
            "characters": chars,
            "words": word_count,
            "sentences": sentence_count,
            "avg_sentence_length": avg_sentence_len,
            "est_reading_time_sec": reading_time_sec,
            "readability_score": flesch
        }

    @classmethod
    def compute_rouge(cls, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """
        High-Performance ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1).
        Uses O(N) rolling buffer memory for Longest Common Subsequence (LCS) calculation.
        """
        ref_tokens = cls.tokenize_words(reference)
        cand_tokens = cls.tokenize_words(candidate)

        if not ref_tokens or not cand_tokens:
            return {
                "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # Fast Identity Short-Circuit
        if ref_tokens == cand_tokens:
            perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            return {"rouge1": perfect, "rouge2": perfect, "rougeL": perfect}

        # Fast PRF helper
        def _calc_prf(ref_grams, cand_grams):
            if not cand_grams or not ref_grams:
                return 0.0, 0.0, 0.0
            ref_counts = Counter(ref_grams)
            cand_counts = Counter(cand_grams)
            overlap = sum(min(count, cand_counts.get(gram, 0)) for gram, count in ref_counts.items())
            precision = overlap / len(cand_grams) if cand_grams else 0.0
            recall = overlap / len(ref_grams) if ref_grams else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return round(precision, 4), round(recall, 4), round(f1, 4)

        # ROUGE-1
        r1_p, r1_r, r1_f1 = _calc_prf(ref_tokens, cand_tokens)

        # ROUGE-2
        ref_bigrams = [tuple(ref_tokens[i:i+2]) for i in range(len(ref_tokens) - 1)]
        cand_bigrams = [tuple(cand_tokens[i:i+2]) for i in range(len(cand_tokens) - 1)]
        r2_p, r2_r, r2_f1 = _calc_prf(ref_bigrams, cand_bigrams)

        # ROUGE-L: High-Speed O(N) Space Rolling DP
        m, n = len(ref_tokens), len(cand_tokens)
        
        # Always make the inner loop over the shorter array for maximum cache locality and speed
        if m < n:
            shorter, longer = ref_tokens, cand_tokens
            short_len, long_len = m, n
        else:
            shorter, longer = cand_tokens, ref_tokens
            short_len, long_len = n, m

        dp = [0] * (short_len + 1)
        for tok_long in longer:
            prev = 0
            for j, tok_short in enumerate(shorter):
                temp = dp[j + 1]
                if tok_long == tok_short:
                    dp[j + 1] = prev + 1
                else:
                    dp[j + 1] = max(dp[j + 1], dp[j])
                prev = temp

        lcs_len = dp[short_len]

        rl_p = round(lcs_len / n, 4) if n > 0 else 0.0
        rl_r = round(lcs_len / m, 4) if m > 0 else 0.0
        rl_f1 = round((2 * rl_p * rl_r) / (rl_p + rl_r), 4) if (rl_p + rl_r) > 0 else 0.0

        return {
            "rouge1": {"precision": r1_p, "recall": r1_r, "f1": r1_f1},
            "rouge2": {"precision": r2_p, "recall": r2_r, "f1": r2_f1},
            "rougeL": {"precision": rl_p, "recall": rl_r, "f1": rl_f1}
        }
