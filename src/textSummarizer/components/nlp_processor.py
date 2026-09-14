import re
import math
from collections import Counter
from typing import List, Dict, Any

class NLPProcessor:
    """Performs tokenization, sentence splitting, stopword removal, keyword extraction, key-point extraction, and ROUGE evaluation."""

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
        """Splits text into discrete sentences with boundary preservation."""
        if not text:
            return []
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            if len(s_clean) > 5:
                sentences.append(s_clean)
        return sentences if sentences else [text.strip()]

    @classmethod
    def tokenize_words(cls, text: str) -> List[str]:
        """Extracts alphabetic words of length >= 2."""
        return re.findall(r'\b[A-Za-z0-9_-]{2,}\b', text.lower())

    @classmethod
    def extract_keywords(cls, text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Extracts salient domain keywords using TF and stopword filtering."""
        tokens = cls.tokenize_words(text)
        filtered = [t for t in tokens if t not in cls.STOPWORDS and not t.isdigit() and len(t) >= 3]
        if not filtered:
            return []

        counts = Counter(filtered)
        max_freq = counts.most_common(1)[0][1] if counts else 1

        salient = []
        for word, count in counts.most_common(top_k):
            salient.append({
                "keyword": word,
                "count": count,
                "importance": round(count / max_freq, 2)
            })
        return salient

    @classmethod
    def extract_key_points(cls, text: str, top_k: int = 4) -> List[str]:
        """Extracts bulleted key takeaways and essential declarative clauses."""
        sentences = cls.split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= top_k:
            return sentences

        # Score sentences based on informative keywords and position
        keywords_dict = {item['keyword']: item['count'] for item in cls.extract_keywords(text, top_k=15)}
        scored = []
        for idx, sentence in enumerate(sentences):
            words = cls.tokenize_words(sentence)
            if len(words) < 4:
                continue
            kw_score = sum(keywords_dict.get(w, 0) for w in words)
            pos_weight = 1.3 if idx == 0 else (1.1 if idx == len(sentences) - 1 else 1.0)
            score = (kw_score / math.sqrt(len(words))) * pos_weight
            scored.append((idx, score, sentence))

        if not scored:
            return sentences[:top_k]

        top_sentences = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        top_sentences.sort(key=lambda x: x[0])  # Preserve narrative sequence
        return [s[2] for s in top_sentences]

    @classmethod
    def compute_stats(cls, text: str) -> Dict[str, Any]:
        """Computes comprehensive NLP metrics and readability score for text."""
        words = cls.tokenize_words(text)
        sentences = cls.split_sentences(text)
        chars = len(text)
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_sentence_len = round(word_count / sentence_count, 1)
        reading_time_sec = round((word_count / 200) * 60, 1)  # 200 wpm baseline

        # Approximate syllable estimation for Flesch Reading Ease
        syllable_count = 0
        for w in words:
            w_lower = w.lower()
            vowels = len(re.findall(r'[aeiouy]', w_lower))
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
        Computes ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1)
        comparing reference text with the generated summary.
        """
        ref_tokens = cls.tokenize_words(reference)
        cand_tokens = cls.tokenize_words(candidate)

        if not ref_tokens or not cand_tokens:
            return {
                "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # Helper for n-gram overlap
        def _get_ngrams(tokens: List[str], n: int):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

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
        ref_bigrams = _get_ngrams(ref_tokens, 2)
        cand_bigrams = _get_ngrams(cand_tokens, 2)
        r2_p, r2_r, r2_f1 = _calc_prf(ref_bigrams, cand_bigrams)

        # ROUGE-L (Longest Common Subsequence)
        # Optimized LCS length
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if ref_tokens[i] == cand_tokens[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        lcs_len = dp[m][n]

        rl_p = round(lcs_len / n, 4) if n > 0 else 0.0
        rl_r = round(lcs_len / m, 4) if m > 0 else 0.0
        rl_f1 = round((2 * rl_p * rl_r) / (rl_p + rl_r), 4) if (rl_p + rl_r) > 0 else 0.0

        return {
            "rouge1": {"precision": r1_p, "recall": r1_r, "f1": r1_f1},
            "rouge2": {"precision": r2_p, "recall": r2_r, "f1": r2_f1},
            "rougeL": {"precision": rl_p, "recall": rl_r, "f1": rl_f1}
        }
