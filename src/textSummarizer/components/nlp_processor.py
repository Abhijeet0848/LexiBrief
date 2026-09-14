import re
from collections import Counter
from typing import List, Dict, Any

class NLPProcessor:
    """Performs tokenization, sentence splitting, stopword removal, and keyword extraction."""

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
    def extract_keywords(cls, text: str, top_k: int = 6) -> List[Dict[str, Any]]:
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
    def compute_stats(cls, text: str) -> Dict[str, Any]:
        """Computes comprehensive NLP metrics for given text."""
        words = cls.tokenize_words(text)
        sentences = cls.split_sentences(text)
        chars = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
        reading_time_sec = round((word_count / 200) * 60, 1) # assuming 200 wpm

        return {
            "characters": chars,
            "words": word_count,
            "sentences": sentence_count,
            "avg_sentence_length": avg_sentence_len,
            "est_reading_time_sec": reading_time_sec
        }
