import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

# Precompiled regular expressions for multilingual performance (supports English, Indic, and Global scripts)
_RE_SENTENCE_SPLIT = re.compile(r'(?:(?<=[.!?।॥؛۔\n])\s+)|(?:\n+)')
_RE_WORDS = re.compile(r'[^\s.,!?;:()\[\]{}"\'`।॥؛۔«»–—/\\<>+=@#$%^&*~|]+', re.UNICODE)
_RE_VOWELS = re.compile(r'[aeiouy\u0904-\u0914\u0960-\u0963]', re.UNICODE)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
    "pa": "Punjabi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic"
}


class NLPProcessor:
    """High-performance NLP processing engine: multilingual tokenization, language detection, stats, keywords, and ROUGE."""

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
        'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves', 'uses', 'used', 'using',
        'also', 'such', 'major', 'large', 'different', 'examples', 'combines', 'improved', 'process',
        'tasks', 'learn', 'complex', 'patterns', 'component', 'provides', 'based',

        # Hindi & Indic Stopwords (particles, pronouns, auxiliary verbs, conjunctions, prepositions)
        'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'ने', 'है', 'हैं', 'था', 'थे', 'थी', 'थीं',
        'और', 'या', 'तथा', 'एवं', 'भी', 'तो', 'ही', 'होता', 'होती', 'होते', 'होना', 'होने', 'हो', 'हों', 'होगा', 'होगी', 'होंगे', 'होकर',
        'रहा', 'रहे', 'रही', 'गया', 'गए', 'गई', 'कर', 'करता', 'करते', 'करती', 'करना', 'करने', 'करके',
        'किया', 'किए', 'यह', 'वह', 'ये', 'वे', 'इस', 'उस', 'इन', 'उन', 'जिस', 'जिसके',
        'जिसकी', 'जिसमें', 'जिससे', 'जिसे', 'जिन', 'जिन्हें', 'जिसका', 'जिनका', 'जिनकी', 'जिनके',
        'जो', 'जब', 'तब', 'अब', 'कब', 'तक', 'यहाँ', 'वहाँ', 'कहाँ', 'जहाँ', 'कैसे', 'ऐसा', 'ऐसी', 'ऐसे',
        'वैसा', 'वैसे', 'वैसी', 'जैसे', 'तैसे', 'क्या', 'क्यों', 'कौन', 'किस', 'किसी', 'कुछ', 'कोई',
        'बहुत', 'ज्यादा', 'कम', 'अधिक', 'सकता', 'सकते', 'सकती', 'सकना', 'सके', 'सका', 'सकी', 'सकेंगे', 'सकेंगी',
        'हुए', 'हुई', 'हुआ', 'अपने', 'अपनी', 'अपना', 'अपनों', 'द्वारा', 'लिए', 'साथ', 'बाद', 'पहले', 'बीच',
        'अनुसार', 'कारण', 'बारे', 'तरह', 'रूप', 'प्रकार', 'भाँति', 'भांति', 'दिया', 'दिए', 'दी', 'देना', 'देते', 'देती',
        'लेना', 'लेते', 'लेती', 'लिए', 'लिया', 'ली', 'जाना', 'जाते', 'जाती', 'जाता', 'आना', 'आते', 'आती', 'आता', 'आए', 'आया', 'आई',
        'पाना', 'पाते', 'पाती', 'पाता', 'पाए', 'पाया', 'पाई', 'रहना', 'रहते', 'रहती', 'रहता',
        'एक', 'दो', 'तीन', 'चार', 'पाँच', 'कुल', 'अन्य', 'दूसरा', 'दूसरे', 'दूसरी', 'सभी', 'सब', 'सारे', 'सारी', 'सारा',
        'वाले', 'वाली', 'वाला', 'कहा', 'कहे', 'कही', 'कहना', 'कहते', 'कहती', 'कहने', 'बात',
        'कि', 'यदि', 'आज', 'कल', 'नहीं', 'न', 'ना', 'परंतु', 'लेकिन', 'किंतु', 'मगर', 'क्योंकि', 'चूंकि', 'ताकि',
        'यद्यपि', 'तथापि', 'चाहे', 'मात्र', 'सिर्फ', 'सिर्फ़', 'केवल', 'बस', 'प्रायः', 'प्राय', 'स्वयं', 'खुद',
        'मुझे', 'मुझको', 'मेरा', 'मेरी', 'मेरे', 'हमें', 'हमको', 'हमारा', 'हमारी', 'हमारे',
        'तुम्हें', 'तुमको', 'तुम्हारा', 'तुम्हारी', 'तुम्हारे', 'आपको', 'आपका', 'आपकी', 'आपके',
        'उसे', 'उसको', 'इसका', 'इसकी', 'इसके', 'इन्हें', 'इनको', 'इनका', 'इनकी', 'इनके'
    }

    @classmethod
    def detect_language(cls, text: str) -> Tuple[str, str]:
        """
        Fast and accurate language detection.
        Combines Unicode script distribution and langdetect. Returns (lang_code, lang_name).
        """
        if not text or not text.strip():
            return "en", "English"

        clean = text.strip()[:2000]

        # 1. Script checks for Indic & Asian scripts
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', clean))
        bengali_count = len(re.findall(r'[\u0980-\u09FF]', clean))
        tamil_count = len(re.findall(r'[\u0B80-\u0BFF]', clean))
        telugu_count = len(re.findall(r'[\u0C00-\u0C7F]', clean))
        gujarati_count = len(re.findall(r'[\u0A80-\u0AFF]', clean))
        kannada_count = len(re.findall(r'[\u0C80-\u0CFF]', clean))
        malayalam_count = len(re.findall(r'[\u0D00-\u0D7F]', clean))
        arabic_count = len(re.findall(r'[\u0600-\u06FF]', clean))

        if devanagari_count > 10:
            return "hi", "Hindi"
        if bengali_count > 10:
            return "bn", "Bengali"
        if tamil_count > 10:
            return "ta", "Tamil"
        if telugu_count > 10:
            return "te", "Telugu"
        if gujarati_count > 10:
            return "gu", "Gujarati"
        if kannada_count > 10:
            return "kn", "Kannada"
        if malayalam_count > 10:
            return "ml", "Malayalam"
        if arabic_count > 10:
            return "ur", "Urdu"

        # 2. Pure ASCII check for short text / titles / signatures to prevent langdetect hallucinations
        if clean.isascii() and (len(clean.split()) < 6 or re.search(r'\b(?:the|is|and|of|in|to|with|for|a|an|sign|signature)\b', clean.lower())):
            return "en", "English"

        # 3. General statistical detection with langdetect for multi-sentence passages
        try:
            from langdetect import detect
            code = detect(clean)
            return code, LANGUAGE_NAMES.get(code, code.upper())
        except Exception:
            return "en", "English"

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Splits text into discrete sentences or structured clauses with boundary preservation."""
        if not text:
            return []
        raw_sentences = _RE_SENTENCE_SPLIT.split(text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
        return sentences if sentences else [text.strip()]

    @classmethod
    def tokenize_words(cls, text: str) -> List[str]:
        """Extracts lowercase words of length >= 2 using accurate multilingual Unicode regex."""
        if not text:
            return []
        raw_tokens = _RE_WORDS.findall(text.lower())
        return [t.strip("-–_") for t in raw_tokens if len(t.strip("-–_")) >= 2]

    @classmethod
    def extract_keywords(cls, text: str, top_k: int = 12, precomputed_tokens: List[str] = None) -> List[Dict[str, Any]]:
        """
        Extracts salient domain keyphrases, technical multi-word concepts, and acronyms
        with rapid single-pass clause tokenization and canonical casing across multilingual scripts.
        """
        if not text:
            return []

        stopwords = cls.STOPWORDS
        # Split by punctuation, symbols, and line breaks into discrete candidate clauses
        clauses = re.split(r'[,;.!?()\[\]{}"\'`।॥؛۔«»\n]+', text)
        casing_map: Dict[str, str] = {}
        phrase_counts = Counter()
        multi_word_pool = set()

        for cl in clauses:
            raw_words = _RE_WORDS.findall(cl)
            words = [w.strip("-–_") for w in raw_words if len(w.strip("-–_")) >= 2]
            cur = []
            for w in words:
                w_lower = w.lower()
                if w_lower in stopwords or w.isdigit() or len(w) < 2:
                    if cur:
                        cls._process_keyword_candidate(cur, phrase_counts, casing_map, multi_word_pool, stopwords)
                        cur = []
                else:
                    cur.append(w)
            if cur:
                cls._process_keyword_candidate(cur, phrase_counts, casing_map, multi_word_pool, stopwords)

        if not phrase_counts:
            return []

        # Filter out standalone sub-words if they only exist as fragments of an extracted multi-word phrase
        filtered_items = []
        max_freq = max(phrase_counts.values()) if phrase_counts else 1.0

        for norm_phrase, count in phrase_counts.most_common():
            words_in_p = norm_phrase.split()
            display_name = casing_map.get(norm_phrase, norm_phrase.title() if re.search(r'[a-zA-Z]', norm_phrase) else norm_phrase)

            if len(words_in_p) == 1:
                is_fragment = any(norm_phrase in other_p.split() and norm_phrase != other_p for other_p in multi_word_pool)
                # Keep standalone if it's an acronym, entity or distinct term
                if is_fragment and not (display_name.isupper() or any(c.isdigit() for c in display_name) or len(norm_phrase) <= 3 or display_name in ['Transformer', 'Pegasus', 'BERT', 'BART', 'T5']):
                    continue

            filtered_items.append({
                "keyword": display_name,
                "count": int(count),
                "importance": round(min(1.0, count / max_freq), 2)
            })
            if len(filtered_items) >= top_k:
                break

        return filtered_items

    @classmethod
    def _process_keyword_candidate(cls, words: List[str], counter: Counter, casing_map: Dict[str, str], pool: set, stopwords: set):
        if not words:
            return

        def _format_display(w: str) -> str:
            # If word is in a cased script (Latin / ASCII), apply canonical capitalization
            if re.search(r'[a-zA-Z]', w):
                if w.isupper() or (len(w) <= 3 and any(c.isdigit() for c in w)):
                    return w.upper()
                return w.capitalize()
            # For non-cased scripts (Hindi, Devanagari, Bengali, Tamil, etc.), preserve ligature/word intact
            return w

        # Register full multi-word phrase (up to 3 words)
        if len(words) <= 3:
            norm_full = ' '.join(w.lower() for w in words)
            display_full = ' '.join(_format_display(w) for w in words)
            casing_map[norm_full] = display_full
            counter[norm_full] += 2.0 if len(words) > 1 else 1.0
            if len(words) > 1:
                pool.add(norm_full)

        # Register prominent standalone terms / acronyms / entities
        for w in words:
            nw = w.lower()
            if nw not in stopwords and len(nw) >= 2:
                dw = _format_display(w)
                casing_map[nw] = dw
                if len(words) == 1 or dw.isupper() or any(c.isdigit() for c in dw) or dw in ['Transformer', 'Pegasus', 'BERT', 'BART', 'T5']:
                    counter[nw] += 1.0

    @classmethod
    def extract_key_points(cls, text: str, top_k: int = 4, precomputed_sentences: List[str] = None) -> List[str]:
        """
        Extracts salient, high-diversity bulleted key takeaways and essential declarative clauses
        using TF-IDF term scoring, position biasing, and Maximal Marginal Relevance (MMR) redundancy suppression.
        """
        sentences = precomputed_sentences if precomputed_sentences is not None else cls.split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= top_k:
            return [s.strip() for s in sentences if s.strip()]

        # Precompute sentence tokens and content words
        sent_tokens = [cls.tokenize_words(s) for s in sentences]
        sent_content_words = [
            [w for w in toks if w not in cls.STOPWORDS and not w.isdigit() and len(w) >= 2]
            for toks in sent_tokens
        ]

        all_words = [w for words in sent_content_words for w in words]
        if not all_words:
            return [s.strip() for s in sentences[:top_k]]

        word_counts = Counter(all_words)
        max_wc = max(word_counts.values()) if word_counts else 1.0
        word_weights = {w: math.log(1.0 + (cnt / max_wc)) + 1.0 for w, cnt in word_counts.items()}

        scored_candidates = []
        num_sentences = len(sentences)

        for idx, sentence in enumerate(sentences):
            words = sent_content_words[idx]
            token_count = len(sent_tokens[idx])
            if token_count == 0 or len(words) == 0:
                continue

            # Base keyword coverage score
            kw_score = sum(word_weights.get(w, 0.0) for w in words)
            norm = math.sqrt(len(words)) if len(words) > 0 else 1.0
            base_score = kw_score / norm

            # Position weight: strong intro premise boost and concluding summary boost
            pos_weight = 1.15 if idx == 0 else (1.08 if idx == num_sentences - 1 else 1.0)

            # Rhetorical questions are inquiry prompts rather than conclusive takeaways
            cleaned_s = sentence.strip()
            question_penalty = 0.60 if cleaned_s.endswith('?') or '?' in cleaned_s else 1.0

            # Penalize isolated dangling demonstratives without prior context
            first_words = sent_tokens[idx][:2]
            dangling_penalty = 0.85 if any(fw in {'यह', 'इस', 'ये', 'वे', 'this', 'these', 'it', 'they'} for fw in first_words) and idx > 0 else 1.0

            # Length normalization (favor informative, concise clauses between 8 and 35 words)
            if 8 <= token_count <= 35:
                length_factor = 1.10
            elif token_count > 45:
                length_factor = 0.80
            else:
                length_factor = 0.95

            final_score = base_score * pos_weight * question_penalty * dangling_penalty * length_factor

            # Clean leading bullet markers if present (e.g. •, -, 1.)
            display_sent = re.sub(r'^(?:[•\-\*]|\d+[\.\)])\s*', '', cleaned_s).strip()

            scored_candidates.append({
                "idx": idx,
                "score": final_score,
                "sentence": display_sent,
                "words_set": set(words)
            })

        if not scored_candidates:
            return [s.strip() for s in sentences[:top_k]]

        # Maximal Marginal Relevance (MMR) selection for high information coverage without redundancy
        selected = []
        selected_word_sets = []
        candidates = list(scored_candidates)

        # 1. Select the top-ranked foundation sentence
        candidates.sort(key=lambda x: x["score"], reverse=True)
        first = candidates.pop(0)
        selected.append(first)
        selected_word_sets.append(first["words_set"])

        # 2. Iteratively select subsequent key points balancing topical relevance & lexical diversity
        lambda_param = 0.65  # 65% relevance, 35% diversity penalty

        while len(selected) < top_k and candidates:
            best_mmr_score = -1e9
            best_candidate_idx = 0

            for c_idx, cand in enumerate(candidates):
                cand_set = cand["words_set"]
                max_sim = 0.0
                for sel_set in selected_word_sets:
                    if cand_set and sel_set:
                        sim = len(cand_set.intersection(sel_set)) / len(cand_set.union(sel_set))
                        if sim > max_sim:
                            max_sim = sim

                mmr_score = (lambda_param * cand["score"]) - ((1.0 - lambda_param) * max_sim * cand["score"] * 2.0)
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_idx = c_idx

            picked = candidates.pop(best_candidate_idx)
            selected.append(picked)
            selected_word_sets.append(picked["words_set"])

        # Sort chronologically to preserve logical discourse progression
        selected.sort(key=lambda x: x["idx"])
        return [item["sentence"] for item in selected]

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

    PERSONA_KEYWORDS = {
        "executive": {
            # English
            "revenue", "profit", "loss", "growth", "margin", "cost", "q1", "q2", "q3", "q4",
            "million", "billion", "percent", "%", "$", "increased", "decreased", "roi", "kpi",
            "market", "strategic", "decision", "ebitda", "guidance", "target", "outcome", "earnings", "quarter", "annual",
            "status", "state", "summary", "overview", "impact", "tracking", "account", "accounting", "governance", "unique", "resource", "usage", "identify", "identification", "pid",
            # Hindi (शासन, उद्देश्य, लक्ष्य, नीति, राष्ट्र, समाज, कल्याण, महत्व)
            "लक्ष्य", "उद्देश्य", "नीति", "राष्ट्र", "समाज", "कल्याण", "महत्व", "महत्त्व", "परिणाम", "प्रभाव",
            "शासन", "निर्णय", "साध्य", "विकास", "ऐतिहासिक", "राजनीतिक", "आर्थिक", "व्यवस्था", "प्रमुख", "मुख्य", "सत्य", "मानव"
        },
        "technical": {
            # English
            "api", "architecture", "framework", "database", "pipeline", "model", "transformer",
            "latency", "throughput", "algorithm", "neural", "gpu", "docker", "server", "code",
            "function", "memory", "cpu", "scale", "performance", "deployment", "protocol", "parameter", "embedding", "loss", "accuracy",
            "scheduling", "queue", "queues", "descriptor", "descriptors", "devices", "device", "stack", "heap", "structure", "layout", "network", "allocated", "hardware", "i/o", "io", "pointers",
            # Hindi (भाषा-विज्ञान, नृतत्त्व, शास्त्र, व्याकरण, ग्रंथ, तकनीकी, सिद्धांत)
            "भाषा-विज्ञान", "नृतत्त्व", "नृतत्त्व-शास्त्र", "भाषाशास्त्र", "व्याकरण", "सर्वे", "शोध", "वैज्ञानिक", "शास्त्र",
            "ग्रंथ", "आर्यभाषा", "संस्कृत", "पालि", "प्राकृत", "बोली", "प्रणाली", "प्रक्रिया", "तकनीकी", "संरचना", "सिद्धांत", "तथ्य", "अध्ययन", "विश्लेषण"
        },
        "eli5": {
            # English
            "is", "are", "means", "example", "like", "simple", "main", "works", "help", "called", "known", "way", "idea", "part",
            "number", "shows", "look", "time", "what", "how", "easy", "unique", "files", "using", "open",
            # Hindi (सरल, सहज, उदाहरण, जैसे, अर्थ, सीख, भलाई)
            "सहज", "सरल", "आसान", "जैसे", "उदाहरण", "सीधा", "अर्थात", "अर्थात्", "अर्थ", "सीख", "बात", "कहानी", "सुख", "दुख", "दुःख", "जीवन", "भलाई", "प्रेम"
        },
        "action_items": {
            # English
            "must", "should", "will", "scheduled", "deploy", "implement", "prepare", "fix",
            "review", "coordinate", "deadline", "todo", "action", "task", "deliverable", "assigned", "urgent", "step", "plan", "lock",
            "information", "tracks", "helps", "allocated", "decide", "priority", "scheduling", "descriptors",
            # Hindi (चाहिए, होगा, कर्तव्य, संकल्प, मार्ग, सुधार, प्रयत्न, रक्षा)
            "चाहिए", "होगा", "होगी", "होंगे", "पड़ेगा", "पड़ेगी", "पड़ेंगे", "कर्तव्य", "संकल्प", "व्रत", "उपाय", "रास्ता", "मार्ग", "सुधार", "प्रयत्न", "प्रयास", "कार्य", "सेवा", "बचाना", "लड़ना", "उठाना", "तैयार", "आवश्यक"
        }
    }

    @classmethod
    def compute_attribution(cls, source_text: str, summary_text: str) -> Dict[str, Any]:
        """
        Computes sentence-level semantic attribution mapping from summary sentences to original source sentences.
        Uses TF-IDF, lexical overlap, and n-gram matching to calculate traceability confidence scores.
        """
        source_sentences = cls.split_sentences(source_text)
        summary_sentences = cls.split_sentences(summary_text)

        if not source_sentences or not summary_sentences:
            return {
                "source_sentences": source_sentences,
                "summary_sentences": summary_sentences,
                "attribution_map": []
            }

        # Tokenize source sentences
        src_tokens_list = [cls.tokenize_words(s) for s in source_sentences]
        num_src = len(source_sentences)

        # Compute document frequencies across source sentences for TF-IDF
        df_counts = Counter()
        for toks in src_tokens_list:
            unique_toks = set(toks)
            for t in unique_toks:
                if t not in cls.STOPWORDS:
                    df_counts[t] += 1

        attribution_map = []
        for s_idx, sum_sent in enumerate(summary_sentences):
            sum_toks = cls.tokenize_words(sum_sent)
            if not sum_toks:
                continue

            sum_toks_filtered = [t for t in sum_toks if t not in cls.STOPWORDS]
            sum_set = set(sum_toks_filtered) if sum_toks_filtered else set(sum_toks)
            sum_bigrams = set(zip(sum_toks, sum_toks[1:])) if len(sum_toks) > 1 else set()

            best_src_idx = 0
            best_score = 0.0

            for src_idx, src_sent in enumerate(source_sentences):
                src_toks = src_tokens_list[src_idx]
                if not src_toks:
                    continue

                src_toks_filtered = [t for t in src_toks if t not in cls.STOPWORDS]
                src_set = set(src_toks_filtered) if src_toks_filtered else set(src_toks)

                # 1. Jaccard token overlap
                intersection = sum_set.intersection(src_set)
                union = sum_set.union(src_set)
                jaccard = len(intersection) / len(union) if union else 0.0

                # 2. Bigram overlap
                src_bigrams = set(zip(src_toks, src_toks[1:])) if len(src_toks) > 1 else set()
                bg_intersection = sum_bigrams.intersection(src_bigrams)
                bg_union = sum_bigrams.union(src_bigrams)
                bg_score = len(bg_intersection) / len(bg_union) if bg_union else 0.0

                # 3. Weighted TF-IDF cosine approximation
                tfidf_score = 0.0
                if intersection:
                    tfidf_sum = sum(math.log((num_src + 1) / (df_counts.get(w, 1) + 1)) for w in intersection)
                    tfidf_score = min(1.0, tfidf_sum / (math.sqrt(len(sum_set)) * math.sqrt(len(src_set)) + 1e-5))

                combined = (0.50 * jaccard) + (0.30 * tfidf_score) + (0.20 * bg_score)

                if combined > best_score:
                    best_score = combined
                    best_src_idx = src_idx

            # Calculate normalized percentage confidence (minimum 45% floor for closest match, max 99%)
            conf_pct = round(min(99.0, max(45.0, (best_score * 100) + 20.0)), 1) if best_score > 0.05 else round(min(99.0, best_score * 100), 1)

            attribution_map.append({
                "summary_idx": s_idx,
                "summary_sentence": sum_sent,
                "source_idx": best_src_idx,
                "source_sentence": source_sentences[best_src_idx],
                "confidence": conf_pct,
                "score": round(best_score, 3)
            })

        return {
            "source_sentences": source_sentences,
            "summary_sentences": summary_sentences,
            "attribution_map": attribution_map
        }

    @classmethod
    def extract_named_entities(cls, text: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        High-speed multilingual Named Entity Recognition (NER) heuristics identifying:
        - Organizations / Technologies (e.g., Google, OpenAI, Microsoft, LexiBrief, PyTorch)
        - People / Roles (e.g., Dr. Smith, CEO, Director)
        - Locations / Geographies (e.g., California, India, London, Europe)
        - Dates / Periods (e.g., 2026, Q3 2024, September 17, annual)
        - Metrics & Financial Values (e.g., $12.4 billion, 34%, 12ms, 50MB)
        """
        if not text:
            return []

        entities = []
        seen = set()

        # 1. Metrics, Currencies, Percentages & Numerical stats
        metric_matches = re.finditer(r'(?:[\$\€\£\₹]\s*\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|k|m|b))?|\b\d+(?:\.\d+)?\s*(?:percent|ms|sec|MB|GB|TB|KB|GHz|MHz|ARR|ROI|KPI|parameters|layers|epochs)\b|\b\d+(?:\.\d+)?\s*%)', text, re.IGNORECASE)
        for m in metric_matches:
            val = m.group(0).strip()
            if val.lower() not in seen:
                seen.add(val.lower())
                entities.append({"text": val, "label": "METRIC", "category": "Numbers & Metrics"})

        # 2. Dates, Years, Quarters
        date_matches = re.finditer(r'\b(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?|Q[1-4]\s*(?:\d{4})?|\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)\b', text, re.IGNORECASE)
        for m in date_matches:
            val = m.group(0).strip()
            if val.lower() not in seen:
                seen.add(val.lower())
                entities.append({"text": val, "label": "DATE", "category": "Dates & Timeframes"})

        # 3. Capitalized Multi-word Organizations / Proper Names (for Latin scripts)
        prop_matches = re.finditer(r'\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){1,3})\b', text)
        for m in prop_matches:
            val = m.group(1).strip()
            val_lower = val.lower()
            if val_lower not in seen and not any(w in cls.STOPWORDS for w in val_lower.split()[:1]):
                # Distinguish Person prefix vs Organization
                if re.match(r'^(?:Dr|Prof|Mr|Mrs|Ms|Chief|President|Director|Minister)\b', val):
                    cat = "Person / Role"
                    lbl = "PERSON"
                else:
                    cat = "Organization / Concept"
                    lbl = "ORG"
                seen.add(val_lower)
                entities.append({"text": val, "label": lbl, "category": cat})

        # 4. Known uppercase acronyms & technical organizations (e.g. NASA, WHO, MIT, BERT, GPT, BART)
        acronym_matches = re.finditer(r'\b([A-Z]{2,6})\b', text)
        for m in acronym_matches:
            val = m.group(1).strip()
            if val.lower() not in seen and val.lower() not in cls.STOPWORDS and val not in {'AND', 'THE', 'FOR', 'NOT', 'ALL', 'BUT'}:
                seen.add(val.lower())
                entities.append({"text": val, "label": "ORG", "category": "Organization / Acronym"})

        return entities[:top_k]

    @classmethod
    def extract_important_facts(cls, text: str, top_k: int = 5) -> List[str]:
        """
        Extracts verified factual statements, numerical findings, definitions, and conclusive takeaways.
        Prioritizes declarative sentences containing metrics, quantitative percentages, dates, and causality.
        """
        sentences = cls.split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= top_k:
            return [s.strip() for s in sentences if s.strip()]

        fact_keywords = {
            "increased", "decreased", "grew", "reached", "achieved", "reported", "shows", "proved",
            "found", "is defined as", "refers to", "consists of", "generates", "delivers", "resulting in",
            "percent", "%", "$", "billion", "million", "surged", "dropped", "reduced", "discovered"
        }

        scored = []
        for idx, s in enumerate(sentences):
            s_clean = s.strip()
            if len(s_clean) < 15 or s_clean.endswith('?'):
                continue
            s_lower = s_clean.lower()
            score = 0.0

            # Boost for numerical facts & statistics
            if re.search(r'[\$\€\£\₹%]\s*\d+|\b\d+(?:\.\d+)?\s*(?:%|billion|million|percent|users|ms|x)\b', s_clean):
                score += 3.0
            
            # Boost for factual / result predicate verbs
            for kw in fact_keywords:
                if kw in s_lower:
                    score += 1.5

            # Deduct for speculative / uncertain clauses
            if any(w in s_lower for w in ["might", "could be", "maybe", "perhaps", "possibly", "hypothetically"]):
                score -= 1.0

            if score > 0:
                scored.append((score, idx, s_clean))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top_facts = [item[2] for item in scored[:top_k]]
        return top_facts if top_facts else [s.strip() for s in sentences[:top_k]]

    @classmethod
    def detect_sections(cls, text: str) -> List[Dict[str, Any]]:
        """
        Section-aware structure parser.
        Detects markdown headers (#, ##), presentation slide dividers (--- Slide X ---),
        numbered sections (1. Introduction, Section 2: ...), or ALL-CAPS section titles.
        """
        if not text:
            return []

        lines = text.split('\n')
        sections = []
        current_title = "Overview"
        current_lines = []

        header_pattern = re.compile(r'^(?:#{1,4}\s+(.+)|---\s*(?:Slide\s*\d+|Page\s*\d+)?\s*---|Section\s*\d+[:.-]\s*(.+)|[0-9]+\.\s+([A-Z][A-Za-z\s]{3,40}):?$|([A-Z\s]{4,40}):$)', re.IGNORECASE)

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                if current_lines:
                    current_lines.append("")
                continue

            match = header_pattern.match(trimmed)
            if match:
                title = next((g for g in match.groups() if g), trimmed).strip("#- :")
                if current_lines and "".join(current_lines).strip():
                    sections.append({
                        "title": current_title,
                        "content": "\n".join(current_lines).strip()
                    })
                current_title = title if title else f"Section {len(sections) + 1}"
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines and "".join(current_lines).strip():
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines).strip()
            })

        return sections if sections else [{"title": "Overview", "content": text.strip()}]
