import re
import requests
from typing import Tuple, List
from textSummarizer.logging import logger


class Translator:
    """
    High-Speed Free Neural Translation Engine (Zero API Key Required).
    Translates non-English multilingual text, articles, and summaries into clean English.
    """

    @classmethod
    def translate_to_english(cls, text: str, source_lang: str = "auto") -> Tuple[str, str]:
        """
        Translates text to English with automatic multi-tier fallback.
        Returns (translated_english_text, detected_source_language_code).
        """
        if not text or not text.strip():
            return "", "en"

        cleaned_text = text.strip()
        
        # Check if text is already primarily English/ASCII with minimal non-Latin
        if source_lang == "en":
            return cleaned_text, "en"

        # Split large passages into manageable paragraphs / sentences to avoid URL query limit
        chunks = cls._chunk_text(cleaned_text, max_chars=1800)
        translated_chunks = []
        detected_lang = source_lang if source_lang != "auto" else "unknown"

        for chunk in chunks:
            chunk_trans, chunk_lang = cls._translate_single_chunk(chunk, source_lang=source_lang)
            translated_chunks.append(chunk_trans)
            if detected_lang == "unknown" and chunk_lang and chunk_lang != "unknown":
                detected_lang = chunk_lang

        result_text = " ".join(translated_chunks).strip()
        result_text = re.sub(r'\s+([.,!?;:])', r'\1', result_text)
        result_text = re.sub(r'[ \t]+', ' ', result_text)

        return result_text if result_text else cleaned_text, detected_lang

    @classmethod
    def _translate_single_chunk(cls, chunk: str, source_lang: str = "auto") -> Tuple[str, str]:
        """Translates a single chunk up to 2000 characters using tiered free neural endpoints."""
        if not chunk or not chunk.strip():
            return "", source_lang

        # Tier 1: Google Neural Translation API (Free Client5 Endpoint)
        try:
            url = "https://clients5.google.com/translate_a/t"
            params = {
                "client": "dict-chrome-ex",
                "sl": source_lang,
                "tl": "en",
                "q": chunk
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) > 0:
                        trans_text = data[0][0]
                        detected = data[0][1] if len(data[0]) > 1 else source_lang
                        if trans_text and isinstance(trans_text, str):
                            return trans_text.strip(), detected
                    elif isinstance(data[0], str):
                        return data[0].strip(), source_lang
        except Exception as e:
            logger.debug(f"Tier 1 translation notice: {e}")

        # Tier 2: MyMemory Free Translation API Fallback
        try:
            url = "https://api.mymemory.translated.net/get"
            langpair = f"{source_lang}|en" if source_lang != "auto" else "autodetect|en"
            params = {
                "q": chunk,
                "langpair": langpair
            }
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                trans_text = data.get("responseData", {}).get("translatedText")
                detected = data.get("responseData", {}).get("detectedSourceLanguage", source_lang)
                if trans_text and "MYMEMORY WARNING" not in trans_text.upper():
                    return trans_text.strip(), detected
        except Exception as e:
            logger.debug(f"Tier 2 translation notice: {e}")

        # Tier 3: Return original chunk if all translation endpoints are unreachable
        return chunk, source_lang

    @classmethod
    def _chunk_text(cls, text: str, max_chars: int = 1800) -> List[str]:
        """Splits long text into sentence/paragraph-respecting chunks."""
        if len(text) <= max_chars:
            return [text]

        paragraphs = text.split('\n')
        chunks = []
        current = []
        current_len = 0

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue
            if current_len + len(p_strip) > max_chars and current:
                chunks.append("\n".join(current))
                current = [p_strip]
                current_len = len(p_strip)
            else:
                current.append(p_strip)
                current_len += len(p_strip) + 1

        if current:
            chunks.append("\n".join(current))

        # Further split any oversized paragraph by sentence boundaries
        final_chunks = []
        for c in chunks:
            if len(c) > max_chars:
                sentences = re.split(r'([।!?.\n]+)', c)
                sub_c = ""
                for s in sentences:
                    if len(sub_c) + len(s) > max_chars and sub_c:
                        final_chunks.append(sub_c.strip())
                        sub_c = s
                    else:
                        sub_c += s
                if sub_c.strip():
                    final_chunks.append(sub_c.strip())
            else:
                final_chunks.append(c)

        return [fc for fc in final_chunks if fc.strip()]
