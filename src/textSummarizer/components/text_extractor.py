import re
import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Tuple
from textSummarizer.logging import logger

_RE_CRLF = re.compile(r'\r\n')
_RE_SPACES = re.compile(r'[ \t]+')
_RE_MULTILINES = re.compile(r'\n\s*\n+')
_RE_XML_TAGS = re.compile(r'<[^>]+>')
_RE_PDF_TEXT_BLOCKS = re.compile(r'\((.*?)\)\s*T[jJ]')
_RE_NON_PRINTABLE_PDF = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_RE_READABLE_CHUNKS = re.compile(r'[A-Za-z0-9\s.,;:\'"?!-]{5,}')


class TextExtractor:
    """High-speed text extractor, cleaner, and format detector (Raw text, PDF, DOCX, TXT)."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans and normalizes extracted text with compiled regex patterns."""
        if not text:
            return ""
        # Normalize whitespace and line breaks
        text = _RE_CRLF.sub('\n', text)
        text = _RE_SPACES.sub(' ', text)
        text = _RE_MULTILINES.sub('\n\n', text)
        # Remove non-printable control characters while preserving valid punctuation & whitespace
        return "".join(ch for ch in text if ch.isprintable() or ch in '\n\t').strip()

    @staticmethod
    def extract_from_docx(file_bytes: bytes) -> str:
        """Extracts text from DOCX files using standard library zipfile and XML parsing."""
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                # Security: Check uncompressed XML size to prevent Zip Bomb / memory exhaustion
                info = docx_zip.getinfo('word/document.xml')
                if info.file_size > 25 * 1024 * 1024:  # 25 MB max uncompressed XML
                    raise ValueError("DOCX document XML exceeds maximum safe size (25 MB).")
                
                xml_content = docx_zip.read('word/document.xml')
                tree = ET.fromstring(xml_content)
                namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                
                paragraphs = []
                for p in tree.findall('.//w:p', namespaces):
                    texts = [node.text for node in p.findall('.//w:t', namespaces) if node.text]
                    if texts:
                        paragraphs.append("".join(texts))
                return "\n\n".join(paragraphs)
        except Exception as e:
            logger.warning(f"DOCX extraction fallback: {e}")
            # Try raw text extraction from xml as fallback
            try:
                raw_xml = xml_content.decode('utf-8', errors='ignore')
                cleaned = _RE_XML_TAGS.sub(' ', raw_xml)
                return _RE_SPACES.sub(' ', cleaned).strip()
            except Exception:
                return file_bytes.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
        """
        High-fidelity PDF text extraction.
        Uses PyMuPDF (fitz) with automatic OCR fallback for scanned forms and image PDFs,
        with resilient pypdf and raw text stream fallbacks. Returns (text, page_count).
        """
        # 1. State-of-the-art: PyMuPDF with structured layout parsing & OCR fallback
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            pages = []
            page_count = len(doc)
            for idx, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    pages.append(page_text.strip())
                else:
                    # Automatic OCR extraction for scanned forms, application photos, and image-only PDFs
                    try:
                        tp = page.get_textpage_ocr(language="eng", dpi=150)
                        ocr_text = page.get_text(textpage=tp)
                        if ocr_text and ocr_text.strip():
                            pages.append(ocr_text.strip())
                    except Exception:
                        pass
            doc.close()
            if pages:
                return "\n\n".join(pages), max(1, page_count)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PyMuPDF parser notice: {e}")

        # 2. Secondary extractor: pypdf with per-page resilience
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes), strict=False)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            pages = []
            page_count = len(reader.pages) if hasattr(reader, "pages") else 1
            for idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(text.strip())
                except Exception as p_err:
                    logger.debug(f"pypdf page {idx+1} extract notice: {p_err}")
            if pages:
                return "\n\n".join(pages), max(1, page_count)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pypdf extraction error: {e}")

        # 3. Resilient fallback for raw text streams
        try:
            content = file_bytes.decode('latin-1', errors='ignore')
            text_blocks = _RE_PDF_TEXT_BLOCKS.findall(content)
            if text_blocks:
                cleaned_blocks = [b.strip() for b in text_blocks if len(b.strip()) > 1]
                if cleaned_blocks:
                    return " ".join(cleaned_blocks), 1
            cleaned = _RE_NON_PRINTABLE_PDF.sub('', content)
            readable = _RE_READABLE_CHUNKS.findall(cleaned)
            if readable:
                return " ".join(readable), 1
            return file_bytes.decode('utf-8', errors='ignore'), 1
        except Exception as e:
            logger.error(f"PDF stream fallback error: {e}")
            return file_bytes.decode('utf-8', errors='ignore'), 1

    @classmethod
    def extract(cls, filename: str, file_bytes: bytes) -> Tuple[str, str, int]:
        """Extracts text based on file extension and returns (clean_text, detected_format, page_count)."""
        fn = filename.lower()
        pages = 1
        if fn.endswith('.docx') or fn.endswith('.doc'):
            fmt = "DOCX"
            raw = cls.extract_from_docx(file_bytes)
        elif fn.endswith('.pdf'):
            fmt = "PDF"
            raw, pages = cls.extract_from_pdf(file_bytes)
        else:
            fmt = "TXT"
            raw = file_bytes.decode('utf-8', errors='ignore')

        cleaned = cls.clean_text(raw)
        return cleaned, fmt, pages

    @staticmethod
    def extract_youtube_video_id(url: str) -> str:
        """Extracts standard 11-character YouTube video ID from various URL structures."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$'
        ]
        for pattern in patterns:
            match = re.search(pattern, url.strip())
            if match:
                return match.group(1)
        return ""

    @classmethod
    def extract_from_youtube(cls, url: str) -> dict:
        """
        Extracts subtitle transcript and metadata from a YouTube video URL.
        Returns dict with video_id, title, thumbnail_url, text, and timestamped chunks.
        """
        video_id = cls.extract_youtube_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL. Please provide a valid YouTube watch, short, or share link.")

        title = f"YouTube Video ({video_id})"
        author_name = "YouTube Creator"
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        # 1. Fetch metadata via zero-key oEmbed endpoint
        try:
            import requests
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            resp = requests.get(oembed_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", title)
                author_name = data.get("author_name", author_name)
                thumbnail_url = data.get("thumbnail_url", thumbnail_url)
        except Exception as oe_err:
            logger.debug(f"YouTube oEmbed notice: {oe_err}")

        # 2. Fetch transcript via youtube_transcript_api (compatible with v1.2.4+ and older versions)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = None
            
            # Try new v1.2+ API (instance based)
            try:
                ytt = YouTubeTranscriptApi()
                try:
                    t_list = ytt.list(video_id)
                    try:
                        t = t_list.find_transcript(['en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de'])
                        transcript_list = t.fetch()
                    except Exception:
                        for t in t_list:
                            transcript_list = t.fetch()
                            break
                except Exception:
                    transcript_list = ytt.fetch(video_id)
            except Exception:
                pass

            # Try legacy static API if instance didn't work
            if not transcript_list:
                try:
                    if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                        t_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                        try:
                            t = t_obj.find_transcript(['en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de'])
                            transcript_list = t.fetch()
                        except Exception:
                            for t in t_obj:
                                transcript_list = t.fetch()
                                break
                    elif hasattr(YouTubeTranscriptApi, 'get_transcript'):
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                except Exception as leg_err:
                    logger.debug(f"Legacy transcript method notice: {leg_err}")

            if not transcript_list:
                raise ValueError("No subtitles or transcripts available for this YouTube video.")

            full_text_pieces = []
            formatted_chunks = []
            for item in transcript_list:
                snippet = getattr(item, 'text', item.get('text', '') if isinstance(item, dict) else '')
                snippet = str(snippet).replace("\n", " ").strip()
                if not snippet:
                    continue
                full_text_pieces.append(snippet)
                start_raw = getattr(item, 'start', item.get('start', 0) if isinstance(item, dict) else 0)
                start_sec = int(float(start_raw))
                minutes = start_sec // 60
                seconds = start_sec % 60
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                formatted_chunks.append({
                    "time": timestamp_str,
                    "seconds": start_sec,
                    "text": snippet
                })

            full_text = cls.clean_text(" ".join(full_text_pieces))
            word_count = len(full_text.split())
            return {
                "video_id": video_id,
                "title": title,
                "author": author_name,
                "thumbnail": thumbnail_url,
                "text": full_text,
                "words": word_count,
                "chunks": formatted_chunks,
                "format": "YOUTUBE",
                "source_url": f"https://www.youtube.com/watch?v={video_id}"
            }
        except Exception as yt_err:
            logger.error(f"YouTube transcript extraction error: {yt_err}")
            raise ValueError(f"Could not retrieve transcript from YouTube video: {str(yt_err)}")

    @classmethod
    def extract_from_url(cls, url: str) -> dict:
        """
        Scrapes and extracts clean main article text and title from a web URL.
        Removes navigation, scripts, ads, and footers.
        """
        import requests
        from bs4 import BeautifulSoup

        clean_url = url.strip()
        if not clean_url.startswith(('http://', 'https://')):
            clean_url = 'https://' + clean_url

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }

        try:
            resp = requests.get(clean_url, headers=headers, timeout=12)
            resp.raise_for_status()
        except Exception as req_err:
            raise ValueError(f"Failed to fetch content from URL: {str(req_err)}")

        soup = BeautifulSoup(resp.content, 'html.parser')

        # Extract title
        page_title = "Web Article"
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()
        elif soup.find('h1'):
            page_title = soup.find('h1').get_text().strip()

        # Remove irrelevant elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'svg', 'noscript', 'iframe']):
            tag.decompose()

        # Extract main text content
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|article|post|body|entry', re.I))
        if main_content:
            paragraphs = [p.get_text().strip() for p in main_content.find_all(['p', 'h2', 'h3', 'li']) if len(p.get_text().strip()) > 20]
        else:
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 20]

        extracted_text = "\n\n".join(paragraphs) if paragraphs else soup.get_text(separator='\n')
        cleaned_text = cls.clean_text(extracted_text)

        if not cleaned_text or len(cleaned_text) < 50:
            raise ValueError("Could not extract substantial article text from the provided webpage.")

        return {
            "title": page_title,
            "text": cleaned_text,
            "format": "WEB_URL",
            "source_url": clean_url
        }

