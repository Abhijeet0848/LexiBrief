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
        """Extracts text from PDF files using pypdf if available or regex text stream parsing. Returns (text, page_count)."""
        # Try pypdf if installed
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = []
            page_count = len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if pages:
                return "\n\n".join(pages), max(1, page_count)
            return "", max(1, page_count)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pypdf extraction notice: {e}")

        # Stream extraction fallback for PDF text blocks
        try:
            content = file_bytes.decode('latin-1', errors='ignore')
            text_blocks = _RE_PDF_TEXT_BLOCKS.findall(content)
            if text_blocks:
                return " ".join(text_blocks), 1
            cleaned = _RE_NON_PRINTABLE_PDF.sub('', content)
            readable = _RE_READABLE_CHUNKS.findall(cleaned)
            return " ".join(readable), 1
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
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

