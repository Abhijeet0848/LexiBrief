import re
import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Tuple
from textSummarizer.logging import logger

class TextExtractor:
    """Extracts, cleans, and standardizes text from various input formats (Raw text, PDF, DOCX, TXT)."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans and normalizes extracted text."""
        if not text:
            return ""
        # Normalize whitespace and line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Remove non-printable control characters while preserving valid punctuation
        text = "".join(ch for ch in text if ch.isprintable() or ch in '\n\t')
        return text.strip()

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
                cleaned = re.sub(r'<[^>]+>', ' ', raw_xml)
                return re.sub(r'\s+', ' ', cleaned).strip()
            except Exception:
                return file_bytes.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_from_pdf(file_bytes: bytes) -> str:
        """Extracts text from PDF files using pypdf if available or regex text stream parsing."""
        # Try pypdf if installed
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if pages:
                return "\n\n".join(pages)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pypdf extraction notice: {e}")

        # Stream extraction fallback for PDF text blocks
        try:
            content = file_bytes.decode('latin-1', errors='ignore')
            # Extract plain strings inside parentheses / stream blocks
            text_blocks = re.findall(r'\((.*?)\)\s*T[jJ]', content)
            if text_blocks:
                return " ".join(text_blocks)
            # General fallback
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
            readable = re.findall(r'[A-Za-z0-9\s.,;:\'"?!-]{5,}', cleaned)
            return " ".join(readable)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return file_bytes.decode('utf-8', errors='ignore')

    @classmethod
    def extract(cls, filename: str, file_bytes: bytes) -> Tuple[str, str]:
        """Extracts text based on file extension and returns (clean_text, detected_format)."""
        fn = filename.lower()
        if fn.endswith('.docx') or fn.endswith('.doc'):
            fmt = "DOCX"
            raw = cls.extract_from_docx(file_bytes)
        elif fn.endswith('.pdf'):
            fmt = "PDF"
            raw = cls.extract_from_pdf(file_bytes)
        else:
            fmt = "TXT"
            raw = file_bytes.decode('utf-8', errors='ignore')

        cleaned = cls.clean_text(raw)
        return cleaned, fmt
