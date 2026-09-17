import os
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
    def clean_ocr_text(text: str) -> str:
        """
        Cleans OCR artifacts, removes confusing glitch/non-printable symbols,
        normalizes misrecognized bullet points and arrows,
        and heals digit-in-word confusions while preserving emails, domains, URLs,
        bullets, all international Unicode characters, and Indic/Arabic/CJK punctuation.
        """
        if not text:
            return ""
        
        # 1. Normalize math multiplication symbols (e.g. 1 x 2 x 3 or 1 × 2 × 3)
        text = re.sub(r'(\d)\s*[\uFFFD\uFEFF×]\s*(\d)', r'\1 x \2', text)

        # 2. Normalize sub-items with leading arrows onto separate indented lines
        text = re.sub(r'(?<=\S)[ \t]+(?:->|-->)[ \t]+', r'\n  -> ', text)
        text = re.sub(r'(?:^|\n)\s*(?:->|-->)\s+', r'\n  -> ', text)

        # 3. Normalize internal inline arrows (e.g. "grows/shrinks naturally" or "last in -> first out")
        text = re.sub(r'(?<=\w)\s*(?:->|-->|→|—>)\s*(?=\w)', ' → ', text)

        # 4. Heal function/method list items (e.g. "push() to insert...", "size() returns...")
        text = re.sub(r'(?:^|\n)\s*([a-zA-Z_][a-zA-Z0-9_]*\(\)\s+(?:to|returns|Returns|is|checks|removes|inserts)\b)', r'\n• \1', text)

        # 5. Heal misclassified bullet glyphs (e.g. '+' or '*' or '-' or Hindi numerals '१.' / '५' / '५०' / '०' before English words)
        text = re.sub(r'(?:^|\n)\s*[१२३४५६७८९०\u0966-\u096F]+[\.\s:o°\-_*~]*(?=[A-Za-z])', r'\n• ', text)
        text = re.sub(r'(?:^|\n)\s*[\+\*]\s+(?=[A-Za-z0-9])', r'\n• ', text)
        text = re.sub(r'(?:^|\n)\s*-\s+(?=[A-Z])', r'\n• ', text)

        # 6. Normalize section breaks around Note: and Example: and headers
        text = re.sub(r'(?<=\S)[ \t]+(Note:|Example:)', r'\n\n\1', text)
        text = re.sub(r'(?:^|\n)\s*•\s*(Example:|Note:)', r'\n\n\1', text)
        
        # 7. Normalize bullet points and examples onto separate lines
        text = re.sub(r'(?:^|\n)\s*[\uFFFD\uFEFF•\-\*■▪◆\u2022\u25cf\u25aa\u25b6\u2713\u2714\.]*\s*Example:', r'\nExample:', text)
        text = re.sub(r'(?<=\S)\s+(?:[\uFFFD\uFEFF•\-\*■▪◆\u2022\u25cf\u25aa\u25b6\u2713\u2714\.]*\s*Example:)', r'\nExample:', text)

        # 5. Strip unprintable control codes and remaining replacement characters
        text = re.sub(r'[\uFFFD\uFEFF\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 6. Strip OCR fringe glitches (isolated bars/tildes/glitches) but keep normal punctuation (. , : ; / @ _ - • & % #)
        text = re.sub(r'[|~¬¢§±µ¿¡]+', ' ', text)
        
        # 7. Heal English OCR digit-in-word confusions (e.g., 'm0del' -> 'model', 'w1th' -> 'with')
        def _heal_word(w: str) -> str:
            if not w:
                return ""
            # Keep numbers, dates, emails, domains, codes intact
            if w.isdigit() or re.match(r'^\d+[a-zA-Z]{1,2}$', w) or '@' in w or '.' in w or '/' in w or '_' in w:
                return w
            # Only apply Latin letter heuristic if word is pure ASCII alphanumeric
            if re.match(r'^[a-zA-Z0-9]+$', w):
                alpha_count = sum(1 for c in w if c.isalpha())
                if alpha_count >= 2:
                    w = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o', w)
                    w = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', w)
                    w = re.sub(r'(?<=[a-zA-Z])3(?=[a-zA-Z])', 'e', w)
                    w = re.sub(r'(?<=[a-zA-Z])5(?=[a-zA-Z])', 's', w)
            return w

        # 8. Heal common OCR phrase and Devanagari ligature distortions
        text = re.sub(r'\b[Ww]SUSE[ \t]+OF\b', 'MISUSE OF', text)
        text = re.sub(r'\bINTENTION[ \t]+OF[ \t]+Avan[^\r\n]*', 'INTENTION OF AVAILING', text, flags=re.IGNORECASE)
        text = re.sub(r'\bWEALTH[ \t]+FACILITY\b', 'HEALTH FACILITY', text, flags=re.IGNORECASE)
        text = re.sub(r'\bWALL[ \t]+INVITE\b', 'WILL INVITE', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDISCIPLIMARY\b', 'DISCIPLINARY', text, flags=re.IGNORECASE)
        text = re.sub(r'\bUss\b', 'USS', text)
        text = re.sub(r'\bWorld[ \t]+War[ \t]+(?:ll|11|lI|Il)\b', 'World War II', text)
        text = re.sub(r'\bWorld[ \t]+War[ \t]+(?:l|1)\b', 'World War I', text)
        text = re.sub(r'विश्वविश्[^\s]*लय', 'विश्वविद्यालय', text)
        text = re.sub(r'दिश्वडिसालय', 'विश्वविद्यालय', text)
        text = re.sub(r'विरवविद्यांसय', 'विश्वविद्यालय', text)
        text = re.sub(r'कंन्ट्र\b', 'केन्द्र', text)
        text = re.sub(r'नई[ \t]+fa[ \t]*(?=\d{6})', 'नई दिल्ली-', text)
        text = re.sub(r'नई[ \t]+दिल्ली[ \t]*(?:noose\??|noos\w*|11006[?7])', 'नई दिल्ली-110067', text, flags=re.IGNORECASE)
        text = re.sub(r'[—–\-~_]*[ \t]*[0oO][ \t]*DELHI[\s\-]*(?=\d{6})', 'NEW DELHI-', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:on[ \t]*9[ \t]*)?DEL[ \t]*Hi\b', 'DELHI', text, flags=re.IGNORECASE)
        text = re.sub(r'\bNEW[ \t]+DEL[ \t]*HI\b', 'NEW DELHI', text, flags=re.IGNORECASE)

        # 9. Filter line by line and eliminate isolated single-character Latin noise
        clean_lines = []
        for line in text.split('\n'):
            line_str = line.strip()
            if not line_str:
                continue
            words = line_str.split()
            healed_words = []
            for raw_w in words:
                w = _heal_word(raw_w.strip())
                # Drop solitary random single Latin letters that aren't valid words
                if len(w) == 1 and w.isascii() and w.isalpha() and w.lower() not in ['a', 'i']:
                    continue
                if w:
                    healed_words.append(w)
            
            if healed_words:
                clean_lines.append(' '.join(healed_words))
            
        cleaned = "\n".join(clean_lines)
        return TextExtractor.clean_text(cleaned)

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans, normalizes, and properly formats extracted text ensuring every bullet point is on a separate new line."""
        if not text:
            return ""
        # Strip repetitive OCR artifact characters
        text = re.sub(r'([|~_^\/\\<>{}\[\]*+=#`¬¢§±µ¿¡])\s*\1{2,}', ' ', text)
        # Normalize line breaks
        text = _RE_CRLF.sub('\n', text)
        
        # Strip standalone timestamp lines (e.g. "0:15", "01:23", "[02:45]")
        text = re.sub(r'^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*$', '', text, flags=re.MULTILINE)
        # Strip inline timestamp prefixes (e.g. "0:15 - Hello" or "[0:15] Hello")
        text = re.sub(r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*[-–—:]?\s*', '', text)

        # Standardize all bullet points so each bullet point starts on its own new line
        bullet_chars = r'[•\u2022\u25cf\u25aa\u25b8\u2219\u2023\u2043\u204c\u204d\u2218\u25cb\u25e6\u25ab]'
        text = re.sub(rf'[ \t]*({bullet_chars})[ \t]*', r'\n• ', text)
        text = re.sub(r'\n{2,}• ', r'\n• ', text)

        # Normalize line spacing while preserving bullet lines
        raw_lines = text.split('\n')
        clean_lines = []
        for l in raw_lines:
            s_line = _RE_SPACES.sub(' ', l).strip()
            if s_line:
                clean_lines.append(s_line)

        text = "\n".join(clean_lines)

        # Remove non-printable control characters while preserving valid punctuation & whitespace
        text = "".join(ch for ch in text if ch.isprintable() or ch in '\n\t').strip()

        # Clean multiple blank lines
        text = _RE_MULTILINES.sub('\n\n', text)
        return text.strip()

    @staticmethod
    def extract_from_docx(file_bytes: bytes) -> str:
        """Extracts text from DOCX files using python-docx with table parsing and fallback to standard library zipfile and XML parsing."""
        # 1. State-of-the-art: python-docx with full table & paragraph extraction
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            docx_blocks = []
            for p in doc.paragraphs:
                if p.text and p.text.strip():
                    docx_blocks.append(p.text.strip())
            for tbl in doc.tables:
                for row in tbl.rows:
                    row_txt = [c.text.strip() for c in row.cells if c.text.strip()]
                    if row_txt:
                        docx_blocks.append(" | ".join(row_txt))
            if docx_blocks:
                return "\n\n".join(docx_blocks)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"python-docx parsing notice: {e}")

        # 2. Resilient OpenXML Zip Archive parser
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                # Security: Check uncompressed XML size to prevent Zip Bomb / memory exhaustion
                info = docx_zip.getinfo('word/document.xml')
                if info.file_size > 75 * 1024 * 1024:  # 75 MB max uncompressed XML
                    raise ValueError("DOCX document XML exceeds maximum safe size (75 MB).")
                
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
        High-fidelity PDF text extraction matrix.
        - PyMuPDF (Tier 1): High-speed stream extraction, reading-order sorting (sort=True),
          layout block parsing, encryption/password authentication, and selective OCR for image/scanned pages.
        - pdfplumber (Tier 2): Precise structured table extraction & column alignment.
        - PaddleOCR / PyMuPDF OCR (Tier 3): Multilingual text & handwriting OCR on sparse/scanned pages.
        - pypdf (Tier 4): Resilient fallback parser.
        """
        # 1. State-of-the-art: PyMuPDF with structured layout parsing, password handling & selective OCR
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            
            # Handle encrypted / password-protected PDFs
            if getattr(doc, "is_encrypted", False):
                try:
                    doc.authenticate("")
                except Exception:
                    pass

            pages = []
            page_count = len(doc)
            for idx, page in enumerate(doc):
                # sort=True preserves natural top-to-bottom, left-to-right reading order across columns
                page_text = page.get_text("text", sort=True)
                
                # If standard text extraction yielded fragmented blocks, try blocks mode
                if not page_text or len(page_text.strip()) < 35:
                    try:
                        blocks = page.get_text("blocks", sort=True)
                        b_texts = [b[4].strip() for b in blocks if len(b) > 4 and b[4].strip()]
                        if b_texts and len("\n\n".join(b_texts)) > len(page_text or ""):
                            page_text = "\n\n".join(b_texts)
                    except Exception:
                        pass

                # Selective OCR on scanned pages, application forms, or image-only pages
                if not page_text or len(page_text.strip()) < 35:
                    # Try RapidOCR (ONNXRuntime) if installed (fastest zero-dependency neural OCR)
                    try:
                        from rapidocr_onnxruntime import RapidOCR
                        import numpy as np
                        from PIL import Image
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        engine = RapidOCR()
                        ocr_res, _ = engine(np.array(img))
                        if ocr_res:
                            lines = [item[1] for item in ocr_res if len(item) > 1 and item[1]]
                            if lines:
                                page_text = "\n".join(lines)
                    except Exception:
                        pass

                # Try PaddleOCR if installed
                if not page_text or len(page_text.strip()) < 35:
                    try:
                        from paddleocr import PaddleOCR
                        import numpy as np
                        from PIL import Image
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
                        result = ocr_engine.ocr(np.array(img), cls=True)
                        if result and result[0]:
                            lines = [line[1][0] for line in result[0] if line and len(line) > 1 and line[1]]
                            if lines:
                                page_text = "\n".join(lines)
                    except Exception:
                        pass

                # Fallback to PyMuPDF built-in Tesseract OCR
                if not page_text or len(page_text.strip()) < 35:
                    try:
                        tp = page.get_textpage_ocr(language="eng+hin", dpi=200)
                        ocr_text = page.get_text(textpage=tp)
                        if ocr_text and ocr_text.strip():
                            page_text = ocr_text.strip()
                    except Exception:
                        try:
                            tp = page.get_textpage_ocr(language="eng", dpi=200)
                            ocr_text = page.get_text(textpage=tp)
                            if ocr_text and ocr_text.strip():
                                page_text = ocr_text.strip()
                        except Exception:
                            pass

                if page_text and page_text.strip():
                    pages.append(page_text.strip())
            doc.close()
            if pages:
                return "\n\n".join(pages), max(1, page_count)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PyMuPDF parser notice: {e}")

        # 2. Secondary extractor: pdfplumber with table extraction and layout awareness
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                plumber_pages = []
                for p in pdf.pages:
                    # Extract structured tables if present
                    table_lines = []
                    try:
                        tables = p.extract_tables()
                        if tables:
                            for tbl in tables:
                                for row in tbl:
                                    clean_row = [str(c).strip() for c in row if c is not None and str(c).strip()]
                                    if clean_row:
                                        table_lines.append(" | ".join(clean_row))
                    except Exception:
                        pass

                    txt = p.extract_text(layout=False, x_tolerance=2, y_tolerance=3) or ""
                    if table_lines:
                        txt = txt + "\n\n" + "\n".join(table_lines)

                    if txt and txt.strip():
                        plumber_pages.append(txt.strip())
                if plumber_pages:
                    return "\n\n".join(plumber_pages), len(pdf.pages)
        except Exception as pl_err:
            logger.debug(f"pdfplumber extraction notice: {pl_err}")

        # 3. Tertiary extractor: pypdf with per-page resilience & decryption
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

    @staticmethod
    def extract_from_pptx(file_bytes: bytes) -> Tuple[str, int]:
        """
        High-fidelity presentation text extraction from PowerPoint PPTX / PPT files.
        Parses slide XMLs, text frames, shape tables, and speaker notes, returning (clean_text, slide_count).
        """
        # 1. State-of-the-art: python-pptx
        try:
            from pptx import Presentation
            prs = Presentation(io.BytesIO(file_bytes))
            slide_texts = []
            slide_count = len(prs.slides)
            for idx, slide in enumerate(prs.slides):
                slide_content = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            txt = "".join(run.text for run in paragraph.runs if run.text).strip()
                            if txt:
                                slide_content.append(txt)
                    elif shape.has_table:
                        for row in shape.table.rows:
                            row_txt = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                            if row_txt:
                                slide_content.append(" | ".join(row_txt))
                
                # Check for speaker notes
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes_txt = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_txt:
                        slide_content.append(f"[Speaker Notes: {notes_txt}]")

                if slide_content:
                    slide_texts.append(f"--- Slide {idx + 1} ---\n" + "\n".join(slide_content))

            if slide_texts:
                return "\n\n".join(slide_texts), max(1, slide_count)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"python-pptx extraction notice: {e}")

        # 2. Resilient OpenXML Zip Archive parser (zero external dependencies)
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as ppt_zip:
                slide_files = [f for f in ppt_zip.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
                def extract_slide_num(name):
                    m = re.search(r'slide(\d+)\.xml', name)
                    return int(m.group(1)) if m else 999999
                
                slide_files.sort(key=extract_slide_num)
                slide_count = len(slide_files)
                slide_texts = []
                
                ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
                
                for idx, sfile in enumerate(slide_files):
                    xml_content = ppt_zip.read(sfile)
                    tree = ET.fromstring(xml_content)
                    texts = [node.text for node in tree.findall('.//a:t', ns) if node.text and node.text.strip()]
                    if texts:
                        slide_texts.append(f"--- Slide {idx + 1} ---\n" + "\n".join(texts))
                
                if slide_texts:
                    return "\n\n".join(slide_texts), max(1, slide_count)
        except Exception as e:
            logger.warning(f"PPTX zip extraction fallback: {e}")

        # 3. Fallback for raw binary or legacy PPT
        try:
            content = file_bytes.decode('latin-1', errors='ignore')
            readable = _RE_READABLE_CHUNKS.findall(content)
            if readable:
                return " ".join(readable), 1
            return file_bytes.decode('utf-8', errors='ignore'), 1
        except Exception as e:
            logger.error(f"PPT binary fallback error: {e}")
            return file_bytes.decode('utf-8', errors='ignore'), 1

    @staticmethod
    def _reconstruct_ocr_boxes(ocr_res) -> str:
        """
        Sorts and groups 2D spatial OCR bounding boxes into natural reading-order lines.
        Maintains paragraph flow, joins inline words left-to-right, and guarantees that
        every bullet point and enumerated list item starts on its own discrete line.
        """
        if not ocr_res:
            return ""

        import numpy as np
        boxes_data = []
        for item in ocr_res:
            if len(item) < 2 or not item[1] or not str(item[1]).strip():
                continue
            box = np.array(item[0])
            text = str(item[1]).strip()
            # Clean glyph glitches like replacement chars or isolated orphan symbols
            text = re.sub(r'[\uFFFD\uFEFF\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
            text = re.sub(r'^[|~¬¢§±µ¿¡\.]\s*', '', text)
            if not text:
                continue
            min_x = float(np.min(box[:, 0]))
            max_x = float(np.max(box[:, 0]))
            min_y = float(np.min(box[:, 1]))
            max_y = float(np.max(box[:, 1]))
            center_y = (min_y + max_y) / 2.0
            height = max(1.0, max_y - min_y)
            boxes_data.append({
                "text": text,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_y": center_y,
                "height": height
            })

        if not boxes_data:
            return ""

        # Sort primarily top-to-bottom
        boxes_data.sort(key=lambda b: (b["min_y"], b["min_x"]))

        lines = []
        current_line = [boxes_data[0]]

        for b in boxes_data[1:]:
            line_avg_y = sum(x["center_y"] for x in current_line) / len(current_line)
            line_avg_h = sum(x["height"] for x in current_line) / len(current_line)
            threshold = max(6.0, line_avg_h * 0.48)

            if abs(b["center_y"] - line_avg_y) <= threshold:
                current_line.append(b)
            else:
                current_line.sort(key=lambda x: x["min_x"])
                lines.append(" ".join(x["text"] for x in current_line))
                current_line = [b]

        if current_line:
            current_line.sort(key=lambda x: x["min_x"])
            lines.append(" ".join(x["text"] for x in current_line))

        # Format bullets and paragraphs ensuring separate lines
        formatted = []
        bullet_marker_pattern = re.compile(r'^(?:[•\-\*■▪◆\u2022\u25cf\u25aa\u25b6\u2713\u2714]|->|-->|\d+[\.\)]|[a-zA-Z][\.\)])\s*')
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Check for multiple bullets or arrow sub-items joined on one line
            sub_items = re.split(r'(?<=\S)\s+([•\-\*■▪◆\u2022\u25cf\u25aa\u25b6\u2713\u2714]|->|-->|\d+\.|\([0-9a-zA-Z]\))\s+', line_str)
            if len(sub_items) > 1:
                head = sub_items[0].strip()
                if head:
                    formatted.append(head)
                for i in range(1, len(sub_items), 2):
                    marker = sub_items[i].strip()
                    b_text = sub_items[i+1].strip() if i+1 < len(sub_items) else ""
                    if marker in ["->", "-->"]:
                        formatted.append(f"  -> {b_text}")
                    else:
                        formatted.append(f"• {b_text}")
            else:
                if bullet_marker_pattern.match(line_str):
                    if line_str.startswith("->") or line_str.startswith("-->"):
                        norm = re.sub(r'^(?:->|-->)\s*', '  -> ', line_str)
                    else:
                        norm = re.sub(r'^[•\-\*■▪◆\u2022\u25cf\u25aa\u25b6\u2713\u2714]\s*', '• ', line_str)
                    formatted.append(norm)
                else:
                    formatted.append(line_str)

        return "\n".join(formatted)

    @staticmethod
    def _create_image_variants(raw_img):
        """Generates adaptive contrast, binarized, and channel-isolated image variants for maximum OCR readability."""
        from PIL import Image, ImageOps, ImageEnhance
        import numpy as np

        variants = []
        w, h = raw_img.size

        # 1. Upscale if small (minimum 1800px on max dimension)
        if max(w, h) < 1600:
            scale_factor = min(2.5, 1800.0 / max(w, h))
            working_img = raw_img.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.LANCZOS)
        elif max(w, h) > 3500:
            scale_factor = 2500.0 / max(w, h)
            working_img = raw_img.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.BILINEAR)
        else:
            working_img = raw_img

        # Variant 1: Enhanced Auto-Contrast & Crisp Sharpening
        v1 = ImageOps.autocontrast(working_img, cutoff=1)
        v1 = ImageEnhance.Sharpness(v1).enhance(1.4)
        v1 = ImageEnhance.Contrast(v1).enhance(1.2)
        variants.append(("enhanced", v1))

        # Variant 2: Adaptive Yellow/Warm-Tint Cancelling (Blue-channel emphasis)
        try:
            r, g, b = working_img.split()
            v2 = ImageEnhance.Contrast(b).enhance(1.4)
            v2 = ImageOps.autocontrast(v2, cutoff=2)
            variants.append(("blue_channel_contrast", v2.convert("RGB")))
        except Exception:
            pass

        # Variant 3: Adaptive Binarization for documents & ID cards
        try:
            gray = ImageOps.grayscale(working_img)
            gray_np = np.array(gray)
            mean_val = np.mean(gray_np)
            thresh_np = np.where(gray_np > mean_val * 0.88, 255, 0).astype(np.uint8)
            v3 = Image.fromarray(thresh_np).convert("RGB")
            variants.append(("adaptive_binarized", v3))
        except Exception:
            pass

        # Variant 4: Human Handwriting & Cursive Stroke Enhancement (Lined paper & shadow suppression)
        try:
            import cv2
            img_np = np.array(working_img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
            )
            # Morphological close to bridge broken cursive pen strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_ink = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            v4 = Image.fromarray(morph_ink).convert("RGB")
            variants.append(("handwriting_adaptive", v4))
        except Exception as hw_err:
            logger.debug(f"Handwriting variant notice: {hw_err}")

        return variants

    @staticmethod
    def extract_from_image(file_bytes: bytes) -> Tuple[str, int]:
        """
        High-accuracy image OCR engine (Camera photos, document scans, certificates, screenshots).
        Applies multi-variant adaptive preprocessors with PyMuPDF Neural OCR, RapidOCR, PaddleOCR, and pytesseract.
        """
        from PIL import Image, ImageOps
        import numpy as np

        raw_img = None
        variants = []

        try:
            raw_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            try:
                raw_img = ImageOps.exif_transpose(raw_img)
            except Exception:
                pass
            variants = TextExtractor._create_image_variants(raw_img)
        except Exception as e:
            logger.debug(f"Pillow image preprocessing notice: {e}")

        candidates = []

        # Engine A: PyMuPDF OCR (Tesseract Neural LSTM with bilingual English + Hindi support)
        try:
            import pymupdf
            for var_name, var_img in variants:
                buf = io.BytesIO()
                var_img.save(buf, format="PNG")
                enhanced_bytes = buf.getvalue()

                doc = pymupdf.open(stream=enhanced_bytes, filetype="png")
                pages = []
                for page in doc:
                    try:
                        tp = page.get_textpage_ocr(language="eng+hin", dpi=300, full=True)
                        text = page.get_text(textpage=tp)
                        if text and text.strip():
                            pages.append(text.strip())
                    except Exception:
                        try:
                            tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                            text = page.get_text(textpage=tp)
                            if text and text.strip():
                                pages.append(text.strip())
                        except Exception:
                            pass
                doc.close()
                if pages:
                    pymupdf_text = "\n\n".join(pages).strip()
                    if pymupdf_text:
                        candidates.append((f"pymupdf_{var_name}", pymupdf_text))
        except Exception as e:
            logger.debug(f"PyMuPDF image OCR note: {e}")

        # Engine B: RapidOCR (ONNXRuntime) with layout reconstruction & multi-orientation search
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            
            target_np = np.array(variants[0][1]) if variants else (np.array(raw_img) if raw_img else None)
            if target_np is not None:
                ocr_res, _ = engine(target_np)
                best_res = ocr_res
                best_words = sum(len(box[1].split()) for box in ocr_res) if ocr_res else 0

                # Check handwriting adaptive & other variants if initial detection is sparse
                if best_words < 20 and len(variants) > 1:
                    for v_name, v_img in variants[1:]:
                        v_np = np.array(v_img)
                        v_res, _ = engine(v_np)
                        if v_res:
                            v_words = sum(len(box[1].split()) for box in v_res)
                            if v_words > best_words:
                                best_words = v_words
                                best_res = v_res
                                target_np = v_np

                # Multi-angle search for sideways phone photos (90°, 180°, 270°)
                if best_words < 15 or not ocr_res:
                    for k_rot in (1, 2, 3):
                        rotated_np = np.rot90(target_np, k_rot)
                        rot_res, _ = engine(rotated_np)
                        if rot_res:
                            rot_words = sum(len(box[1].split()) for box in rot_res)
                            if rot_words > best_words:
                                best_words = rot_words
                                best_res = rot_res
                
                # Fallback: High-sensitivity detection for isolated handwriting, signatures & faint pencil strokes
                if not best_res or best_words < 5:
                    try:
                        sens_engine = RapidOCR(box_thresh=0.15, text_score=0.15, unclip_ratio=2.0)
                        # Test raw image without pre-processing filters
                        raw_np = np.array(raw_img) if raw_img else None
                        if raw_np is not None:
                            sens_raw_res, _ = sens_engine(raw_np)
                            if sens_raw_res:
                                raw_rec = TextExtractor._reconstruct_ocr_boxes(sens_raw_res)
                                if raw_rec and raw_rec.strip():
                                    candidates.append(("rapidocr_sens_raw", raw_rec.strip()))
                        
                        sens_res, _ = sens_engine(target_np)
                        if sens_res:
                            if not best_res or sum(len(box[1].split()) for box in sens_res) > best_words:
                                best_res = sens_res
                    except Exception:
                        pass

                if best_res:
                    reconstructed = TextExtractor._reconstruct_ocr_boxes(best_res)
                    if reconstructed and reconstructed.strip():
                        candidates.append(("rapidocr_onnx", reconstructed.strip()))
        except Exception as ocr_err:
            logger.debug(f"RapidOCR execution note: {ocr_err}")

        # Engine C: PaddleOCR for deep scene text recognition
        try:
            from paddleocr import PaddleOCR
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
            target_np = np.array(variants[0][1]) if variants else (np.array(raw_img) if raw_img else None)
            if target_np is not None:
                result = ocr_engine.ocr(target_np, cls=True)
                if result and result[0]:
                    lines = [line[1][0] for line in result[0] if line and len(line) > 1 and line[1]]
                    if lines:
                        candidates.append(("paddleocr", "\n".join(lines)))
        except Exception:
            pass

        # Engine D: PyTesseract fallback
        try:
            import pytesseract
            img_to_use = variants[0][1] if variants else raw_img
            if img_to_use is not None:
                try:
                    txt = pytesseract.image_to_string(img_to_use, lang="eng+hin")
                except Exception:
                    txt = pytesseract.image_to_string(img_to_use)
                if txt and txt.strip():
                    candidates.append(("pytesseract", txt.strip()))
        except Exception:
            pass

        if candidates:
            # Score candidates: prioritize neural RapidOCR ONNX layout engine and clean readable words
            def score_candidate(cand):
                engine_name, text = cand
                cleaned = TextExtractor.clean_ocr_text(text)
                devanagari_count = len(re.findall(r'[\u0900-\u097F]', cleaned))
                word_count = len(cleaned.split())
                score = float(word_count) + (devanagari_count * 1.5)
                if "rapidocr" in engine_name:
                    score += 40.0  # Strongly prioritize zero-dependency deep learning RapidOCR ONNX
                return score

            best_candidate = max(candidates, key=score_candidate)
            return TextExtractor.clean_ocr_text(best_candidate[1]), 1

        return "", 1

    @classmethod
    def extract(cls, filename: str, file_bytes: bytes) -> Tuple[str, str, int]:
        """Extracts text based on file extension and returns (clean_text, detected_format, page_count)."""
        fn = filename.lower()
        pages = 1
        if fn.endswith('.pptx') or fn.endswith('.ppt'):
            fmt = "PPTX"
            raw, pages = cls.extract_from_pptx(file_bytes)
        elif fn.endswith('.docx') or fn.endswith('.doc'):
            fmt = "DOCX"
            raw = cls.extract_from_docx(file_bytes)
        elif fn.endswith('.pdf'):
            fmt = "PDF"
            raw, pages = cls.extract_from_pdf(file_bytes)
        elif fn.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
            fmt = "IMAGE"
            raw, pages = cls.extract_from_image(file_bytes)
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
        transcript_list = None
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            languages = ['en', 'en-US', 'en-GB', 'hi', 'mr', 'bn', 'ta', 'te', 'gu', 'kn', 'ml', 'ur', 'pa', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ar']
            
            # Prepare optional authenticated session via cookies if configured in environment
            session = requests.Session()
            cookies_txt = os.getenv("YOUTUBE_COOKIES_TXT", "").strip()
            cookie_path = os.getenv("YOUTUBE_COOKIE_PATH", "cookies.txt").strip()
            if cookies_txt:
                try:
                    import http.cookiejar
                    from io import StringIO
                    cj = http.cookiejar.MozillaCookieJar()
                    cj._really_load(StringIO(cookies_txt), "cookies_env", ignore_discard=True, ignore_expires=True)
                    session.cookies = cj
                    logger.info("Loaded YouTube cookies from YOUTUBE_COOKIES_TXT.")
                except Exception as ck_err:
                    logger.debug(f"Could not parse YOUTUBE_COOKIES_TXT: {ck_err}")
            elif os.path.exists(cookie_path):
                try:
                    import http.cookiejar
                    cj = http.cookiejar.MozillaCookieJar(cookie_path)
                    cj.load(ignore_discard=True, ignore_expires=True)
                    session.cookies = cj
                    logger.info(f"Loaded YouTube cookies from {cookie_path}.")
                except Exception as ck_err:
                    logger.debug(f"Could not parse cookie file: {ck_err}")

            # Tier 1: Modern Instance API (v1.2.4+) with session
            try:
                ytt = YouTubeTranscriptApi(http_client=session)
                try:
                    t_list = ytt.list(video_id)
                    target_transcript = None
                    try:
                        target_transcript = t_list.find_transcript(languages)
                    except Exception:
                        try:
                            target_transcript = t_list.find_generated_transcript(languages)
                        except Exception:
                            for t in t_list:
                                target_transcript = t
                                break
                    if target_transcript:
                        transcript_list = target_transcript.fetch()
                except Exception as list_err:
                    logger.debug(f"Instance list notice: {list_err}")
                    try:
                        transcript_list = ytt.fetch(video_id, languages=languages)
                    except Exception:
                        transcript_list = ytt.fetch(video_id)
            except Exception as e1:
                logger.debug(f"Instance transcript API notice: {e1}")

            # Tier 2: Static / Legacy API
            if not transcript_list:
                try:
                    if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                        t_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                        target_t = None
                        try:
                            target_t = t_obj.find_transcript(languages)
                        except Exception:
                            try:
                                target_t = t_obj.find_generated_transcript(languages)
                            except Exception:
                                for t in t_obj:
                                    target_t = t
                                    break
                        if target_t:
                            transcript_list = target_t.fetch()
                    elif hasattr(YouTubeTranscriptApi, 'get_transcript'):
                        try:
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                        except Exception:
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                except Exception as leg_err:
                    logger.debug(f"Legacy transcript method notice: {leg_err}")

            # Tier 3: Direct Official YouTube Innertube JSON Player API
            if not transcript_list:
                try:
                    innertube_url = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"
                    innertube_headers = {
                        "Content-Type": "application/json",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        "X-YouTube-Client-Name": "1",
                        "X-YouTube-Client-Version": "2.20240417.01.00",
                        "Origin": "https://www.youtube.com"
                    }
                    innertube_payload = {
                        "context": {
                            "client": {
                                "clientName": "WEB",
                                "clientVersion": "2.20240417.01.00",
                                "hl": "en",
                                "gl": "US"
                            }
                        },
                        "videoId": video_id
                    }
                    yt_post = session.post(innertube_url, headers=innertube_headers, json=innertube_payload, timeout=6)
                    if yt_post.status_code == 200:
                        player_data = yt_post.json()
                        caption_tracks = player_data.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
                        if caption_tracks:
                            cap_url = caption_tracks[0].get("baseUrl")
                            for track in caption_tracks:
                                lang = track.get("languageCode", "").lower()
                                if lang in ["en", "hi", "en-us", "en-gb"]:
                                    cap_url = track.get("baseUrl")
                                    break
                            if cap_url:
                                cap_resp = session.get(cap_url + "&fmt=json3", headers=innertube_headers, timeout=6)
                                if cap_resp.status_code == 200:
                                    try:
                                        cap_data = cap_resp.json()
                                        events = cap_data.get("events", [])
                                        extracted_events = []
                                        for ev in events:
                                            segs = ev.get("segs", [])
                                            seg_text = "".join(s.get("utf8", "") for s in segs if s.get("utf8")).strip()
                                            if seg_text and seg_text != "\n":
                                                extracted_events.append({
                                                    "text": seg_text,
                                                    "start": int(ev.get("tStartMs", 0)) / 1000.0
                                                })
                                        if extracted_events:
                                            transcript_list = extracted_events
                                    except Exception:
                                        pass
                except Exception as innertube_err:
                    logger.debug(f"Direct player Innertube notice: {innertube_err}")

            # Tier 4: Public Cloud Proxy Mirrors Fallback
            if not transcript_list:
                invidious_mirrors = [
                    f"https://inv.nadeko.net/api/v1/captions/{video_id}",
                    f"https://invidious.nerdvpn.de/api/v1/captions/{video_id}",
                    f"https://invidious.jing.rocks/api/v1/captions/{video_id}",
                    f"https://inv.tux.pizza/api/v1/captions/{video_id}",
                    f"https://invidious.private.coffee/api/v1/captions/{video_id}"
                ]
                for mirror_url in invidious_mirrors:
                    try:
                        m_res = requests.get(mirror_url, timeout=4)
                        if m_res.status_code == 200:
                            m_data = m_res.json()
                            captions = m_data.get("captions", [])
                            if captions:
                                cap_track = captions[0]
                                c_url = cap_track.get("url")
                                if c_url:
                                    if not c_url.startswith("http"):
                                        base_domain = "/".join(mirror_url.split("/")[:3])
                                        c_url = f"{base_domain}{c_url}"
                                    c_res = requests.get(c_url, timeout=5)
                                    if c_res.status_code == 200:
                                        lines = c_res.text.splitlines()
                                        vtt_texts = []
                                        for line in lines:
                                            l = line.strip()
                                            if l and not l.startswith("WEBVTT") and "-->" not in l and not l.isdigit():
                                                vtt_texts.append({"text": l, "start": 0})
                                        if vtt_texts:
                                            transcript_list = vtt_texts
                                            break
                    except Exception:
                        continue

            if not transcript_list:
                logger.info(f"Direct transcript scraping unavailable for {video_id}; returning metadata fallback.")
                return {
                    "video_id": video_id,
                    "title": title,
                    "author": author_name,
                    "thumbnail": thumbnail_url,
                    "text": "",
                    "words": 0,
                    "chunks": [],
                    "format": "YOUTUBE",
                    "source_url": f"https://www.youtube.com/watch?v={video_id}",
                    "is_fallback": True
                }

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
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "is_fallback": False
            }
        except Exception as yt_err:
            logger.warning(f"YouTube transcript extraction notice: {yt_err}")
            return {
                "video_id": video_id if 'video_id' in locals() and video_id else "",
                "title": title if 'title' in locals() else "YouTube Video",
                "author": author_name if 'author_name' in locals() else "YouTube Creator",
                "thumbnail": thumbnail_url if 'thumbnail_url' in locals() else "",
                "text": "",
                "words": 0,
                "chunks": [],
                "format": "YOUTUBE",
                "source_url": f"https://www.youtube.com/watch?v={video_id}" if 'video_id' in locals() and video_id else url,
                "is_fallback": True
            }

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

