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
        Cleans OCR artifacts, repetitive non-alphanumeric symbol runs, 
        and noisy lines resulting from camera photo scanning or handwritten text recognition.
        """
        if not text:
            return ""
        
        # 1. Strip repetitive solitary non-alphanumeric noise characters (e.g. | | | ~~~ ^^^ ___)
        text = re.sub(r'([|~_^\/\\<>{}\[\]*+=#`¬¢§±µ¿¡])\s*\1+', ' ', text)
        # Remove standalone isolated symbol tokens
        text = re.sub(r'(?:^|\s)[|~_^\/\\<>{}\[\]*+=#`¬¢§±µ¿¡]{1,3}(?=\s|$)', ' ', text)
        
        # 2. Filter line by line
        clean_lines = []
        for line in text.split('\n'):
            line_str = line.strip()
            if not line_str:
                continue
            # Calculate alphanumeric vs total printable length
            alpha_chars = sum(1 for c in line_str if c.isalnum())
            total_printable = sum(1 for c in line_str if not c.isspace())
            
            # If line is mostly symbols/noise (>65% non-alphanumeric) with very few letters, ignore it
            if total_printable > 0 and (alpha_chars / total_printable) < 0.35 and alpha_chars < 3:
                continue
            
            # Clean internal spacing
            line_clean = re.sub(r'\s+', ' ', line_str)
            clean_lines.append(line_clean)
            
        cleaned = "\n".join(clean_lines)
        return TextExtractor.clean_text(cleaned)

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans and normalizes extracted text with compiled regex patterns and auto-strips timestamps and OCR noise."""
        if not text:
            return ""
        # Strip repetitive OCR artifact characters
        text = re.sub(r'([|~_^\/\\<>{}\[\]*+=#`¬¢§±µ¿¡])\s*\1{2,}', ' ', text)
        # Normalize whitespace and line breaks
        text = _RE_CRLF.sub('\n', text)
        text = _RE_SPACES.sub(' ', text)
        # Strip standalone timestamp lines (e.g. "0:15", "01:23", "[02:45]")
        text = re.sub(r'^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*$', '', text, flags=re.MULTILINE)
        # Strip inline timestamp prefixes (e.g. "0:15 - Hello" or "[0:15] Hello")
        text = re.sub(r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*[-–—:]?\s*', '', text)
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
    def extract_from_image(file_bytes: bytes) -> Tuple[str, int]:
        """Extracts text from images (PNG, JPG, JPEG, WEBP, BMP) using PyMuPDF OCR or Pillow."""
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="png")
            pages = []
            page_count = len(doc)
            for page in doc:
                try:
                    tp = page.get_textpage_ocr(language="eng", dpi=150)
                    text = page.get_text(textpage=tp)
                    if text and text.strip():
                        pages.append(text.strip())
                except Exception:
                    pass
            doc.close()
            if pages:
                return "\n\n".join(pages), max(1, page_count)
        except Exception as e:
            logger.debug(f"PyMuPDF image OCR note: {e}")
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

