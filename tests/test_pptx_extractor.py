import io
import pytest
from pptx import Presentation
from pptx.util import Inches, Pt
from textSummarizer.components.text_extractor import TextExtractor


def create_sample_pptx():
    """Generates an in-memory PPTX presentation with multiple slides, bullet points, tables, and notes."""
    prs = Presentation()
    
    # Slide 1: Title & Subtitle
    title_slide_layout = prs.slide_layouts[0]
    slide1 = prs.slides.add_slide(title_slide_layout)
    title1 = slide1.shapes.title
    subtitle1 = slide1.placeholders[1]
    title1.text = "LexiBrief NLP Architecture"
    subtitle1.text = "High Precision Neural Text Summarization"

    # Slide 2: Bullet Points
    bullet_slide_layout = prs.slide_layouts[1]
    slide2 = prs.slides.add_slide(bullet_slide_layout)
    title2 = slide2.shapes.title
    title2.text = "Key System Capabilities"
    body2 = slide2.placeholders[1]
    tf2 = body2.text_frame
    tf2.text = "Multi-Format Ingestion (PDF, DOCX, PPTX, TXT)"
    p2 = tf2.add_paragraph()
    p2.text = "Dynamic Hallucination Guardrails"
    p3 = tf2.add_paragraph()
    p3.text = "Real-Time Cloud Deployment with MongoDB"

    # Slide 3: Speaker Notes
    slide3 = prs.slides.add_slide(title_slide_layout)
    slide3.shapes.title.text = "Conclusion and Q&A"
    slide3.placeholders[1].text = "Thank you for listening!"
    notes_slide = slide3.notes_slide
    notes_slide.notes_text_frame.text = "Emphasize zero-dependency fallback for Vercel."

    # Save to BytesIO
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_pptx_extraction():
    """Validates that TextExtractor extracts text, slide counts, and notes from PPTX."""
    pptx_bytes = create_sample_pptx()
    text, fmt, pages = TextExtractor.extract("quarterly_presentation.pptx", pptx_bytes)

    assert fmt == "PPTX"
    assert pages == 3
    assert "LexiBrief NLP Architecture" in text
    assert "Key System Capabilities" in text
    assert "Multi-Format Ingestion (PDF, DOCX, PPTX, TXT)" in text
    assert "Conclusion and Q&A" in text
    assert "Emphasize zero-dependency fallback for Vercel" in text


def test_pptx_fallback_without_library():
    """Validates the zero-dependency zipfile/ElementTree fallback parser for PPTX."""
    pptx_bytes = create_sample_pptx()
    # Call the fallback extractor directly
    text, pages = TextExtractor.extract_from_pptx(pptx_bytes)

    assert pages == 3
    assert "LexiBrief NLP Architecture" in text
    assert "Key System Capabilities" in text


def test_pptx_with_tables():
    """Validates table extraction in PPTX slides."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6]) # blank layout
    table_shape = slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(2))
    table = table_shape.table
    table.cell(0, 0).text = "Model"
    table.cell(0, 1).text = "Accuracy"
    table.cell(1, 0).text = "Pegasus"
    table.cell(1, 1).text = "94.5%"
    
    buf = io.BytesIO()
    prs.save(buf)
    
    text, fmt, pages = TextExtractor.extract("table_test.pptx", buf.getvalue())
    assert fmt == "PPTX"
    assert "Model | Accuracy" in text
    assert "Pegasus | 94.5%" in text
