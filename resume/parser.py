"""
Module: resume/parser.py
Phase 8: In-Memory PDF Resume Text Extraction & Normalization

Extracts clean text from uploaded PDF resume byte streams strictly in-memory.
Enforces privacy rules by avoiding permanent disk storage of candidate resumes.
"""

import io
import re
import pypdf


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts raw text from PDF bytes in-memory without writing files to disk.

    Parameters:
        pdf_bytes (bytes): Binary content of uploaded PDF file

    Returns:
        str: Cleaned extracted text from the PDF
    """
    if not pdf_bytes:
        return ""

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        extracted_pages = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

        full_text = "\n".join(extracted_pages)
        return clean_resume_text(full_text)

    except Exception as e:
        print(f"[ERROR] Failed to parse PDF resume bytes: {e}")
        return ""


def clean_resume_text(text: str) -> str:
    """
    Cleans and normalizes raw extracted resume text.

    Parameters:
        text (str): Raw text string

    Returns:
        str: Standardized, single-space normalized text string
    """
    if not text:
        return ""

    # Replace newlines, carriage returns, and tabs with single spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", text)

    # Remove non-printable control characters
    cleaned = re.sub(r"[^\x20-\x7E]+", " ", cleaned)

    # Collapse multiple consecutive spaces into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
