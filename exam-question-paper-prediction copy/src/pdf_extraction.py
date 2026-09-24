"""
pdf_extraction.py
------------------
Extracts question text from an uploaded PDF question paper using
pdfplumber, then splits the raw text into individual question rows
using a simple, tunable question-numbering heuristic (e.g. "1.",
"Q1.", "12)") so it can be previewed and appended to the dataset.
"""

import io
import re
import pandas as pd

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except Exception:
    _PDFPLUMBER_AVAILABLE = False


# Matches question numbering at the start of a line, e.g.:
#   "1. What is..."   "Q1) Which of..."   "12 . Explain..."
_QUESTION_START_RE = re.compile(
    r"(?m)^\s*(?:Q\.?\s*)?(\d{1,3})\s*[\.\)]\s+"
)


def pdfplumber_available() -> bool:
    return _PDFPLUMBER_AVAILABLE


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from all pages of an uploaded PDF (bytes)."""
    if not _PDFPLUMBER_AVAILABLE:
        raise ImportError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        )
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def detect_questions(raw_text: str) -> list:
    """
    Split raw extracted text into individual question strings using
    numbered-question boundaries. Falls back to line-based splitting
    if no numbering pattern is detected.
    """
    if not raw_text or not raw_text.strip():
        return []

    matches = list(_QUESTION_START_RE.finditer(raw_text))
    questions = []

    if matches:
        for idx, m in enumerate(matches):
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            q_text = raw_text[start:end].strip()
            q_text = re.sub(r"\s+", " ", q_text)
            if q_text:
                questions.append(q_text)
    else:
        # Fallback: treat each non-empty line as a candidate question
        for line in raw_text.split("\n"):
            line = line.strip()
            if len(line) > 15:  # skip headers/footers/short noise
                questions.append(line)

    return questions


def build_preview_dataframe(questions: list, year: int, subject: str,
                             chapter: str = "Unknown", topic: str = "Unknown",
                             marks: int = 4) -> pd.DataFrame:
    """
    Wrap detected raw question strings into a dataframe matching the
    application's standard schema, ready for user review before it
    is appended to the main dataset.
    """
    rows = []
    for i, q in enumerate(questions, start=1):
        rows.append({
            "ID": None,  # assigned when merged into the main dataset
            "Year": year,
            "Subject": subject,
            "Question_No": i,
            "Chapter": chapter,
            "Question": q,
            "Topic": topic,
            "Marks": marks,
        })
    return pd.DataFrame(rows, columns=[
        "ID", "Year", "Subject", "Question_No", "Chapter", "Question", "Topic", "Marks"
    ])
