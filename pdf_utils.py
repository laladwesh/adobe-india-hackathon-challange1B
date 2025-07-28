from pypdf import PdfReader
from typing import List, Tuple
import re
import fitz

_title_re = re.compile(r"\b[A-Z][a-z]+\b")

def extract_heading_from_text(text: str) -> str:
    """
    Fallback: pick the first line that
    - starts with a letter
    - has at least two Title-Case words
    Otherwise the first non-empty letter-starting line.
    """
    for line in text.splitlines():
        ln = line.strip()
        if len(ln) > 120 or not ln:
            continue
        if not ln[0].isalpha():
            continue
        if len(_title_re.findall(ln)) >= 2:
            return ln
    for line in text.splitlines():
        ln = line.strip()
        if ln and ln[0].isalpha():
            return ln[:120]
    return "Untitled Section"

def _valid_heading(s: str) -> bool:
    """True if starts with letter and has at least 2 words."""
    if not s or not s[0].isalpha():
        return False
    return len(s.split()) >= 2

def extract_sections_from_pdf(pdf_path: str) -> List[Tuple[int, str, str]]:
    """
    Use PDF bookmarks if present.
    Returns list of (page_number, heading, full_page_text).
    """
    reader = PdfReader(pdf_path)
    raw = []

    def _recurse(out):
        for e in out:
            if isinstance(e, list):
                _recurse(e)
            else:
                try:
                    idx = reader.get_destination_page_number(e)
                    raw.append((idx + 1, (e.title or "").strip()))
                except Exception:
                    pass

    try:
        _recurse(reader.outline)
    except Exception:
        pass

    seen = set()
    secs = []
    for pg, title in raw:
        if pg in seen:
            continue
        seen.add(pg)
        text = reader.pages[pg - 1].extract_text() or ""
        heading = title if _valid_heading(title) else extract_heading_from_text(text)
        secs.append((pg, heading, text))
    return secs

def extract_layout_headings(pdf_path: str) -> List[Tuple[int, str, str]]:
    """
    For each page, pick the largest‑font span at top.
    Fallback to extract_heading_from_text if invalid.
    """
    doc = fitz.open(pdf_path)
    out = []
    for i in range(len(doc)):
        page = doc[i]
        full_text = page.get_text()
        best = {"size": 0, "y0": float("inf"), "text": ""}
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sz, y0 = span["size"], span["bbox"][1]
                    if sz > best["size"] or (sz == best["size"] and y0 < best["y0"]):
                        best.update(size=sz, y0=y0, text=span["text"].strip())
        heading = best["text"]
        if not _valid_heading(heading):
            heading = extract_heading_from_text(full_text)
        out.append((i + 1, heading, full_text))
    return out

def load_sections(pdf_path: str) -> List[dict]:
    """
    Merge bookmarks + layout headings, guaranteeing valid titles.
    """
    pages_seen = set()
    sections = []

    for pg, heading, text in extract_sections_from_pdf(pdf_path):
        sections.append({"page": pg, "section_title": heading, "text": text})
        pages_seen.add(pg)

    for pg, heading, text in extract_layout_headings(pdf_path):
        if pg in pages_seen:
            continue
        sections.append({"page": pg, "section_title": heading, "text": text})

    return sections
