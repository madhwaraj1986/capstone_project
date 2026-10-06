"""
Document ingestion utilities for job descriptions and candidate CVs.

Supported formats: plain text (.txt, .md), PDF (.pdf), and Word (.docx).
The loader returns UTF-8 text so downstream LLM stages receive a uniform input.
"""

from __future__ import annotations

import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


# Extensions treated as candidate/job documents.
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def is_supported_file(path: str) -> bool:
    """
    Return True when the path has a supported document extension.

    Hidden files (names starting with '.') are ignored so macOS metadata
    such as .DS_Store is never sent to an LLM.
    """
    name = os.path.basename(path)
    if not name or name.startswith("."):
        return False
    _, ext = os.path.splitext(name.lower())
    return ext in _SUPPORTED_EXTENSIONS


def load_text_file(path: str) -> str:
    """Read a UTF-8 (with fallback) plain-text or markdown file."""
    with open(path, "rb") as handle:
        raw = handle.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_pdf_file(path: str) -> str:
    """
    Extract text from a PDF using pdfplumber.

    Pages are concatenated with blank lines so section boundaries remain
    roughly visible to the extraction LLM.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required to read PDF files. "
            "Install it with: pip install pdfplumber"
        ) from exc

    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages.append(extracted.strip())
    return "\n\n".join(pages)


def load_docx_file(path: str) -> str:
    """
    Extract paragraph text from a DOCX file without python-docx.

    A DOCX file is a ZIP archive. Word stores body text in word/document.xml.
    This keeps the dependency set small for evaluation environments.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(f"Invalid DOCX file: {path}") from exc

    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        texts = [
            node.text or ""
            for node in paragraph.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            )
        ]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def load_document(path: str) -> str:
    """
    Load a single document path into normalized text.

    Raises FileNotFoundError or RuntimeError for unreadable/unsupported files.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Document not found: {path}")

    _, ext = os.path.splitext(path.lower())
    if ext in {".txt", ".md"}:
        text = load_text_file(path)
    elif ext == ".pdf":
        text = load_pdf_file(path)
    elif ext == ".docx":
        text = load_docx_file(path)
    else:
        raise RuntimeError(f"Unsupported file type '{ext}' for {path}")

    cleaned = text.strip()
    if not cleaned:
        raise RuntimeError(f"No extractable text found in {path}")
    return cleaned


def collect_cv_paths(
    cv_paths: List[str], cv_dir: str | None, exclude_paths: List[str] | None = None
) -> List[str]:
    """
    Build a de-duplicated list of CV file paths.

    Accepts explicit file paths and/or a directory of CV documents.
    exclude_paths is used to skip the job-description file if it lives in --cv-dir.
    """
    discovered: List[str] = []
    excluded = {os.path.abspath(path) for path in (exclude_paths or [])}

    for path in cv_paths or []:
        discovered.append(os.path.abspath(path))

    if cv_dir:
        if not os.path.isdir(cv_dir):
            raise FileNotFoundError(f"CV directory not found: {cv_dir}")
        for name in os.listdir(cv_dir):
            full = os.path.join(cv_dir, name)
            if os.path.isfile(full) and is_supported_file(full):
                discovered.append(os.path.abspath(full))

    unique: List[str] = []
    seen = set()
    for path in discovered:
        if path in excluded or path in seen:
            continue
        seen.add(path)
        unique.append(path)

    if not unique:
        raise RuntimeError(
            "No CV files were provided. Use --cvs and/or --cv-dir with "
            ".txt, .md, .pdf, or .docx files."
        )
    return unique


def load_named_documents(
    paths: List[str],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Load many documents and return ((source_name, text) pairs, skipped files).

    source_name is the basename, which is used as a fallback candidate label
    if the extraction LLM cannot recover a person name. An empty, corrupt, or
    scanned (image-only) CV is reported in the skipped list instead of
    aborting the ranking of every other candidate.
    """
    loaded: List[Tuple[str, str]] = []
    skipped: List[Dict[str, str]] = []
    for path in paths:
        try:
            text = load_document(path)
        except (OSError, RuntimeError) as exc:
            skipped.append({"source_file": os.path.basename(path), "reason": str(exc)})
            continue
        loaded.append((os.path.basename(path), text))
    return loaded, skipped
