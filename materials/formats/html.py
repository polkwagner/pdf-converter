"""HTMLConverter — convert HTML files to markdown using Docling.

Stage 2: adds HTML support to materials-md. Optional bs4 pre-cleaning strips
noise (script, style, nav, footer, aside, sidebar/advert/cookie/consent classes)
before passing the cleaned HTML to Docling. Section markers are inserted at
H1 and H2 boundaries via regex over the markdown stream.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter

from materials.core.base import BaseConverter, ConversionOptions, ConversionResult
from materials.core.output import sanitize_heading_text
from materials.core.verify import (
    VerifyReport,
    check_non_empty,
    check_word_retention,
    count_words,
)


HTML_NOISE_TAGS = ("script", "style", "nav", "footer", "aside")
HTML_NOISE_CLASSES_RE = re.compile(r"(?i)sidebar|advert|cookie|consent")
HTML_WORD_RETENTION_MIN = 0.60

_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _count_html_words(html: str) -> int:
    """Estimate visible-word count by stripping tags. Used as the source-word
    baseline for the word-retention verifier."""
    text = _HTML_TAG_RE.sub(" ", html)
    return count_words(text)


def _strip_html_noise(html: str) -> str:
    """Use beautifulsoup4 to remove nav, scripts, styles, footers, asides,
    and elements with classes matching common ad/sidebar/cookie patterns.
    Raises RuntimeError if bs4 is not installed (the optional dependency)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "--strip-html-noise requires beautifulsoup4. "
            "Install with: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(HTML_NOISE_TAGS):
        tag.decompose()
    for elem in soup.find_all(class_=HTML_NOISE_CLASSES_RE):
        elem.decompose()
    return str(soup)


def _insert_section_markers(markdown: str) -> str:
    """Walk the markdown looking for H1/H2 lines (^# or ^##) and insert numbered
    `<!-- Section K: heading-text -->` markers before each one. H3+ are ignored.
    If no headings exist, prepend a single `<!-- Section 1: (untitled) -->` marker
    per spec §5.4.
    """
    matches = list(_HEADING_RE.finditer(markdown))

    if not matches:
        return f"<!-- Section 1: (untitled) -->\n\n{markdown}"

    parts = []
    cursor = 0
    for idx, match in enumerate(matches, start=1):
        parts.append(markdown[cursor:match.start()])
        heading_text = sanitize_heading_text(match.group(2))
        parts.append(f"<!-- Section {idx}: {heading_text} -->\n\n")
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(markdown[cursor:])
    return "".join(parts)


def _default_output_path(input_path: str) -> Path:
    src = Path(input_path)
    out_dir = src.parent / "converted"
    out_dir.mkdir(exist_ok=True)
    return out_dir / src.with_suffix(".md").name


class HTMLConverter(BaseConverter):
    """Convert HTML files to markdown using Docling, with section markers."""

    extensions = (".html", ".htm")

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("html_converter")

        src = Path(input_path)
        if not src.exists():
            return ConversionResult(status="error", error=f"File not found: {input_path}")

        try:
            html = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html = src.read_text(encoding="latin-1")

        source_words = _count_html_words(html)

        if options.strip_html_noise:
            try:
                html = _strip_html_noise(html)
            except RuntimeError as exc:
                logger.warning(str(exc))
                return ConversionResult(status="error", error=str(exc))

        # Docling wants a file path; write the (possibly cleaned) HTML to a temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(html)
            tmp_path = tmp.name

        try:
            converter = DocumentConverter()
            docling_result = converter.convert(tmp_path)
            markdown = docling_result.document.export_to_markdown()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if options.page_markers:
            markdown = _insert_section_markers(markdown)

        # Cheap verification (always runs).
        report = VerifyReport()
        report.results.append(check_non_empty(markdown))
        if report.overall == "FAIL":
            return ConversionResult(
                status="error",
                error="Cheap verification failed: output empty",
            )
        report.results.append(
            check_word_retention(source_words, count_words(markdown), HTML_WORD_RETENTION_MIN)
        )

        if report.overall == "WARN":
            markdown = f"<!-- VERIFY: WARN -->\n\n{markdown}"

        out_path = Path(options.output_path) if options.output_path else _default_output_path(input_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")

        return ConversionResult(
            status="success",
            output_file=str(out_path),
            statistics={
                "sections": markdown.count("<!-- Section "),
                "source_words": source_words,
                "output_words": count_words(markdown),
                "verify_status": report.overall,
            },
        )
