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
from materials.core.output import default_output_path, sanitize_heading_text
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
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w-]+)""", re.IGNORECASE)


def _detect_encoding(raw: bytes) -> str:
    """Best-effort encoding detection for an HTML byte stream.

    Order:
      1. UTF BOMs (utf-8-sig, utf-16-le, utf-16-be)
      2. <meta charset="..."> declaration in the first 4KB
      3. Default to utf-8

    The caller wraps the returned name with an encoding-error fallback chain
    (utf-8 → cp1252 → latin-1) so undetected mis-encoded files still read,
    just with mojibake rather than crashes. cp1252 is the dominant real-world
    "lying about being utf-8" case (Word HTML exports).
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    head = raw[:4096]
    match = _META_CHARSET_RE.search(head)
    if match:
        try:
            return match.group(1).decode("ascii", errors="ignore").strip().lower() or "utf-8"
        except Exception:
            pass
    return "utf-8"


def _read_html_text(path: Path) -> str:
    """Read an HTML file as text with encoding-aware fallback.

    Tries the detected encoding first, then cp1252 (Word HTML exports), then
    latin-1 (always succeeds — every byte is a valid latin-1 codepoint, but
    the result will be mojibake for true non-Western files). Logs a warning
    when the fallback is taken so the user has a breadcrumb if the output
    looks corrupted.
    """
    raw = path.read_bytes()
    detected = _detect_encoding(raw)
    logger = logging.getLogger("html_converter")
    for encoding in (detected, "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 cannot raise UnicodeDecodeError, so this is unreachable in
    # practice. Log and fall through with errors="replace" as a final guard.
    logger.warning("All encoding attempts failed; falling back to utf-8 with replacement")
    return raw.decode("utf-8", errors="replace")


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


class HTMLConverter(BaseConverter):
    """Convert HTML files to markdown using Docling, with section markers."""

    extensions = (".html", ".htm")

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("html_converter")

        src = Path(input_path)
        if not src.exists():
            return ConversionResult(status="error", error=f"File not found: {input_path}")

        try:
            html = _read_html_text(src)
        except Exception as exc:
            return ConversionResult(status="error", error=f"Could not read {input_path}: {exc}")

        if options.strip_html_noise:
            try:
                html = _strip_html_noise(html)
            except RuntimeError as exc:
                logger.warning(str(exc))
                return ConversionResult(status="error", error=str(exc))

        # Source-word baseline for the verifier is the HTML Docling actually sees,
        # so we measure AFTER stripping. Otherwise --strip-html-noise spuriously
        # depresses the retention ratio because the stripped nav/footer counted
        # toward source but never made it to output.
        source_words = _count_html_words(html)

        # Docling wants a file path; write the (possibly cleaned) HTML to a temp file.
        # Track the temp path independently so cleanup runs even if write() raises.
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(html)

            try:
                converter = DocumentConverter()
                docling_result = converter.convert(tmp_path)
                markdown = docling_result.document.export_to_markdown()
            except Exception as exc:
                logger.warning(f"Docling failed on {input_path}: {exc}")
                return ConversionResult(
                    status="error",
                    error=f"Docling could not parse the HTML: {exc}",
                )
        finally:
            if tmp_path:
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
        # Capture output_words ONCE before any prefix is added, so the value
        # used for the verifier and the value reported in statistics agree.
        output_words = count_words(markdown)
        report.results.append(
            check_word_retention(source_words, output_words, HTML_WORD_RETENTION_MIN)
        )

        if report.overall == "WARN":
            markdown = f"<!-- VERIFY: WARN -->\n\n{markdown}"

        out_path = default_output_path(input_path, options.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")

        # Canonical statistics schema (see ConversionResult docstring in base.py).
        return ConversionResult(
            status="success",
            output_file=str(out_path),
            statistics={
                "words": output_words,
                "characters": len(markdown),
                "source_words": source_words,
                "verify_status": report.overall,
                "sections": markdown.count("<!-- Section "),
                "headings": len(_HEADING_RE.findall(markdown)),
            },
        )
