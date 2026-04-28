"""PPTXConverter — convert PowerPoint PPTX files to markdown using Docling.

Stage 4: each slide gets a numbered <!-- Slide K --> marker. Speaker notes
(extracted by Docling into ContentLayer.FURNITURE since 2.65.0) appear
inline at slide-end with a <!-- Speaker notes --> marker. The --notes-only
flag emits a clean transcript: slide numbers + notes text only, dropping
bullet content. Slides without notes are skipped from the transcript.

No extra deps — Docling's PPTX backend handles speaker-note extraction
natively. python-pptx is used only by the fixture builder.

Empirical Docling behavior (verified at Stage 4):
- Default export shows slides separated by `# Title` headings only — no
  page-break markers. We split per-slide by detecting `^# ` boundaries.
- With BODY+FURNITURE included, speaker notes appear inline at slide-end
  as plain text. We extract them separately via iterate_items so we can
  emit them with their own marker.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import ContentLayer

from materials.core.base import BaseConverter, ConversionOptions, ConversionResult
from materials.core.output import default_output_path
from materials.core.verify import (
    VerifyReport,
    check_non_empty,
    check_word_retention,
    count_words,
)


PPTX_WORD_RETENTION_MIN = 0.75  # PPTX is title+bullets — high retention expected
_SLIDE_BREAK_TOKEN = "<!--__SLIDE_BREAK__-->"
# Match a top-level heading at line start (^# Title); used as the fallback
# slide-boundary detector when Docling's PPTX backend doesn't emit
# page_break_placeholder occurrences.
_TOP_HEADING_RE = re.compile(r"^# ", re.MULTILINE)


def _extract_notes_by_slide(doc) -> Dict[int, List[str]]:
    """Walk the DoclingDocument tree and collect FURNITURE-layer text items
    per slide, filtering out slide footers and page-number placeholders.
    Returns {slide_number: [note_text, ...]}.

    Docling 2.65.0's PPTX backend stores speaker notes with
    `content_layer == ContentLayer.FURNITURE` and the slide's `prov[0].page_no`
    set to the slide index (1-based).

    Heuristic for filtering: footers and page-number placeholders are typically
    short (under 80 chars) and repeat identically across multiple slides. Speaker
    notes are unique per slide and typically longer prose. Items that appear
    identically on 2+ slides are treated as footers and excluded.
    """
    # First pass: collect all raw text items per slide and track occurrences.
    raw_by_slide: Dict[int, List[str]] = {}
    text_occurrences: Dict[str, int] = {}  # text → count across all slides

    for item, _level in doc.iterate_items(
        included_content_layers={ContentLayer.FURNITURE}
    ):
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        slide_no = getattr(prov[0], "page_no", None)
        if slide_no is None:
            continue
        raw_by_slide.setdefault(slide_no, []).append(text)
        text_occurrences[text] = text_occurrences.get(text, 0) + 1

    # Second pass: filter out short text strings that appear on multiple slides.
    # These are likely footers (e.g., "Confidential", "© 2024") rather than
    # speaker notes.
    repeat_threshold = 2  # appears on >=2 slides
    short_threshold = 80  # chars
    notes_by_slide: Dict[int, List[str]] = {}
    for slide_no, items in raw_by_slide.items():
        kept = [
            t for t in items
            if not (
                text_occurrences.get(t, 0) >= repeat_threshold
                and len(t) < short_threshold
            )
        ]
        if kept:
            notes_by_slide[slide_no] = kept

    return notes_by_slide


def _split_body_markdown_by_slide(doc) -> List[str]:
    """Split body markdown into per-slide chunks.

    Strategy 1 (provenance-based): walk BODY items, group their text by
    `prov[0].page_no`. This is the most reliable path — Docling's PPTX
    backend populates per-element provenance accurately. Returns one chunk
    per slide number in ascending order.

    Strategy 2 (page_break_placeholder): try export_to_markdown with a
    sentinel token and split on it. Only reliable when Docling emits one
    token per slide boundary (empirically it emits N-2 tokens for an
    N-slide file in Docling 2.65.0, so this is NOT the primary path).

    Strategy 3 (heading split fallback): split on `^# ` heading boundaries.
    Used only when provenance and placeholder both fail.
    """
    from collections import defaultdict

    try:
        from docling_core.types.doc.document import TitleItem, SectionHeaderItem
        _TITLE_TYPES: tuple = (TitleItem, SectionHeaderItem)
    except ImportError:
        _TITLE_TYPES = ()

    # Strategy 1: group by provenance page_no.
    slide_lines: Dict[int, List[str]] = defaultdict(list)
    for item, _level in doc.iterate_items(
        included_content_layers={ContentLayer.BODY}
    ):
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = getattr(prov[0], "page_no", None)
        if page_no is None:
            continue
        # Render title items as markdown H1.
        if _TITLE_TYPES and isinstance(item, _TITLE_TYPES):
            slide_lines[page_no].append(f"# {text}")
        else:
            slide_lines[page_no].append(text)

    if slide_lines:
        # Check for gaps in slide numbering. If some slides have no provenance
        # items and are absent from slide_lines, log a warning and insert empty
        # entries to keep numbering sequential.
        logger = logging.getLogger("pptx_converter")
        max_slide = max(slide_lines.keys())
        expected_slides = set(range(1, max_slide + 1))
        actual_slides = set(slide_lines.keys())
        missing = expected_slides - actual_slides
        if missing:
            logger.warning(
                f"PPTX: {len(missing)} slide(s) had no provenance items and "
                f"are absent from output: {sorted(missing)}. "
                f"Slide numbering may not match the source deck."
            )
            # Insert empty entries so output numbering stays sequential
            for s in missing:
                slide_lines[s] = []

        return [
            "\n\n".join(lines).strip()
            for _, lines in sorted(slide_lines.items())
        ]

    # Strategy 2: page_break_placeholder export.
    md = doc.export_to_markdown(page_break_placeholder=_SLIDE_BREAK_TOKEN)
    if _SLIDE_BREAK_TOKEN in md:
        chunks = md.split(_SLIDE_BREAK_TOKEN)
        return [c.strip() for c in chunks if c.strip()]

    # Strategy 3: fallback split on top-level headings.
    md_default = doc.export_to_markdown()
    matches = list(_TOP_HEADING_RE.finditer(md_default))
    if not matches:
        return [md_default.strip()] if md_default.strip() else []

    chunks: List[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_default)
        chunks.append(md_default[start:end].strip())
    return chunks


def _format_default(slides_md: List[str], notes_by_slide: Dict[int, List[str]]) -> str:
    """Default rendering: <!-- Slide K --> + body + <!-- Speaker notes --> + notes."""
    parts: List[str] = []
    for i, body in enumerate(slides_md, start=1):
        parts.append(f"<!-- Slide {i} -->\n")
        if body:
            parts.append("\n")
            parts.append(body)
            parts.append("\n")
        if i in notes_by_slide:
            parts.append("\n<!-- Speaker notes -->\n\n")
            parts.append("\n\n".join(notes_by_slide[i]))
            parts.append("\n")
        parts.append("\n")
    return "".join(parts).rstrip() + "\n"


def _format_notes_only(num_slides: int, notes_by_slide: Dict[int, List[str]]) -> str:
    """--notes-only rendering: slide marker + notes text per slide that has
    notes; slides without notes are skipped entirely."""
    parts: List[str] = []
    for i in range(1, num_slides + 1):
        if i not in notes_by_slide:
            continue
        parts.append(f"<!-- Slide {i} -->\n\n")
        parts.append("\n\n".join(notes_by_slide[i]))
        parts.append("\n\n")
    return "".join(parts).rstrip() + "\n" if parts else ""


def _format_no_markers(slides_md: List[str], notes_by_slide: Dict[int, List[str]]) -> str:
    """--no-page-markers rendering: just the slide bodies and notes, no markers."""
    parts: List[str] = []
    for i, body in enumerate(slides_md, start=1):
        if body:
            parts.append(body)
            parts.append("\n\n")
        if i in notes_by_slide:
            parts.append("\n\n".join(notes_by_slide[i]))
            parts.append("\n\n")
    return "".join(parts).rstrip() + "\n"


class PPTXConverter(BaseConverter):
    """Convert PowerPoint PPTX files to markdown via Docling."""

    extensions = (".pptx",)

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("pptx_converter")

        src = Path(input_path)
        if not src.exists():
            return ConversionResult(status="error", error=f"File not found: {input_path}")

        try:
            converter = DocumentConverter()
            docling_result = converter.convert(str(src))
            doc = docling_result.document
        except Exception as exc:
            logger.warning(f"Docling failed on {input_path}: {exc}")
            return ConversionResult(
                status="error",
                error=f"Docling could not parse the PPTX: {exc}",
            )

        slides_md = _split_body_markdown_by_slide(doc)
        # Cross-check slide count against doc.pages if available.
        pages_attr = getattr(doc, "pages", None) or {}
        if hasattr(pages_attr, "__len__"):
            num_slides = max(len(slides_md), len(pages_attr))
        else:
            num_slides = len(slides_md)
        if num_slides == 0:
            return ConversionResult(status="error", error="PPTX has zero slides")

        notes_by_slide = _extract_notes_by_slide(doc)

        if options.notes_only:
            markdown = _format_notes_only(num_slides, notes_by_slide)
            # A deck with zero speaker notes is a legitimate state (visual-aid decks).
            # Don't fail; emit an informational marker so the user knows notes were
            # not found rather than lost.
            if not markdown.strip():
                markdown = "<!-- No speaker notes found in deck -->\n"
        elif options.page_markers:
            markdown = _format_default(slides_md, notes_by_slide)
        else:
            markdown = _format_no_markers(slides_md, notes_by_slide)

        # Cheap verifier
        report = VerifyReport()
        report.results.append(check_non_empty(markdown))
        if report.overall == "FAIL":
            return ConversionResult(
                status="error",
                error="Cheap verification failed: output empty",
            )
        source_md = doc.export_to_markdown(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        )
        source_words = count_words(source_md)
        output_words = count_words(markdown)
        if not options.notes_only:
            report.results.append(
                check_word_retention(source_words, output_words, PPTX_WORD_RETENTION_MIN)
            )

        if report.overall == "WARN":
            markdown = f"<!-- VERIFY: WARN -->\n\n{markdown}"

        out_path = default_output_path(input_path, options.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")

        return ConversionResult(
            status="success",
            output_file=str(out_path),
            statistics={
                "words": output_words,
                "characters": len(markdown),
                "source_words": source_words,
                "verify_status": report.overall,
                "slides": num_slides,
                "notes_slides": len(notes_by_slide),
            },
        )
