"""DOCXConverter — convert Word DOCX files to markdown using Docling +
python-docx.

Stage 3: lean default per spec §4.6 — comments dropped, revisions
accepted-final, images dropped (placeholder kept). Footnotes always preserved.
Three opt-in flags: --full (comments appendix), --show-revisions (inline
ins/del markers), --keep-images (extract images to disk).

Empirical Docling behavior (verified at Stage 3 implementation time):
- Docling drops footnotes entirely; we extract from footnotes.xml ourselves
  and append a `## Footnotes` section.
- Docling drops comments entirely; --full re-injects them via XML extraction.
- Docling drops BOTH w:ins and w:del runs. We always re-surface insertions
  (accepted-final behavior); --show-revisions wraps them with [+ ... +]
  markers and also surfaces deletions as [- ... -].
- Docling shifts heading levels in DOCX output (H1 source → ## markdown,
  H2 source → ###). Section markers auto-detect the two smallest heading
  levels actually present.

Comment-extraction scope: top-level comments only (`comments.xml`).
Threaded replies in `commentsExtended.xml` are a known limitation deferred
to a follow-up after stage 5.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree
from docling.document_converter import DocumentConverter

from materials.core.base import BaseConverter, ConversionOptions, ConversionResult
from materials.core.output import default_output_path, sanitize_heading_text
from materials.core.verify import (
    VerifyReport,
    check_non_empty,
    check_word_retention,
    count_words,
)


# Safe XML parser: don't resolve external entities, don't make network requests.
# Mitigates XXE attacks where a crafted .docx contains DTD entity declarations
# pointing at file:// URLs or http:// servers.
_SAFE_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)

DOCX_WORD_RETENTION_MIN = 0.90

# Match any heading level (1-6); we filter by level dynamically per document.
_ANY_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Word XML namespace
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@dataclass
class _CommentEntry:
    id: str
    author: str
    date: str
    text: str
    referenced_text: str  # body-text snippet the comment is anchored to


@dataclass
class _Revision:
    kind: str  # "ins" or "del"
    text: str


def _qn(local: str) -> str:
    """Resolve a local tag name (with or without 'w:' prefix) to fully-qualified
    Clark notation: {namespace}localname."""
    # Strip namespace prefix if present (e.g. "w:id" → "id")
    local = local.split(":")[-1]
    return f"{{{_W_NS}}}{local}"


def _extract_comments(doc) -> List[_CommentEntry]:
    """Read comments.xml from the .docx archive (via python-docx's part lookup)
    and pair each comment with the body-text snippet it anchors to.

    Best-effort: if a commentRange anchor isn't matchable in the body, the
    comment still appears in the appendix without a referenced-text snippet.
    """
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    comments_part = None
    for rel in doc.part.rels.values():
        if rel.reltype == RT.COMMENTS:
            comments_part = rel.target_part
            break
    if comments_part is None:
        return []

    comments_root = etree.fromstring(comments_part.blob, _SAFE_XML_PARSER)

    # Map comment IDs to their anchored body-text snippet.
    # in_range persists across paragraph boundaries so that commentRangeStart
    # in paragraph N and commentRangeEnd in paragraph N+1 are reconciled correctly.
    anchored: dict[str, str] = {}
    body = doc.element.body
    in_range: dict[str, list[str]] = {}
    for child in body.iter():
        tag = child.tag
        if tag == _qn("commentRangeStart"):
            cid = child.get(_qn("id"))
            if cid is not None:
                in_range[cid] = []
        elif tag == _qn("commentRangeEnd"):
            cid = child.get(_qn("id"))
            if cid is not None and cid in in_range:
                anchored[cid] = "".join(in_range.pop(cid)).strip()
        elif tag == _qn("t") and in_range:
            if child.text:
                for collected in in_range.values():
                    collected.append(child.text)

    comments: List[_CommentEntry] = []
    for c in comments_root:
        if c.tag != _qn("comment"):
            continue
        cid = c.get(_qn("id")) or ""
        author = c.get(_qn("author")) or "Unknown"
        date = c.get(_qn("date")) or ""
        text_parts = [t.text for t in c.iter(_qn("t")) if t.text]
        comments.append(_CommentEntry(
            id=cid,
            author=author,
            date=date.split("T")[0] if "T" in date else date,
            text="".join(text_parts).strip(),
            referenced_text=anchored.get(cid, ""),
        ))
    return comments


def _extract_footnotes(doc) -> List[Tuple[str, str]]:
    """Return [(footnote_id, footnote_text), ...] from footnotes.xml,
    excluding Word's reserved separator/continuation entries (id=-1, 0)."""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    footnotes_part = None
    for rel in doc.part.rels.values():
        if rel.reltype == RT.FOOTNOTES:
            footnotes_part = rel.target_part
            break
    if footnotes_part is None:
        return []

    root = etree.fromstring(footnotes_part.blob, _SAFE_XML_PARSER)
    out: List[Tuple[str, str]] = []
    for fn in root:
        if fn.tag != _qn("footnote"):
            continue
        fid = fn.get(_qn("id")) or ""
        if fid in ("-1", "0"):
            continue
        text_parts = [t.text for t in fn.iter(_qn("t")) if t.text]
        text = "".join(text_parts).strip()
        if text:
            out.append((fid, text))
    return out


def _extract_revisions(doc) -> List[_Revision]:
    """Return all w:ins / w:del text content from top-level body paragraphs.

    Excludes revisions nested inside tables, headers, footers, and footnotes
    because Docling renders those structures inline; surfacing their
    revisions again would duplicate content.
    """
    revisions: List[_Revision] = []
    body = doc.element.body
    # Only iterate direct children that are paragraphs (w:p).
    # Tables (w:tbl) and structured content blocks are skipped to avoid
    # duplicating content that Docling already renders inline.
    for top_level_child in body:
        if top_level_child.tag != _qn("p"):
            continue
        for elem in top_level_child.iter():
            if elem.tag == _qn("ins"):
                text_parts = [t.text for t in elem.iter(_qn("t")) if t.text]
                if text_parts:
                    revisions.append(_Revision("ins", "".join(text_parts)))
            elif elem.tag == _qn("del"):
                text_parts = [t.text for t in elem.iter(_qn("delText")) if t.text]
                if text_parts:
                    revisions.append(_Revision("del", "".join(text_parts)))
    return revisions


def _insert_section_markers(markdown: str) -> str:
    """Insert numbered <!-- Section K: heading --> markers at the two smallest
    heading levels present in the markdown.

    For DOCX, Docling shifts heading levels (source H1 → ## markdown,
    source H2 → ###). Auto-detecting the two smallest levels makes this
    robust to that shift without hardcoding "##" and "###".

    If no headings exist, prepend a single fallback marker per spec §5.4.
    """
    matches = list(_ANY_HEADING_RE.finditer(markdown))
    if not matches:
        return f"<!-- Section 1: (untitled) -->\n\n{markdown}"

    levels_present = sorted(set(len(m.group(1)) for m in matches))
    section_levels = set(levels_present[:2])

    parts: List[str] = []
    cursor = 0
    section_idx = 0
    for match in matches:
        if len(match.group(1)) not in section_levels:
            continue
        section_idx += 1
        parts.append(markdown[cursor:match.start()])
        heading_text = sanitize_heading_text(match.group(2))
        parts.append(f"<!-- Section {section_idx}: {heading_text} -->\n\n")
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(markdown[cursor:])
    return "".join(parts)


def _apply_insertions(markdown: str, revisions: List[_Revision], show_revisions: bool) -> str:
    """Re-surface inserted text that Docling dropped.

    Docling drops w:ins runs entirely. In accepted-final (lean) mode we append
    insertions as plain prose. In show-revisions mode we wrap them with markers.

    Deletions are only surfaced in show-revisions mode.
    """
    ins_items = [r.text.strip() for r in revisions if r.kind == "ins" and r.text.strip()]
    del_items = [r.text.strip() for r in revisions if r.kind == "del" and r.text.strip()]

    for text in ins_items:
        if show_revisions:
            marker = f"[+ {text} +]"
        else:
            marker = text
        # Append at end as prose (Docling lost positional context).
        markdown = markdown.rstrip() + f"\n\n{marker}\n"

    if show_revisions:
        for text in del_items:
            markdown = markdown.rstrip() + f"\n\n[- {text} -]\n"

    return markdown


def _append_footnotes_section(markdown: str, footnotes: List[Tuple[str, str]]) -> str:
    """Append a `## Footnotes` section at the end of the markdown.

    Docling drops footnotes entirely from DOCX output, so we have full
    control over their rendering. Format: standard markdown footnote
    references `[^N]: text`.
    """
    if not footnotes:
        return markdown
    lines = ["", "", "## Footnotes", ""]
    for fid, ftext in footnotes:
        lines.append(f"[^{fid}]: {ftext}")
    return markdown.rstrip() + "\n".join(lines) + "\n"


def _append_reviewer_comments(
    markdown: str, comments: List[_CommentEntry]
) -> str:
    """Append a `## Reviewer Comments` appendix. Best-effort inline `[CN]`
    anchors after the referenced phrase if it's findable in the markdown.
    """
    if not comments:
        return markdown

    for i, c in enumerate(comments, start=1):
        anchor_id = f"[C{i}]"
        if c.referenced_text and c.referenced_text in markdown:
            markdown = markdown.replace(
                c.referenced_text,
                f"{c.referenced_text} {anchor_id}",
                1,
            )

    lines = ["", "", "## Reviewer Comments", ""]
    for i, c in enumerate(comments, start=1):
        date_str = f" ({c.date})" if c.date else ""
        ref_str = (
            f"\n> Referenced text: \"{c.referenced_text}\""
            if c.referenced_text
            else ""
        )
        lines.append(f"[C{i}] **{c.author}**{date_str}: {c.text}{ref_str}")
        lines.append("")
    return markdown.rstrip() + "\n".join(lines) + "\n"


class DOCXConverter(BaseConverter):
    """Convert Word DOCX files to markdown via Docling + python-docx."""

    extensions = (".docx",)

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("docx_converter")

        src = Path(input_path)
        if not src.exists():
            return ConversionResult(status="error", error=f"File not found: {input_path}")

        try:
            from docx import Document
        except ImportError:
            return ConversionResult(
                status="error",
                error=(
                    "python-docx is required for DOCX conversion. "
                    "Install with: pip install python-docx"
                ),
            )

        try:
            doc = Document(str(src))
        except Exception as exc:
            return ConversionResult(
                status="error",
                error=f"Could not open DOCX: {exc}",
            )

        # Always extract revisions — we need insertions even in lean mode.
        revisions = _extract_revisions(doc)
        comments = _extract_comments(doc) if options.full else []
        footnotes = _extract_footnotes(doc)

        try:
            converter = DocumentConverter()
            docling_result = converter.convert(str(src))
            markdown = docling_result.document.export_to_markdown()
        except Exception as exc:
            logger.warning(f"Docling failed on {input_path}: {exc}")
            return ConversionResult(
                status="error",
                error=f"Docling could not parse the DOCX: {exc}",
            )

        # Post-process layers in spec order:

        # Re-surface insertions (always) and optionally deletions.
        if revisions:
            markdown = _apply_insertions(markdown, revisions, options.show_revisions)

        if not options.keep_images:
            # Replace inline image references with the placeholder.
            markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", "<!-- image -->", markdown)

        # Insert section markers BEFORE appending footnotes/comments so that
        # the auto-generated appendix headings don't get counted as sections.
        if options.page_markers:
            markdown = _insert_section_markers(markdown)

        if footnotes:
            markdown = _append_footnotes_section(markdown, footnotes)

        if options.full and comments:
            markdown = _append_reviewer_comments(markdown, comments)

        # Cheap verifier
        report = VerifyReport()
        report.results.append(check_non_empty(markdown))
        if report.overall == "FAIL":
            return ConversionResult(
                status="error",
                error="Cheap verification failed: output empty",
            )
        # Source-word baseline = sum of paragraph text via python-docx.
        source_words = sum(count_words(p.text) for p in doc.paragraphs)
        output_words = count_words(markdown)
        report.results.append(
            check_word_retention(source_words, output_words, DOCX_WORD_RETENTION_MIN)
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
                "sections": markdown.count("<!-- Section "),
                "headings": len(_ANY_HEADING_RE.findall(markdown)),
                "comments": len(comments),
                "footnotes": len(footnotes),
                "revisions": len(revisions),
            },
        )
