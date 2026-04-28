# Stage 1: PDF Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the existing 1632-line `pdf_to_markdown.py` into a `materials/` package structure (`core/` + `formats/pdf.py`) with a new `convert.py` CLI, while producing **byte-identical** PDF→markdown output.

**Architecture:** Two-phase refactor. Phase A adds new layer that wraps the existing code (no functional change, just an indirection). Phase B moves the actual function definitions out of `pdf_to_markdown.py` into `formats/pdf.py` and replaces the old file with a deprecation shim. A migration test runs the same fixture through a frozen legacy snapshot and the new code path, asserting byte equality at every commit.

**Tech Stack:** Python 3.14, Docling 2.65.0, PyMuPDF, RapidFuzz, Rich, pytest (new in this stage), ReportLab (new — fixture builder only).

**Spec:** `docs/superpowers/specs/2026-04-28-materials-converter-design.md`

---

## File map

**Create:**
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/fixtures/build/build_pdf.py`
- `tests/fixtures/sample.pdf` (binary, generated)
- `tests/fixtures/legacy_pdf_to_markdown.py` (frozen copy of current `pdf_to_markdown.py`)
- `tests/fixtures/legacy_console.py` (frozen copy of current `console.py`, since legacy imports it)
- `tests/fixtures/README.md`
- `tests/test_pdf.py`
- `materials/__init__.py`
- `materials/core/__init__.py`
- `materials/core/base.py`
- `materials/core/output.py`
- `materials/core/verify.py`
- `materials/formats/__init__.py`
- `materials/formats/pdf.py`
- `convert.py`

**Modify:**
- `pdf_to_markdown.py` (becomes a deprecation shim — Task 12)
- `requirements.txt` (add `pytest`, `reportlab`)
- `CLAUDE.md` (Task 14)

**Untouched:**
- `console.py`
- `verify_conversion.py`
- `verify_page_markers.py`
- `README.md` (updated in stage 5)

---

## Task 1: Set up test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/build/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add pytest and reportlab to requirements.txt**

Open `requirements.txt` and append:

```text
pytest>=8.0.0  # Test runner (Stage 1+)
reportlab>=4.0.0  # Used only by tests/fixtures/build/build_pdf.py
```

The full file should now be:

```text
docling>=2.0.0
PyMuPDF>=1.23.0  # For PDF metadata and page labels
rapidfuzz>=3.0.0  # 10-100x faster fuzzy matching than difflib
rich>=13.0.0  # Visual progress bars and formatted output
pytest>=8.0.0  # Test runner (Stage 1+)
reportlab>=4.0.0  # Used only by tests/fixtures/build/build_pdf.py
```

- [ ] **Step 2: Install the new dependencies into the venv**

Run: `./venv/bin/pip install pytest>=8.0.0 reportlab>=4.0.0`

Expected: pip installs both, no errors. Verify with `./venv/bin/pytest --version` (should print a version like `pytest 8.x.x`).

- [ ] **Step 3: Create empty package init files**

```bash
mkdir -p tests/fixtures/build
```

Create `tests/__init__.py` (empty file — just `touch tests/__init__.py`).
Create `tests/fixtures/build/__init__.py` (empty file).

- [ ] **Step 4: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures and configuration."""
import sys
from pathlib import Path

# Make repo root importable so tests can `import convert`, `import materials`, etc.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
```

- [ ] **Step 5: Verify pytest collection works (no tests yet, but no errors)**

Run: `./venv/bin/python -m pytest tests/ --collect-only`
Expected: `no tests ran in 0.0Xs` with exit code 0 or 5 (5 = "no tests collected" — fine here).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "Add pytest test infrastructure for stage 1 refactor"
```

---

## Task 2: Build the PDF fixture

**Files:**
- Create: `tests/fixtures/build/build_pdf.py`
- Create: `tests/fixtures/sample.pdf` (generated, committed)

- [ ] **Step 1: Write the fixture builder**

Create `tests/fixtures/build/build_pdf.py`:

```python
"""Build tests/fixtures/sample.pdf — a 3-page PDF with a Roman-numeral page label
on page 1 (front matter), a regular page, and a page containing a simple table.

Run from repo root:
    ./venv/bin/python tests/fixtures/build/build_pdf.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    PageBreak,
    Table,
    TableStyle,
    Spacer,
)

OUT = Path(__file__).resolve().parent.parent / "sample.pdf"


def build() -> Path:
    doc = SimpleDocTemplate(str(OUT), pagesize=LETTER, title="Sample Fixture")
    styles = getSampleStyleSheet()
    story = []

    # Page 1 — front matter heading; we set page label "i" via a low-level hack below
    story.append(Paragraph("Preface", styles["Title"]))
    story.append(Paragraph(
        "This preface page is intended to test page-label translation: the visible "
        "page label should be 'i' (Roman numeral), even though the page index is 0.",
        styles["BodyText"],
    ))
    story.append(PageBreak())

    # Page 2 — body
    story.append(Paragraph("Chapter 1: Introduction", styles["Heading1"]))
    story.append(Paragraph(
        "This second page contains body content. Position markers should label "
        "this as Page 1 if Roman-numeral front matter is offset correctly.",
        styles["BodyText"],
    ))
    story.append(PageBreak())

    # Page 3 — table
    story.append(Paragraph("Chapter 2: Table Test", styles["Heading1"]))
    story.append(Spacer(1, 12))
    data = [
        ["Column A", "Column B", "Column C"],
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon", "zeta"],
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(t)

    doc.build(story)

    # ReportLab does not expose page-label dicts via SimpleDocTemplate. Patch via PyMuPDF.
    import fitz
    pdf = fitz.open(str(OUT))
    pdf.set_page_labels([
        {"startpage": 0, "style": "r", "prefix": "", "firstpagenum": 1},  # i, ii (Roman)
        {"startpage": 1, "style": "D", "prefix": "", "firstpagenum": 1},  # 1, 2 (Decimal)
    ])
    pdf.saveIncr()
    pdf.close()

    return OUT


if __name__ == "__main__":
    p = build()
    size_kb = p.stat().st_size / 1024
    print(f"Wrote {p} ({size_kb:.1f} KB)")
```

- [ ] **Step 2: Run the builder**

Run: `./venv/bin/python tests/fixtures/build/build_pdf.py`
Expected: prints `Wrote .../sample.pdf (X.X KB)` with size under 100KB. File exists at `tests/fixtures/sample.pdf`.

- [ ] **Step 3: Smoke-test the fixture by running the existing converter on it**

Run: `./venv/bin/python pdf_to_markdown.py tests/fixtures/sample.pdf -o /tmp/sample-stage1.md`
Expected: completes without error, `/tmp/sample-stage1.md` contains text like "Preface" and "Chapter 1" with `<!-- Page i -->` and `<!-- Page 1 -->` markers (the exact marker text comes from the page labels we set).

- [ ] **Step 4: Write `tests/fixtures/README.md`**

```markdown
# Test fixtures

Each fixture is **scripted** — committed alongside its builder so it can be regenerated
deterministically. To rebuild a fixture, run its builder from the repo root:

    ./venv/bin/python tests/fixtures/build/build_pdf.py

| Fixture | Purpose | Builder |
|---|---|---|
| `sample.pdf` | 3-page PDF with Roman-numeral page labels (front matter) and a table | `build/build_pdf.py` |

Future stages add `sample_with_comments.docx`, `sample_with_notes.pptx`, `sample_article.html`.
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/build/build_pdf.py tests/fixtures/sample.pdf tests/fixtures/README.md
git commit -m "Add scripted PDF fixture for stage 1 tests"
```

---

## Task 3: Capture the legacy snapshot

**Files:**
- Create: `tests/fixtures/legacy_pdf_to_markdown.py` (byte copy of current `pdf_to_markdown.py`)
- Create: `tests/fixtures/legacy_console.py` (byte copy of current `console.py`)

The legacy snapshot is what we'll compare against. After the refactor, `pdf_to_markdown.py` becomes a shim — the snapshot preserves the *current* behavior so the migration test stays meaningful.

- [ ] **Step 1: Copy the legacy files into fixtures**

```bash
cp pdf_to_markdown.py tests/fixtures/legacy_pdf_to_markdown.py
cp console.py tests/fixtures/legacy_console.py
```

- [ ] **Step 2: Patch the legacy file's import of `console` to use the local copy**

Open `tests/fixtures/legacy_pdf_to_markdown.py` and find the line `from console import (...)` (around line 59). Replace `from console import` with `from tests.fixtures.legacy_console import`. The full block should now read:

```python
try:
    from tests.fixtures.legacy_console import (
        console, ConversionProgress, print_header, print_conversion_report as rich_print_report,
        print_batch_summary, print_success, print_warning, print_error, suppress_docling_logging
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
```

- [ ] **Step 3: Verify the legacy snapshot still runs**

Run: `./venv/bin/python -c "from tests.fixtures import legacy_pdf_to_markdown as legacy; print(legacy.convert_pdf_to_markdown.__doc__[:60])"`
Expected: prints the first 60 characters of the docstring (`Convert a PDF file to markdown format using Docling.`).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/legacy_pdf_to_markdown.py tests/fixtures/legacy_console.py
git commit -m "Snapshot pre-refactor pdf_to_markdown.py as legacy fixture"
```

---

## Task 4: Write the migration test (failing)

**Files:**
- Create: `tests/test_pdf.py`

This test runs the same PDF through both code paths and asserts byte-identical markdown. Right now it will fail because `convert.py` doesn't exist yet — that's the point.

- [ ] **Step 1: Write the test**

Create `tests/test_pdf.py`:

```python
"""PDF conversion tests, including the byte-identical migration test that
guards stage 1's "zero behavior change" promise."""
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample.pdf"


def _run_legacy(tmpdir: Path) -> Path:
    """Run the frozen legacy code path against the fixture, return md path."""
    from tests.fixtures import legacy_pdf_to_markdown as legacy

    out = tmpdir / "legacy.md"
    logger = logging.getLogger("legacy_test")
    logger.addHandler(logging.NullHandler())
    legacy.convert_pdf_to_markdown(
        str(FIXTURE),
        output_path=str(out),
        page_markers=True,
        report=False,
        quiet=True,
        logger=logger,
    )
    return out


def _run_new(tmpdir: Path) -> Path:
    """Run the new convert.py CLI against the fixture, return md path."""
    out = tmpdir / "new.md"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "convert.py"),
            str(FIXTURE),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"convert.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    return out


def test_migration_byte_identical(tmp_path):
    """Stage 1's exit criterion: convert.py output must be byte-identical
    to the frozen legacy pdf_to_markdown.py output for the sample fixture."""
    legacy_md = _run_legacy(tmp_path)
    new_md = _run_new(tmp_path)

    legacy_bytes = legacy_md.read_bytes()
    new_bytes = new_md.read_bytes()

    if legacy_bytes != new_bytes:
        # Surface a useful diff snippet
        legacy_lines = legacy_bytes.decode("utf-8", errors="replace").splitlines()
        new_lines = new_bytes.decode("utf-8", errors="replace").splitlines()
        import difflib
        diff = "\n".join(difflib.unified_diff(legacy_lines, new_lines, lineterm="", n=2)[:60])
        pytest.fail(f"Output diverged:\n{diff}")
```

- [ ] **Step 2: Run the test — expect it to fail**

Run: `./venv/bin/python -m pytest tests/test_pdf.py -v`
Expected: FAIL — `convert.py` does not exist, so `_run_new` will hit a "No such file or directory" or non-zero exit. This is the correct state at this step (TDD red).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pdf.py
git commit -m "Add failing migration test for stage 1 PDF refactor"
```

---

## Task 5: Create `materials/core/base.py`

**Files:**
- Create: `materials/__init__.py`
- Create: `materials/core/__init__.py`
- Create: `materials/core/base.py`

Tasks 5, 6, 7 are independent — they can be dispatched in parallel by a subagent driver.

- [ ] **Step 1: Create empty package `__init__.py` files**

```bash
mkdir -p materials/core materials/formats
touch materials/__init__.py materials/core/__init__.py materials/formats/__init__.py
```

- [ ] **Step 2: Write `materials/core/base.py`**

```python
"""Base types for converter modules.

Each format-specific converter (PDF, DOCX, PPTX, HTML) subclasses BaseConverter
and consumes a ConversionOptions, producing a ConversionResult. Stage 1
introduces this abstraction; stages 2-4 add subclasses.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConversionOptions:
    """Options that apply to any conversion. Format-specific extras live here too;
    each converter ignores options it doesn't understand. Lean default = the AI-
    ingestion case from the spec."""

    output_path: Optional[str] = None
    page_markers: bool = True
    pages: Optional[List[int]] = None
    ocr: bool = False
    extract_images: bool = False
    quiet: bool = False
    verbose: bool = False
    save_report: bool = False
    log_path: Optional[str] = None
    # Format-specific (used in later stages — declared here so options are uniform)
    full: bool = False  # DOCX: include comments appendix
    show_revisions: bool = False  # DOCX: render tracked changes
    keep_images: bool = False  # DOCX: extract images to disk
    notes_only: bool = False  # PPTX: emit speaker-notes transcript only
    strip_html_noise: bool = False  # HTML: bs4 nav/script pre-strip


@dataclass
class ConversionResult:
    """Outcome of a single-file conversion."""

    status: str  # "success" or "error"
    output_file: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    conversion_time: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_legacy(cls, report: Dict[str, Any]) -> "ConversionResult":
        """Adapt the legacy convert_pdf_to_markdown report dict into this shape."""
        return cls(
            status=report.get("status", "error"),
            output_file=report.get("output_file"),
            statistics=report.get("statistics", {}) or {},
            conversion_time=report.get("conversion_time", 0.0),
            error=report.get("error"),
        )


class BaseConverter(ABC):
    """Abstract base for format converters. One subclass per supported format."""

    extensions: tuple[str, ...] = ()  # e.g., (".pdf",) — used for dispatch

    @abstractmethod
    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        """Convert one input file. Implementations must respect options.output_path
        (writing markdown there) and return a ConversionResult."""

    def supports(self, path: str | Path) -> bool:
        """True iff this converter handles the file's extension."""
        ext = Path(path).suffix.lower()
        return ext in self.extensions
```

- [ ] **Step 3: Smoke test the imports**

Run: `./venv/bin/python -c "from materials.core.base import BaseConverter, ConversionOptions, ConversionResult; print('ok')"`
Expected: prints `ok` with no errors.

- [ ] **Step 4: Commit**

```bash
git add materials/__init__.py materials/core/__init__.py materials/core/base.py materials/formats/__init__.py
git commit -m "Add materials.core.base with BaseConverter, ConversionOptions, ConversionResult"
```

---

## Task 6: Create `materials/core/output.py`

**Files:**
- Create: `materials/core/output.py`

This holds path utilities and the heading-text sanitization rule from spec §5.2. The sanitize function isn't used in stage 1 (no section markers for PDF), but landing it here keeps stage 2/3 cleanly additive.

- [ ] **Step 1: Write `materials/core/output.py`**

```python
"""Output path helpers and shared formatting utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Spec §5.2 — sanitize heading text before embedding in <!-- Section K: ... --> markers.
_DASH_RUN = re.compile(r"-{2,}")
_WS_RUN = re.compile(r"\s+")
_MAX_HEADING_LEN = 80


def sanitize_heading_text(text: str) -> str:
    """Make a heading safe for inclusion inside an HTML comment marker.

    Rules:
      1. Collapse runs of two-or-more `-` to a single `-`.
      2. Strip backticks.
      3. Strip newlines/tabs and collapse internal whitespace.
      4. Truncate to 80 characters (append U+2026 if truncated).
      5. If empty after sanitization, return "(untitled)".
    """
    s = text.replace("`", "")
    s = s.replace("\n", " ").replace("\t", " ")
    s = _WS_RUN.sub(" ", s).strip()
    s = _DASH_RUN.sub("-", s)
    if len(s) > _MAX_HEADING_LEN:
        s = s[:_MAX_HEADING_LEN].rstrip() + "…"
    if not s:
        return "(untitled)"
    return s


def default_output_path(input_path: str, override: Optional[str]) -> Path:
    """Stage 1 PDF behavior preserved exactly: if no override, write to
    `<input_dir>/converted/<input_stem>.md`."""
    if override:
        return Path(override)
    src = Path(input_path)
    out_dir = src.parent / "converted"
    out_dir.mkdir(exist_ok=True)
    return out_dir / src.with_suffix(".md").name


def default_log_path(output_location: Path) -> Path:
    """Stage 1 PDF behavior preserved exactly: log file lives next to the markdown."""
    output_location.mkdir(parents=True, exist_ok=True)
    return output_location / "conversion.log"
```

- [ ] **Step 2: Smoke test**

Run:

```bash
./venv/bin/python -c "
from materials.core.output import sanitize_heading_text
assert sanitize_heading_text('Why we use --no-foo') == 'Why we use -no-foo', sanitize_heading_text('Why we use --no-foo')
assert sanitize_heading_text('') == '(untitled)'
assert sanitize_heading_text('a' * 100).endswith('…')
assert sanitize_heading_text('foo \`bar\`') == 'foo bar'
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add materials/core/output.py
git commit -m "Add materials.core.output with sanitize_heading_text and path helpers"
```

---

## Task 7: Create `materials/core/verify.py`

**Files:**
- Create: `materials/core/verify.py`

Cheap-check primitives shared across formats per spec §6.1. PDF will use these in Phase B.

- [ ] **Step 1: Write `materials/core/verify.py`**

```python
"""Cheap, format-agnostic verification primitives.

Spec §6.1 (cheap checks always run) and §6.4 (failure semantics).
Format-specific deep checks live in the per-format modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""


@dataclass
class VerifyReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def overall(self) -> str:
        if any(r.status == "FAIL" for r in self.results):
            return "FAIL"
        if any(r.status == "WARN" for r in self.results):
            return "WARN"
        return "PASS"


def check_non_empty(markdown: str) -> CheckResult:
    """Output must not be empty. Empty output is a FAIL — caller should not write."""
    if not markdown.strip():
        return CheckResult("non_empty", "FAIL", "Output markdown is empty")
    return CheckResult("non_empty", "PASS", f"{len(markdown):,} chars")


def check_word_retention(source_words: int, output_words: int, min_ratio: float) -> CheckResult:
    """Word retention ratio must be at or above the format's minimum band.
    Below band = WARN (write the file, flag it). Empty source = N/A."""
    if source_words == 0:
        return CheckResult(
            "word_retention",
            "PASS",
            "source has no extractable words (skipping)",
        )
    ratio = output_words / source_words
    detail = f"{output_words:,} / {source_words:,} = {ratio:.0%} (min {min_ratio:.0%})"
    if ratio < min_ratio:
        return CheckResult("word_retention", "WARN", detail)
    return CheckResult("word_retention", "PASS", detail)


def count_words(text: str) -> int:
    """Whitespace-delimited word count. Used by every format's cheap verifier."""
    return len(text.split())
```

- [ ] **Step 2: Smoke test**

Run:

```bash
./venv/bin/python -c "
from materials.core.verify import check_non_empty, check_word_retention, count_words
assert check_non_empty('').status == 'FAIL'
assert check_non_empty('hi').status == 'PASS'
assert count_words('one two three') == 3
assert check_word_retention(100, 80, 0.75).status == 'PASS'
assert check_word_retention(100, 50, 0.75).status == 'WARN'
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add materials/core/verify.py
git commit -m "Add materials.core.verify with cheap-check primitives"
```

---

## Task 8: Create `materials/formats/pdf.py` (Phase A — wraps legacy)

**Files:**
- Create: `materials/formats/pdf.py`

Phase A wraps the existing functions in `pdf_to_markdown.py` without moving them yet. This lets us assemble `convert.py` and verify byte-identical output before touching the legacy file.

- [ ] **Step 1: Write `materials/formats/pdf.py`**

```python
"""PDFConverter — wraps PDF→markdown logic.

Stage 1 Phase A: this module imports from pdf_to_markdown.py (the existing
top-level script, untouched) so we can wire up the new dispatcher and prove
byte-identical output before moving definitions.
Phase B (Task 11) moves the function bodies into this file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Phase A: delegate to the legacy module. Phase B replaces these imports with
# locally-defined functions.
import pdf_to_markdown as _legacy

from materials.core.base import BaseConverter, ConversionOptions, ConversionResult


class PDFConverter(BaseConverter):
    """Convert PDF files to markdown using Docling + PyMuPDF + RapidFuzz."""

    extensions = (".pdf",)

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("pdf_converter")
        report = _legacy.convert_pdf_to_markdown(
            input_path,
            output_path=options.output_path,
            pages=options.pages,
            extract_images=options.extract_images,
            ocr=options.ocr,
            page_markers=options.page_markers,
            quiet=options.quiet,
            logger=logger,
        )
        return ConversionResult.from_legacy(report)

    def convert_directory(
        self,
        input_dir: str,
        output_dir: Optional[str],
        recursive: bool,
        save_report: bool,
        page_markers: bool,
    ) -> None:
        logger = logging.getLogger("pdf_converter")
        _legacy.batch_convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=recursive,
            save_report=save_report,
            page_markers=page_markers,
            logger=logger,
        )


# Convenience accessors for legacy helpers used by convert.py during Phase A.
parse_page_range = _legacy.parse_page_range
setup_logging = _legacy.setup_logging
```

- [ ] **Step 2: Smoke test the import path**

Run: `./venv/bin/python -c "from materials.formats.pdf import PDFConverter; c = PDFConverter(); print(c.supports('foo.pdf'), c.extensions)"`
Expected: prints `True ('.pdf',)`.

- [ ] **Step 3: Commit**

```bash
git add materials/formats/pdf.py
git commit -m "Add PDFConverter (Phase A — wraps legacy pdf_to_markdown)"
```

---

## Task 9: Create `convert.py` CLI

**Files:**
- Create: `convert.py`

The new entry point. Argparse layout mirrors the existing PDF flags exactly so the migration test passes. Stage 2-4 will add per-format flag groups; Stage 1 only registers PDF.

- [ ] **Step 1: Write `convert.py`**

```python
#!/usr/bin/env python3
"""convert.py — unified materials-md CLI.

Auto-detects format from the input file's extension and dispatches to the
appropriate converter. Stage 1 supports PDF only; subsequent stages add
DOCX, PPTX, and HTML.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from materials.core.base import ConversionOptions
from materials.formats.pdf import PDFConverter, parse_page_range, setup_logging

# Extension → converter instance. Stage 2-4 register more entries here.
REGISTRY = {ext: PDFConverter() for ext in PDFConverter.extensions}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert documents to markdown optimized for AI ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s chapter1.pdf -o chapter1.md\n"
            "  %(prog)s ./casebooks/ --batch\n"
            "  %(prog)s casebook.pdf --pages 1-50 -o excerpt.md\n"
        ),
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output markdown file or directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch convert all supported files in a directory")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into subdirectories (with --batch)")
    parser.add_argument("--pages",
                        help='Page range (e.g., "1-10" or "1,3,5-8"). PDF only. 1-indexed.')
    parser.add_argument("--images", action="store_true",
                        help="Extract embedded images (PDF only in stage 1)")
    parser.add_argument("--ocr", action="store_true",
                        help="OCR for scanned PDFs (slow; PDF only)")
    parser.add_argument("--no-page-markers", dest="page_markers", action="store_false",
                        help="Disable position markers (enabled by default)")
    parser.add_argument("--save-report", action="store_true",
                        help="Save detailed conversion report JSON (batch mode)")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose (DEBUG) logging")
    return parser


def _dispatch_single(input_path: str, args: argparse.Namespace) -> int:
    ext = Path(input_path).suffix.lower()
    converter = REGISTRY.get(ext)
    if converter is None:
        supported = ", ".join(sorted(REGISTRY.keys()))
        print(
            f"Error: unsupported extension {ext!r}. Supported: {supported}",
            file=sys.stderr,
        )
        return 2

    pages = parse_page_range(args.pages) if args.pages else None

    options = ConversionOptions(
        output_path=args.output,
        page_markers=args.page_markers,
        pages=pages,
        ocr=args.ocr,
        extract_images=args.images,
        quiet=False,
        verbose=args.verbose,
    )
    result = converter.convert(input_path, options)
    return 0 if result.status == "success" else 1


def _dispatch_batch(input_dir: str, args: argparse.Namespace) -> int:
    # Stage 1 batch is PDF-only — same as legacy.
    converter = PDFConverter()
    converter.convert_directory(
        input_dir,
        output_dir=args.output,
        recursive=args.recursive,
        save_report=args.save_report,
        page_markers=args.page_markers,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return 1

    # Logging setup mirrors the legacy main() exactly so output paths match.
    if args.log_file:
        log_path = args.log_file
    elif args.batch or os.path.isdir(args.input):
        out_loc = Path(args.output) if args.output else Path(args.input) / "converted"
        out_loc.mkdir(parents=True, exist_ok=True)
        log_path = out_loc / "conversion.log"
    else:
        out_loc = Path(args.output).parent if args.output else Path(args.input).parent / "converted"
        out_loc.mkdir(parents=True, exist_ok=True)
        log_path = out_loc / "conversion.log"

    setup_logging(str(log_path), verbose=args.verbose, use_rich=True)

    if args.batch or os.path.isdir(args.input):
        return _dispatch_batch(args.input, args)
    return _dispatch_single(args.input, args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x convert.py`

- [ ] **Step 3: Smoke test**

Run: `./venv/bin/python convert.py --help`
Expected: prints the help text including the example block.

- [ ] **Step 4: Commit**

```bash
git add convert.py
git commit -m "Add convert.py CLI dispatcher (PDF-only in stage 1)"
```

---

## Task 10: Run the migration test (must PASS now)

**Files:**
- (none — this task only verifies)

- [ ] **Step 1: Run the migration test**

Run: `./venv/bin/python -m pytest tests/test_pdf.py -v`
Expected: `test_migration_byte_identical PASSED`. If FAIL, the diff in the failure message tells you what diverged — almost always either a path difference or a logger argument that didn't get threaded through. Fix `convert.py` and re-run; do not commit until green.

- [ ] **Step 2: If PASS, commit a marker**

Nothing to commit code-wise; this is a verification gate. Just confirm green and move on.

---

## Task 11: Move PDF function definitions to `formats/pdf.py` (Phase B)

**Files:**
- Modify: `materials/formats/pdf.py` (add definitions)
- Modify: `pdf_to_markdown.py` (remove the moved functions)

This is the bulk move. The function bodies are unchanged — we're just relocating them. The legacy snapshot at `tests/fixtures/legacy_pdf_to_markdown.py` is the comparison point, so the migration test continues to work.

The functions to move (from `pdf_to_markdown.py` → `materials/formats/pdf.py`):

| Function | Current line range in pdf_to_markdown.py |
|---|---|
| `setup_logging` | 74–127 |
| `print_conversion_report` | 130–162 |
| `convert_pdf_to_markdown` | 165–369 |
| `get_pdf_info` | 372–411 |
| `batch_convert_directory` | 413–676 |
| `parse_page_range` | 678–691 |
| `extract_page_text_with_pymupdf` | 693–744 |
| `get_actual_page_number` | 746–805 |
| `to_roman` | 807–829 |
| `to_letters` | 831–839 |
| `normalize_text` | 841–859 |
| `find_text_position` | 861–963 |
| `get_table_page_mapping` | 966–989 |
| `find_table_in_markdown` | 992–1080 |
| `insert_page_markers_hybrid` | 1083–1284 |
| `insert_page_markers_provenance` | 1286–1421 |
| `add_page_markers` | 1423–1488 |

(Note: line numbers are accurate as of the snapshot in `tests/fixtures/legacy_pdf_to_markdown.py`. If the working `pdf_to_markdown.py` has drifted, use the snapshot's line numbers as canonical.)

- [ ] **Step 1: Replace `materials/formats/pdf.py` entirely with the moved bodies**

Build the new `materials/formats/pdf.py` by:

1. Keep the docstring (update Phase A → Phase B note).
2. Keep the imports from `pdf_to_markdown` REMOVED — they're now local.
3. Keep the existing `PDFConverter` class.
4. Inline the moved functions above the class.

The structure:

```python
"""PDFConverter and PDF helpers.

Stage 1 Phase B: function definitions live here. Phase A's import-from-legacy
shim is replaced; pdf_to_markdown.py becomes a deprecation shim (Task 12).
"""
from __future__ import annotations

# === Stdlib ===
import argparse  # used by some legacy helpers' messages
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# === Third-party (kept identical to the snapshot) ===
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False

try:
    from console import (
        console, ConversionProgress, print_header,
        print_conversion_report as rich_print_report,
        print_batch_summary, print_success, print_warning, print_error,
        suppress_docling_logging,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# === Module state preserved from the snapshot ===
_WHITESPACE_PATTERN = re.compile(r'\s+')
_normalized_text_cache = {}


# === Materials types (unchanged from Phase A) ===
from materials.core.base import BaseConverter, ConversionOptions, ConversionResult


# ============================================================================
# === BEGIN: bodies moved verbatim from pdf_to_markdown.py (do NOT modify) ===
# ============================================================================
# The 17 functions below are pasted unchanged from
# tests/fixtures/legacy_pdf_to_markdown.py lines 74-1488, in this exact order:
#
#   setup_logging, print_conversion_report, convert_pdf_to_markdown,
#   get_pdf_info, batch_convert_directory, parse_page_range,
#   extract_page_text_with_pymupdf, get_actual_page_number, to_roman,
#   to_letters, normalize_text, find_text_position, get_table_page_mapping,
#   find_table_in_markdown, insert_page_markers_hybrid,
#   insert_page_markers_provenance, add_page_markers.
#
# Source of truth is tests/fixtures/legacy_pdf_to_markdown.py (frozen in
# Task 3); do NOT use working pdf_to_markdown.py as the source for this
# move because Task 12 will overwrite it.
# ============================================================================

# ============================================================================
# === END: bodies moved verbatim ===
# ============================================================================


class PDFConverter(BaseConverter):
    """Convert PDF files to markdown using Docling + PyMuPDF + RapidFuzz."""

    extensions = (".pdf",)

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("pdf_converter")
        report = convert_pdf_to_markdown(
            input_path,
            output_path=options.output_path,
            pages=options.pages,
            extract_images=options.extract_images,
            ocr=options.ocr,
            page_markers=options.page_markers,
            quiet=options.quiet,
            logger=logger,
        )
        return ConversionResult.from_legacy(report)

    def convert_directory(
        self,
        input_dir: str,
        output_dir: Optional[str],
        recursive: bool,
        save_report: bool,
        page_markers: bool,
    ) -> None:
        logger = logging.getLogger("pdf_converter")
        batch_convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=recursive,
            save_report=save_report,
            page_markers=page_markers,
            logger=logger,
        )
```

The mechanical move:

1. Open `tests/fixtures/legacy_pdf_to_markdown.py` (the frozen snapshot — NOT the working `pdf_to_markdown.py`, since Task 12 will overwrite that).
2. Select lines 74–1488 (every function listed in the table above).
3. Copy.
4. In `materials/formats/pdf.py`, paste between the BEGIN/END marker comments.
5. Do not modify any function body. Do not reorder functions.

After paste, `materials/formats/pdf.py` should contain (in order from top): docstring, stdlib imports, third-party imports, module state (`_WHITESPACE_PATTERN`, `_normalized_text_cache`), the materials-types import, the BEGIN comment block, the 17 pasted function bodies, the END comment block, then `class PDFConverter`.

- [ ] **Step 2: Verify the moved file imports cleanly**

Run: `./venv/bin/python -c "from materials.formats.pdf import PDFConverter, convert_pdf_to_markdown, add_page_markers, parse_page_range; print('ok')"`
Expected: `ok` with no errors.

- [ ] **Step 3: Strip the moved bodies from `pdf_to_markdown.py`**

Open `pdf_to_markdown.py` and delete lines 74–1488 (everything between the imports and `def main():`). The file should keep:
- The shebang and docstring (lines 1–22)
- The imports (lines 24–71) — **leave these alone for now**; Task 12 replaces the file entirely
- `def main():` and `if __name__ == "__main__":` (lines 1490–1632) — **leave these for now**; Task 12 replaces them

After this edit, `pdf_to_markdown.py` will be syntactically broken (`main()` calls functions that no longer exist locally). That's fine — Task 12 fixes it.

- [ ] **Step 4: Verify symbols resolved correctly after the move**

`convert.py` does not need editing — it already imports `parse_page_range` and `setup_logging` from `materials.formats.pdf`, which now defines them locally instead of re-exporting from `pdf_to_markdown`.

Run: `./venv/bin/python -c "from materials.formats.pdf import parse_page_range, setup_logging, convert_pdf_to_markdown; print('ok')"`
Expected: `ok`. If `ImportError`, a function body got missed during the paste — diff `tests/fixtures/legacy_pdf_to_markdown.py` against `materials/formats/pdf.py`.

- [ ] **Step 5: Run the migration test — must still PASS**

Run: `./venv/bin/python -m pytest tests/test_pdf.py -v`
Expected: `test_migration_byte_identical PASSED`. The legacy snapshot still has the bodies; the new code now also has them; both produce identical output. If FAIL, you missed a function — diff `tests/fixtures/legacy_pdf_to_markdown.py` against `materials/formats/pdf.py` to find the omission.

- [ ] **Step 6: Commit**

```bash
git add materials/formats/pdf.py pdf_to_markdown.py
git commit -m "Move PDF function definitions into materials/formats/pdf.py (Phase B)"
```

---

## Task 12: Replace `pdf_to_markdown.py` with the deprecation shim

**Files:**
- Modify: `pdf_to_markdown.py` (replace entirely)

- [ ] **Step 1: Overwrite `pdf_to_markdown.py` with the shim**

Replace the entire file with exactly this content:

```python
#!/usr/bin/env python3
"""Deprecation shim — forwards to convert.py.

This file existed before stage 1 of the materials-md refactor and is kept
for backwards compatibility for one deprecation cycle. It will be removed
in stage 5. New work should call convert.py directly.
"""
import sys

import convert

print(
    "[DEPRECATED] use convert.py — pdf_to_markdown.py will be removed in stage 5",
    file=sys.stderr,
)
sys.argv[0] = "convert.py"
sys.exit(convert.main())
```

- [ ] **Step 2: Verify the shim works**

Run: `./venv/bin/python pdf_to_markdown.py --help 2>&1 | head -5`
Expected: stderr line `[DEPRECATED] use convert.py — pdf_to_markdown.py will be removed in stage 5`, followed by the convert.py help text.

- [ ] **Step 3: Run the migration test once more**

Run: `./venv/bin/python -m pytest tests/test_pdf.py -v`
Expected: `test_migration_byte_identical PASSED`. The legacy snapshot is intact; convert.py is intact; nothing should have broken.

- [ ] **Step 4: Run a real PDF conversion as a manual smoke test**

Run: `./venv/bin/python convert.py tests/fixtures/sample.pdf -o /tmp/manual-smoke.md`
Then: `head -20 /tmp/manual-smoke.md`
Expected: the markdown should contain "Preface", "Chapter 1", and `<!-- Page i -->` / `<!-- Page 1 -->` markers from the fixture.

- [ ] **Step 5: Commit**

```bash
git add pdf_to_markdown.py
git commit -m "Replace pdf_to_markdown.py with deprecation shim"
```

---

## Task 13: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

Per spec §11, every stage updates CLAUDE.md.

- [ ] **Step 1: Read the current CLAUDE.md to find the sections that lie**

Run: `grep -n "pdf_to_markdown.py\|main converter\|three Python files" CLAUDE.md`
Expected: locates the architecture sentence ("The codebase is three Python files plus a Rich helper module") and the "main script invocation" lines.

- [ ] **Step 2: Update the "Environment" section**

In `CLAUDE.md`, find the "Environment" code block that lists invocations. Replace:

```bash
./venv/bin/python pdf_to_markdown.py ...      # main converter
./venv/bin/python verify_conversion.py ...    # output verifier
./venv/bin/python verify_page_markers.py ...  # page-marker accuracy check
```

with:

```bash
./venv/bin/python convert.py ...              # main converter (PDF in stage 1; DOCX/PPTX/HTML in later stages)
./venv/bin/python pdf_to_markdown.py ...      # DEPRECATED — forwards to convert.py; removed in stage 5
./venv/bin/python verify_conversion.py ...    # output verifier (consolidated into verify_cli.py in stage 5)
./venv/bin/python verify_page_markers.py ...  # page-marker accuracy check (consolidated in stage 5)
./venv/bin/python -m pytest tests/            # test suite (new in stage 1)
```

- [ ] **Step 3: Update the "Architecture" section**

Find "The codebase is three Python files plus a Rich helper module. The interesting logic is concentrated in `pdf_to_markdown.py`."

Replace with:

```markdown
The codebase is being migrated from a single-file PDF converter into a multi-format
package under `materials/`. The current state (post-stage-1) is:

- `convert.py` — CLI entry point. Auto-detects format from extension and dispatches.
- `materials/core/` — shared types and utilities. `base.py` (BaseConverter ABC,
  ConversionOptions, ConversionResult), `output.py` (path helpers and
  `sanitize_heading_text`), `verify.py` (cheap-check primitives).
- `materials/formats/pdf.py` — all PDF logic (formerly in `pdf_to_markdown.py`).
- `pdf_to_markdown.py` — deprecation shim only; removed in stage 5.
- `console.py` — Rich UX helpers (unchanged).
- `verify_conversion.py`, `verify_page_markers.py` — verification scripts
  (consolidated into `verify_cli.py` in stage 5).
- `tests/` — pytest test suite. `tests/fixtures/build/` holds scripted fixture
  builders; `tests/fixtures/legacy_pdf_to_markdown.py` is a frozen snapshot
  used by the migration test.
```

- [ ] **Step 4: Update the "Page-marker insertion" section header**

Find `### Page-marker insertion (the architecturally non-obvious part)` and update the prose to reference `materials/formats/pdf.py` instead of `pdf_to_markdown.py`. Specifically: change "`add_page_markers` is the entry point" to "`materials.formats.pdf.add_page_markers` is the entry point". The function logic itself is unchanged; only the path moved.

- [ ] **Step 5: Add a "Tests" subsection at the end of the Architecture section**

Append:

```markdown
### Tests

`pytest tests/` runs the suite. The load-bearing test is
`tests/test_pdf.py::test_migration_byte_identical`, which guarantees that
`convert.py` produces output byte-equal to the frozen legacy snapshot at
`tests/fixtures/legacy_pdf_to_markdown.py`. Every refactor that touches PDF
logic must keep this test green.

Fixtures are scripted — every binary fixture has a builder under
`tests/fixtures/build/` so they can be regenerated deterministically.
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md to reflect stage 1 architecture"
```

---

## Task 14: Final stage-1 verification + tag

**Files:**
- (none — final verification)

- [ ] **Step 1: Run the full test suite**

Run: `./venv/bin/python -m pytest tests/ -v`
Expected: 1 test passes (`test_migration_byte_identical`). Zero failures.

- [ ] **Step 2: Run a real PDF through both entry points**

Run:
```bash
./venv/bin/python convert.py tests/fixtures/sample.pdf -o /tmp/via-convert.md
./venv/bin/python pdf_to_markdown.py tests/fixtures/sample.pdf -o /tmp/via-shim.md 2>/dev/null
diff /tmp/via-convert.md /tmp/via-shim.md
```
Expected: `diff` produces no output (the shim and direct call produce identical results).

- [ ] **Step 3: Tag stage 1 (optional but useful for stage 2's PR baseline)**

Run: `git tag stage-1-pdf-refactor`

This is a local tag; do not push without user confirmation.

---

## Self-review (run after all tasks complete)

**Spec coverage:**
- §5.1 module layout for PDF — ✓ Task 5-9
- §5.2 sanitize_heading_text — ✓ Task 6
- §6.1 cheap-check primitives — ✓ Task 7 (PDF doesn't use them at runtime in stage 1; stages 2-4 do)
- §7 test infrastructure — ✓ Task 1, 2, 3, 4
- §8 stage 1 deliverables — ✓ all tasks
- §11 CLAUDE.md update rule — ✓ Task 13
- Migration test exists — ✓ Task 4

**Stage 1 deliverables not in this plan:** none. Stages 2-6 each get their own plan after stage 1 merges.

---

## Parallelization opportunities

For the subagent-driven-development driver:

- **Tasks 5, 6, 7 are independent** — no shared files, no shared types between them (Task 5 declares the types Task 6/7 may use, but Tasks 6 and 7 don't depend on each other and Task 5's interface is fully specified above). Dispatch all three in parallel.
- **Tasks 8 and 9 depend on Task 5** but not on Tasks 6/7. Dispatch after Task 5 returns.
- **Task 11 (the move) cannot be parallelized** — it's a single mechanical operation on two files.
- **Tasks 13 (CLAUDE.md) is independent** of Tasks 5-12 and can be drafted in parallel with later tasks, though committed last.

All other tasks are sequential.
