# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A PDF→markdown converter built around **Docling** (IBM Research), with custom logic for accurate page-number markers, batch optimization, and Rich-based terminal UX. Primary use case: converting legal casebooks and academic PDFs to LLM-ingestible markdown. See `README.md` for user-facing docs.

## Environment

The project runs in a venv at `./venv/`. There is no `pip install -e .` — scripts are invoked directly:

```bash
./venv/bin/python convert.py ...              # main converter (PDF in stage 1; DOCX/PPTX/HTML in later stages)
./venv/bin/python pdf_to_markdown.py ...      # DEPRECATED — forwards to convert.py; removed in stage 5
./venv/bin/python verify_conversion.py ...    # output verifier (consolidated into verify_cli.py in stage 5)
./venv/bin/python verify_page_markers.py ...  # page-marker accuracy check (consolidated in stage 5)
./venv/bin/python -m pytest tests/            # test suite (new in stage 1)
```

If you create the venv from scratch:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

There is no linter config and no build step. `requirements.txt` pins minimum versions only (`docling>=2.0.0`, etc.) — the actual versions in the venv are 2.65.0 / 1.26.7 / 3.14.3 / 14.2.0 as of last update. A pytest test suite was added in stage 1 (see the Tests subsection under Architecture).

## Common commands

```bash
# Single file
./venv/bin/python convert.py input.pdf -o output.md

# Batch (reuses the Docling ML model across files — 3-5x faster)
./venv/bin/python convert.py ./pdfs/ --batch

# Page range
./venv/bin/python convert.py book.pdf --pages 1-50 -o ch1.md

# OCR for scanned PDFs (slow)
./venv/bin/python convert.py scan.pdf --ocr

# Disable page markers
./venv/bin/python convert.py doc.pdf --no-page-markers -o out.md

# HTML conversion (auto-detected by extension)
./venv/bin/python convert.py article.html -o article.md

# HTML with noise stripping
./venv/bin/python convert.py article.html --strip-html-noise -o article.md

# Run the test suite
./venv/bin/python -m pytest tests/

# Verify a single conversion (compares pdf↔md word/char/page counts)
./venv/bin/python verify_conversion.py source.pdf output.md

# Audit page-marker accuracy on a sample
./venv/bin/python verify_page_markers.py source.pdf output.md
```

Defaults worth knowing:
- Page markers are **on** by default. Output looks like `<!-- Page N -->` interleaved at page boundaries.
- Batch mode writes to `<input_dir>/converted/` if `-o` isn't given.
- Single-file mode writes a sibling `converted/` subfolder for the log file even when `-o` points elsewhere.
- The `conversion.log` file accumulates session entries (DEBUG with `-v`).

## Architecture

The codebase is being migrated from a single-file PDF converter into a multi-format
package under `materials/`. The current state (post-stage-2, plus a code-review fix pass) is:

- `convert.py` — CLI entry point. Auto-detects format from extension and
  dispatches via the `REGISTRY` dict to a `BaseConverter` subclass. Owns
  argparse, path computation, and exit-code propagation. Imports CLI
  infrastructure (`setup_logging`, `RICH_AVAILABLE`) from `materials.core.logging` —
  not from any format module.
- `materials/core/` — shared types and utilities used by every format converter.
  - `base.py` — `BaseConverter` ABC (with `convert` and `convert_directory`),
    `ConversionOptions` dataclass (uniform across formats; format-specific
    fields ignored by converters that don't care), `ConversionResult` with a
    canonical `statistics` schema documented in its docstring.
  - `output.py` — `default_output_path`, `default_log_path`, `sanitize_heading_text`
    (the §5.2 escaping rule).
  - `verify.py` — cheap-check primitives (`check_non_empty`, `check_word_retention`,
    `count_words`, `VerifyReport`).
  - `logging.py` — `setup_logging` and `RICH_AVAILABLE` detection. Lives in
    core because logging is CLI infrastructure, not a format property.
- `materials/formats/pdf.py` — all PDF logic (formerly in `pdf_to_markdown.py`).
  16 verbatim functions moved from the legacy snapshot, plus the `PDFConverter`
  class. `PDFConverter.convert()` wraps the legacy call in try/except so
  `FileNotFoundError` and other exceptions return `ConversionResult(status="error")`
  instead of raising.
- `materials/formats/html.py` — HTML converter. Pure Docling pipeline plus
  optional bs4 noise-stripping (`--strip-html-noise`). Encoding-aware reader
  (BOM detection, `<meta charset>` sniff, cp1252/latin-1 fallback for Word
  HTML exports). First consumer of `core.output.sanitize_heading_text` and
  `core.verify`.
- `pdf_to_markdown.py` — deprecation shim only; removed in stage 5.
- `console.py` — Rich UX helpers (unchanged).
- `verify_conversion.py`, `verify_page_markers.py` — verification scripts
  (consolidated into `verify_cli.py` in stage 5).
- `tests/` — pytest test suite.
  - `tests/fixtures/build/` — scripted fixture builders (PDF and HTML).
  - `tests/fixtures/sample.golden.md` — pinned reference output for the PDF
    migration test. Regenerate when accepting a deliberate Docling upgrade
    or behavior change.
  - `tests/fixtures/legacy_pdf_to_markdown.py` — frozen pre-refactor snapshot;
    cross-checked against the same golden as `convert.py`.

### PDF conversion pipeline (`materials/formats/pdf.py`)

Orchestrates four stages per PDF:

1. **Metadata pre-scan** (`get_pdf_info`) — PyMuPDF reads page count, page labels, and PDF-level metadata. Page labels matter: a casebook may start at page 41 (Chapter II), use Roman numerals for front matter, or have multiple numbering schemes. `get_actual_page_number` translates a 0-indexed page index back to whatever the PDF declares.
2. **Docling conversion** — `DocumentConverter().convert()` produces a `DoclingDocument` with element-level provenance (every paragraph/table/heading knows which page it came from). The document is exported to markdown via `document.export_to_markdown()`.
3. **Page-marker insertion** — see below; this is the part most likely to be fragile.
4. **Output write + report** — markdown saved, stats logged, Rich panel printed.

### HTML conversion pipeline (`materials/formats/html.py`)

HTML conversion is simpler than PDF because Docling handles the markup natively. The pipeline:

1. **Read the file** with encoding-aware fallback: UTF BOM detection (`utf-8-sig`, `utf-16-le`, `utf-16-be`), then `<meta charset>` sniff in the first 4KB, then cp1252 (the dominant Word HTML export encoding), then latin-1 as a last resort. Word smart-quotes and em-dashes survive the round trip.
2. **Optional noise strip** — if `--strip-html-noise` is set, beautifulsoup4 removes `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, and elements whose class matches `sidebar|advert|cookie|consent`. Without the flag, the raw HTML is passed through.
3. **Docling convert** — the cleaned (or raw) HTML is written to a temp file and passed to `DocumentConverter()`. Docling produces markdown.
4. **Section markers** — a regex walks the markdown for `^#` and `^##` lines and inserts numbered `<!-- Section K: heading-text -->` markers before each one. H3+ are not numbered (sectioning happens at H1/H2 only). `core/output.py::sanitize_heading_text` is applied to the heading text.
5. **Cheap verifier** — output non-empty + word retention ratio ≥60% (HTML loses lots of tag overhead, hence the lower minimum).

bs4 is an **optional** dependency. The converter only imports it if `--strip-html-noise` is set; without the flag, bs4 doesn't need to be installed.

### Page-marker insertion (the architecturally non-obvious part)

`materials.formats.pdf.add_page_markers` is the entry point. It tries three strategies in order, falling back if the prior one returns a poor result:

1. **Internal Docling markers.** Some Docling versions emit `#_#_DOCLING_DOC_PAGE_BREAK_<from>_<to>_#_#` tokens directly in the markdown stream. If present, these are converted to `<!-- Page N -->` comments verbatim — by far the most accurate path.
2. **Provenance-based** (`insert_page_markers_provenance`). Walks the Docling element tree, uses each element's `prov[0].page_no` to determine its source page, then locates that element's text in the markdown stream to insert a marker before it. Most common path in practice.
3. **Hybrid PyMuPDF + RapidFuzz fallback** (`insert_page_markers_hybrid`). For elements where provenance is missing or ambiguous, this extracts per-page text directly with PyMuPDF and finds a fuzzy match against the markdown body. Slower but rescues edge cases.

If all three fail, the converter returns markdown without markers rather than with wrong markers — this is intentional ("better no marker than a misplaced one"). Single-page documents get a special case: a `<!-- Page 1 -->` prepended unconditionally.

Common breakage points: changes to Docling's markdown serialization (whitespace, heading levels, table formatting) that desync the text-position search; PDFs whose page labels parse oddly (the script handles Roman, letters, prefixed forms — see `to_roman` / `to_letters`).

### Batch mode

`batch_convert_directory` initializes a single `DocumentConverter` instance and reuses it across all PDFs. This is the only meaningful performance optimization in the codebase, and it's why batch mode runs 3–5× faster than sequential single-file invocations. Don't refactor batch mode to instantiate per file.

### `console.py` — Rich UX layer

All Rich-dependent output (panels, progress bars, spinners, batch summary tables) is isolated here. The main script imports it lazily under a `RICH_AVAILABLE` flag and degrades to plain `logger.info` calls if Rich is missing. `suppress_docling_logging()` silences Docling's stdout chatter so the Rich progress bars aren't shredded.

### `verify_conversion.py` and `verify_page_markers.py`

Two separate verifiers with different scopes:

- `verify_conversion.py` — coarse sanity check. Compares PDF and markdown by page count, word/char retention ratio, table presence, image-heavy page detection. Has a `--batch` mode that walks a directory of `.md` outputs against a parallel directory of `.pdf` sources. Pass/warn/fail thresholds are encoded in `verify_conversion`.
- `verify_page_markers.py` — fine-grained page-marker correctness audit. Samples markers, extracts surrounding text, fuzzy-matches against the corresponding PyMuPDF page, reports a hit rate. Use this when you suspect provenance is misfiring on a specific corpus.

### Tests

`pytest tests/` runs the suite. The load-bearing tests are in
`tests/test_pdf.py`:

- `test_new_matches_golden` — `convert.py` output must equal
  `tests/fixtures/sample.golden.md`, the pinned reference generated on
  Docling 2.65.0 at Stage 1 completion.
- `test_legacy_matches_golden` — the frozen legacy snapshot must produce
  the same golden output. Cross-coverage: catches accidental drift in
  `tests/fixtures/legacy_pdf_to_markdown.py`.
- `test_fixture_unchanged_after_conversion` — regression guard against
  PyMuPDF/Docling mutating the source fixture during read.

When a deliberate Docling upgrade or behavior change produces different
PDF output, regenerate the golden:

```bash
./venv/bin/python convert.py tests/fixtures/sample.pdf -o tests/fixtures/sample.golden.md
```

This turns the Docling-version-coupled identity comparison into an
explicit, reviewable acceptance step.

HTML tests cover section markers, the §5.2 escaping rule (dashes,
backticks), `--strip-html-noise` removing `<nav>`/`<footer>`,
`--no-page-markers` suppressing markers, and the
beautifulsoup4-not-installed error path (simulated via `sys.modules`).

Fixtures are scripted — every binary fixture has a builder under
`tests/fixtures/build/` so they can be regenerated deterministically.

## Repo state notes

- `.gitignore` excludes `*.md` except `README.md` and `CLAUDE.md`, so any conversion output written to the repo root is gitignored by default — handy for ad-hoc testing without polluting `git status`.
- `conversion.log` accumulates across runs and is gitignored. Delete or rotate it if it grows unwieldy.
- `__pycache__/` is regenerated automatically; safe to delete any time.
