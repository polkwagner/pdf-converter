# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A PDF→markdown converter built around **Docling** (IBM Research), with custom logic for accurate page-number markers, batch optimization, and Rich-based terminal UX. Primary use case: converting legal casebooks and academic PDFs to LLM-ingestible markdown. See `README.md` for user-facing docs.

## Environment

The project runs in a venv at `./venv/`. There is no `pip install -e .` — scripts are invoked directly:

```bash
./venv/bin/python convert.py ...              # main converter (PDF, DOCX, PPTX, HTML)
./venv/bin/python verify_cli.py ...           # verifier CLI with `content` and `markers` subcommands
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

# DOCX (lean default — comments dropped, footnotes preserved)
./venv/bin/python convert.py memo.docx -o memo.md

# DOCX with reviewer comments appendix
./venv/bin/python convert.py memo.docx --full -o memo-with-comments.md

# DOCX with tracked changes shown inline
./venv/bin/python convert.py memo.docx --show-revisions -o memo-redlined.md

# PPTX (default — slide markers + speaker notes)
./venv/bin/python convert.py deck.pptx -o deck.md

# PPTX as a lecture transcript (notes only, slides without notes skipped)
./venv/bin/python convert.py deck.pptx --notes-only -o lecture.md

# Parallel batch (4 worker processes; useful past ~8 large files)
./venv/bin/python convert.py ./casebooks/ --batch --workers 4

# Run the test suite
./venv/bin/python -m pytest tests/

# Verifier (new entry point)
./venv/bin/python verify_cli.py content source.pdf output.md
./venv/bin/python verify_cli.py markers source.pdf output.md
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
- `materials/formats/docx.py` — DOCX converter. Docling for prose +
  python-docx (with lxml) for comments, footnotes, and tracked changes.
  Lean default per spec §4.6: drops comments and accepts revisions, but
  always preserves footnotes as `[^N]` markdown footnotes. Three opt-in
  flags: `--full` (comments appendix), `--show-revisions` (inline ins/del),
  `--keep-images` (extract images to disk).
- `materials/formats/pptx.py` — PPTX converter. Pure Docling pipeline;
  Docling 2.65.0+ extracts speaker notes into `ContentLayer.FURNITURE`
  natively, so no extra dependency. Each slide gets a `<!-- Slide K -->`
  marker; notes appear inline at slide-end with `<!-- Speaker notes -->`.
  The `--notes-only` flag produces a clean lecture transcript (slide
  numbers + notes text only, skipping bullet content and slides without
  notes).
- `console.py` — Rich UX helpers (unchanged).
- `verify_conversion.py`, `verify_page_markers.py` — thin deprecation shims
  that forward to `verify_cli.py`. Function bodies are still importable;
  the CLI invocations print a deprecation warning and forward.
- `verify_cli.py` — consolidated verifier with `content` and `markers` subcommands.
- `materials/core/parallel.py` — `parallel_convert_files` for ProcessPoolExecutor
  batch conversion. Workers re-import the converter class lazily; per-worker
  DocumentConverter warmup is the trade-off cost. Serial path wins on small
  batches; parallelism wins past ~8 large files.
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

### DOCX conversion pipeline (`materials/formats/docx.py`)

DOCX conversion uses two parallel reads:

1. **python-docx + lxml** opens the .docx archive directly to extract
   auxiliary content the Word XML carries: `comments.xml` (top-level
   comments only — threaded replies in `commentsExtended.xml` are a known
   limitation, deferred), `footnotes.xml`, and `<w:ins>` / `<w:del>`
   tracked-changes elements in the body.
2. **Docling** runs its DOCX pipeline on the same file to produce the
   prose markdown (headings, paragraphs, tables).

Empirical Docling behavior worth knowing:
- Docling drops footnotes entirely from DOCX output. The converter extracts
  them from `footnotes.xml` and appends a `## Footnotes` section.
- Docling drops both `<w:ins>` and `<w:del>` runs. The converter
  re-surfaces insertions as plain prose in lean mode (accepted-final
  behavior) or with `[+ ... +]` markers in `--show-revisions`. Deletions
  appear only in `--show-revisions` mode, surfaced with `[- ... -]` markers
  near the end of the output.
- Docling shifts heading levels by one (source H1 → `##` markdown,
  source H2 → `###`). Section markers auto-detect the two smallest
  heading levels actually present, which is functionally equivalent to
  "H1 and H2 in the source" without hardcoding the level offset.

Post-processing layers atop the Docling markdown:

- **Footnotes** — always preserved, appended as a `## Footnotes` section.
- **Comments** — dropped by default. With `--full`, a `## Reviewer Comments`
  appendix lists each comment with author, date, body, and quoted referenced
  text. Best-effort `[C1]` inline anchors are inserted where the referenced
  phrase still appears in the markdown.
- **Tracked changes** — accepted-final by default. With `--show-revisions`,
  insertions wrapped in `[+ ... +]` and deletions surfaced as `[- ... -]`.
- **Images** — replaced with `<!-- image -->` placeholders. With
  `--keep-images`, extracted to disk and referenced via `![](path)` markdown.
- **Section markers** — numbered `<!-- Section K: heading -->` markers via
  `core/output.py::sanitize_heading_text`, with auto-detected heading levels.

Cheap verifier requires ≥90% word retention (DOCX is text-rich; lower
ratios indicate Docling lost meaningful content).

### PPTX conversion pipeline (`materials/formats/pptx.py`)

PPTX conversion is the simplest of the four formats because Docling does
most of the heavy lifting:

1. **Docling convert** produces a `DoclingDocument` with each slide as a
   page. Speaker notes are tagged `content_layer == ContentLayer.FURNITURE`
   and excluded from the default markdown export.
2. **Per-slide split** uses provenance: each item's `prov[0].page_no`
   maps it to a slide. (The `page_break_placeholder` parameter on
   `export_to_markdown` is unreliable for PPTX — Docling 2.65.0 emits
   only N−2 break tokens for an N-slide deck — so we use provenance as
   the primary split strategy with placeholder and `^# ` heading split
   as fallbacks.)
3. **Notes extraction** walks `iterate_items(included_content_layers={FURNITURE})`
   and groups text by slide via `prov[0].page_no`.
4. **Three rendering modes:**
   - **Default** — `<!-- Slide K -->` + body + `<!-- Speaker notes -->` + notes (when notes exist).
   - **`--notes-only`** — clean lecture transcript: only slides with notes contribute, and only the slide marker + notes text appears. No bullet content. Useful for repurposing a deck as prose for an article or LLM ingestion.
   - **`--no-page-markers`** — strip both `<!-- Slide K -->` and `<!-- Speaker notes -->` markers; just the prose.

Cheap verifier: ≥75% word retention against the full BODY+FURNITURE
markdown (the retention check is skipped in `--notes-only` mode because
the transcript intentionally drops body content).

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

### Verification (`verify_cli.py`)

The consolidated verifier provides two subcommands:

- `verify_cli.py content` — coarse sanity check. Compares source and markdown by page count, word/char retention ratio, table presence, image-heavy page detection. Pass/warn/fail thresholds vary by format (90% for DOCX, 75% for PPTX, 60% for HTML).
- `verify_cli.py markers` — fine-grained page/slide marker audit (PDF and PPTX only). Samples markers, extracts surrounding text, fuzzy-matches against the source using PyMuPDF/python-pptx, reports a hit rate. Use this when you suspect provenance is misfiring on a specific corpus.

Both support `--batch` mode: walk a directory of `.md` outputs against a parallel directory of source files.

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

## Claude Code skill wrapper

A natural-language wrapper for the converter lives at
`~/.claude/skills/materials-md/SKILL.md`. The skill is a thin dispatcher —
it maps natural-language requests like "convert this deck" or "extract the
speaker notes" to `convert.py` flags and shells out to the Python tool.
**No conversion logic in the skill itself**; everything goes through the
deterministic Python converter.

The skill is upstream of every text-consuming skill: `convert → eddie`,
`convert → polk-document`, `convert → factual-pipeline-orchestrator`, and
`convert (--notes-only) → polk-slides`. When a downstream skill needs
markdown input from a non-markdown source, this skill is the entry point.

## Repo state notes

- `.gitignore` excludes `*.md` except `README.md` and `CLAUDE.md`, so any conversion output written to the repo root is gitignored by default — handy for ad-hoc testing without polluting `git status`.
- `conversion.log` accumulates across runs and is gitignored. Delete or rotate it if it grows unwieldy.
- `__pycache__/` is regenerated automatically; safe to delete any time.
