# PDF to Markdown Converter

Convert PDF, DOCX, PPTX, and HTML files to markdown format optimized for AI tools, using **Docling** - IBM Research's document understanding library.

## Features

- **Advanced layout analysis** - Preserves document structure with high fidelity
- **Table structure recognition** - Maintains complex table formatting
- **Heading hierarchy detection** - Proper markdown heading levels
- **Formula extraction** - Handles mathematical notation (if present)
- **Accurate page markers** - 96%+ accuracy using Docling's provenance system
- **PDF page label support** - Respects actual page numbers (e.g., Chapter 2 starting at page 41)
- **Rich visual feedback** - Progress bars, spinners, and formatted output panels
- **Optimized batch processing** - Reuses ML models across files for 3-5x faster batch conversion
- **OCR support** - Optional OCR for scanned documents
- **Page range selection** - Extract specific sections
- **HTML support** - Convert articles, web pages, and HTML exports to clean markdown with numbered section markers. Optional `--strip-html-noise` removes navigation, footers, and ad/cookie clutter via beautifulsoup4.
- **DOCX support** - Convert Word documents to clean markdown with section markers. Footnotes always preserved as `[^N]` references; reviewer comments included via `--full`; tracked changes shown via `--show-revisions`.
- **PPTX support** - Convert PowerPoint decks to markdown with numbered slide markers and inline speaker notes. The `--notes-only` flag emits a clean lecture transcript — slide numbers + speaker notes text only — useful for repurposing decks as prose.
- **Optimized for AI** - Output tailored for LLM consumption (Claude, GPT-4, etc.)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single File Conversion

```bash
python convert.py input.pdf -o output.md
```

### Batch Conversion

```bash
# Convert all PDFs in a directory
python convert.py ./casebooks/ --batch

# Recursive conversion (includes subdirectories)
python convert.py ./casebooks/ --batch --recursive

# Save detailed JSON report
python convert.py ./casebooks/ --batch --save-report
```

### Page Range Selection

```bash
# Convert specific pages (1-indexed)
python convert.py casebook.pdf --pages 1-50 -o chapter1.md

# Convert multiple ranges
python convert.py casebook.pdf --pages 1-10,25-35 -o sections.md
```

### Page Number Markers (for AI/RAG)

**✓ NOW FULLY SUPPORTED** for both single-page and multi-page documents!

Page markers are added automatically as HTML comments (`<!-- Page N -->`) throughout the markdown output, preserving the original PDF page numbers. This is essential for legal citations and RAG (Retrieval-Augmented Generation) applications.

```bash
# Page markers enabled by default
python convert.py casebook.pdf -o output.md
# Output includes: <!-- Page 1 -->, <!-- Page 2 -->, etc.

# Disable page markers if needed
python convert.py document.pdf --no-page-markers -o output.md
```

**How it works - Provenance-Based Approach:**

The tool uses Docling's element provenance system for accurate page tracking:

1. **Docling Provenance**: Each element in Docling's output includes the source page number, enabling precise page boundary detection
2. **PDF Page Labels**: Automatically reads PDF metadata for actual page numbering
   - Supports different numbering styles: Arabic (1, 2, 3), Roman (i, ii, iii), Letters (a, b, c)
   - Handles prefixes (e.g., "A-1" for appendices)
   - Supports multiple numbering schemes (e.g., roman numerals for front matter, then arabic for body)
3. **Fallback Hybrid Matching**: For edge cases, falls back to PyMuPDF text extraction with RapidFuzz matching

**Accuracy:**
- Tested on 1,309-page casebook: **98.5% of pages accurately marked** (1,290/1,309)
- Tested on 122-page chapter (pages 41-162): **98% accuracy** (120/122, 2 blank pages)
- Handles complex multi-page tables
- **Full PDF page label support:**
  - ✓ Non-sequential numbering (Chapter 2 starting at page 41)
  - ✓ Roman numerals (i, ii, iii, iv...)
  - ✓ Uppercase/lowercase variants (I/i, A/a)
  - ✓ Letter sequences (a, b, c... z, aa, ab...)
  - ✓ Prefixes (Appendix A-1, A-2...)
  - ✓ Multiple numbering schemes (roman front matter → arabic body → appendix)
- Blank pages are detected and reported (not marked, since no content exists)

### Extract Images

```bash
python convert.py input.pdf --images
```

### OCR for Scanned Documents

```bash
# Enable OCR for image-based PDFs (slower but necessary for scanned docs)
python convert.py scanned_casebook.pdf --ocr
```

### HTML Conversion

```bash
# Convert a single HTML article (auto-detected by extension)
python convert.py article.html -o article.md

# Strip navigation, footers, ads, and cookie banners before conversion
python convert.py article.html --strip-html-noise -o article.md
```

HTML output uses **section markers** instead of page markers: `<!-- Section 1: Title -->`, `<!-- Section 2: First heading -->`, etc. Markers are inserted at H1 and H2 boundaries; H3 and deeper headings appear as ordinary `### ...` markdown without their own markers.

### DOCX Conversion

```bash
# Lean default — comments dropped, footnotes preserved
python convert.py memo.docx -o memo.md

# Include reviewer comments in a `## Reviewer Comments` appendix
python convert.py memo.docx --full -o memo.md

# Render tracked changes inline as [+ added +] / [- removed -]
python convert.py memo.docx --show-revisions -o memo.md

# Extract embedded images to <output>_files/ alongside the markdown
python convert.py memo.docx --keep-images -o memo.md
```

DOCX uses **section markers** (numbered, like HTML). The lean default produces clean prose suited to AI ingestion; opt into the maximalist behavior with `--full` when reviewer-comment workflows matter. Footnotes — common in legal memos — are always preserved as `[^N]` markdown footnotes with a `## Footnotes` section appended at the end.

### PPTX conversion

```bash
# Default — slide markers + speaker notes inline
python convert.py deck.pptx -o deck.md

# Lecture transcript — only speaker notes, slides without notes are skipped
python convert.py deck.pptx --notes-only -o lecture.md

# Just the slide bodies, no markers
python convert.py deck.pptx --no-page-markers -o deck.md
```

PPTX uses `<!-- Slide N -->` markers (numbered, 1-indexed). Speaker notes appear inline at slide-end with a `<!-- Speaker notes -->` marker. The `--notes-only` mode is designed for converting a lecture deck into linear prose — the speaker notes typically *are* the lecture, with the bullets serving as visual cues.

### Batch parallelism

```bash
# Convert a directory of PDFs with 4 worker processes
python convert.py ./casebooks/ --batch --workers 4
```

Each worker process pays a one-time DocumentConverter warmup cost. For small batches (≤ ~8 files), the serial path is faster because the per-worker warmup dominates. For larger batches and large per-file content (legal casebooks, multi-hundred-page documents), parallelism wins. Run `tests/bench/run_bench.py <pdf_dir>` against your own corpus to find the threshold for your hardware.

## Verification

Verify conversion completeness to ensure all content was captured:

### Single File Verification

```bash
python verify_cli.py content source.pdf output.md
```

### Marker Accuracy Audit

```bash
# Fine-grained page/slide marker audit
python verify_cli.py markers source.pdf output.md
```

### Batch Verification

```bash
# Verify all markdown files against source documents (auto-detects format)
python verify_cli.py content --batch output_dir/ --source-dir source_dir/
```

The verification tool compares:
- **Page counts** - Number of pages in PDF
- **Word/character counts** - Retention ratios (typically 95-105%)
- **Table detection** - Verifies tables were converted
- **Image analysis** - Flags potentially scanned pages

**Verification Status:**
- ✓ **PASSED** - Conversion is complete and accurate
- ⚠ **WARNING** - Minor issues (e.g., image-heavy pages detected)
- ✗ **FAILED** - Significant content loss detected

## Logging

All conversions are automatically logged to `conversion.log`:

```bash
# Default: saves to conversion.log in output directory
python convert.py input.pdf

# Custom log file location
python convert.py input.pdf --log-file /path/to/custom.log

# Verbose logging (includes DEBUG messages)
python convert.py input.pdf -v
```

**Log file includes:**
- Timestamps for all operations
- Conversion progress and statistics
- Error messages and stack traces
- Session start/end markers

**Example log entry:**
```
2025-12-22 09:41:27 - INFO - Converting: Chapter_I_Introduction.pdf
2025-12-22 09:41:27 - INFO - Output: Chapter_I_Introduction.md
2025-12-22 09:41:36 - INFO - ✓ Status: SUCCESS
2025-12-22 09:41:36 - INFO - ⏱  Time: 8.77s
```

## Conversion Reports

### Single File Report

Each conversion displays a rich visual interface with progress tracking:
```
╭───────────────────────── PDF to Markdown Converter ──────────────────────────╮
│ Input:  Chapter_II_Trade_Secret.pdf (122 pages, pp. 41-162, 0.7 MB)          │
│ Output: Chapter_II_Trade_Secret.md                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

⠋ Converting PDF with Docling... 0:00:19
⠋ Adding page markers... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   98% 0:00:00
⠋ Writing output... 0:00:00

╭──────────────────────────── Conversion Complete ─────────────────────────────╮
│   Time                       19.2s                                           │
│   Pages              122 (pp. 41-160)                                        │
│   Words                     61,279                                           │
│   Headings                     104                                           │
│   Tables                         1                                           │
│   Page Markers    120 / 122 (2 blank)                                        │
╰──────────────────────────────────────────────────────────────────────────────╯

Saved to: Chapter_II_Trade_Secret.md
```

### Batch Report

Batch conversions show overall progress and a comprehensive summary:
```
╭────────────────────────────── Batch Conversion ──────────────────────────────╮
│ Directory: /path/to/Chapters                                                 │
│ Output: /path/to/converted                                                   │
│ Files: 6 PDFs                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

  Scanning PDFs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Converting 6/6: Chapter_I_Introduction.pdf... ━━━━━━━━━━━ 100% 0:03:28

╭───────────────────────── Batch Conversion Complete ──────────────────────────╮
│   Files           6 / 6 successful                                           │
│   Total Time                3m 28s                                           │
│   Total Pages                1,309                                           │
│   Total Words              651,301                                           │
│   Avg per File               34.8s                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Batch optimization:** The converter reuses ML models across files, making batch processing 3-5x faster than converting files individually.

### JSON Report

Use `--save-report` to generate a detailed JSON file with per-file statistics:
```json
{
  "summary": {
    "files_processed": 6,
    "successful": 6,
    "total_pages": 1309,
    "total_words": 646140
  },
  "files": [
    {
      "status": "success",
      "input_file": "chapter.pdf",
      "conversion_time": 59.44,
      "statistics": {
        "pages": 352,
        "words": 174385,
        "headings": 305,
        "tables": 4
      }
    }
  ]
}
```

## Output Format

The generated markdown includes:
- Preserved heading hierarchy
- Properly formatted paragraphs
- Tables (where detected)
- Footnotes
- Metadata header with source file information

## Requirements

- Python 3.8+
- Docling (IBM Research PDF converter)
- PyMuPDF (for PDF metadata and page labels)
- RapidFuzz (for fast text matching - 10-100x faster than difflib)
- Rich (for visual progress feedback)

## Why Docling?

Docling is IBM Research's state-of-the-art PDF conversion tool that significantly outperforms other solutions for complex documents:

- Better table structure preservation than PyMuPDF or pdfplumber
- Advanced layout analysis for multi-column documents
- Superior heading detection and hierarchy
- Optimized specifically for feeding documents to LLMs

## Use Cases

- Preparing legal casebooks for AI analysis
- Converting academic papers for LLM context
- Creating searchable markdown archives
- RAG (Retrieval-Augmented Generation) pipelines
- Building knowledge bases from PDF documentation

## License

MIT
