# PDF to Markdown Converter

Convert PDF files (especially large legal casebooks) to markdown format optimized for AI tools using **Docling** - IBM Research's state-of-the-art PDF conversion tool.

## Features

- **Advanced layout analysis** - Preserves document structure with high fidelity
- **Table structure recognition** - Maintains complex table formatting
- **Heading hierarchy detection** - Proper markdown heading levels
- **Formula extraction** - Handles mathematical notation (if present)
- **Page number markers** - ⚠️ Currently limited to single-page PDFs (multi-page support pending)
- **OCR support** - Optional OCR for scanned documents
- **Batch processing** - Convert entire directories of PDFs
- **Page range selection** - Extract specific sections
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
python pdf_to_markdown.py input.pdf -o output.md
```

### Batch Conversion

```bash
# Convert all PDFs in a directory
python pdf_to_markdown.py ./casebooks/ --batch

# Recursive conversion (includes subdirectories)
python pdf_to_markdown.py ./casebooks/ --batch --recursive

# Save detailed JSON report
python pdf_to_markdown.py ./casebooks/ --batch --save-report
```

### Page Range Selection

```bash
# Convert specific pages (1-indexed)
python pdf_to_markdown.py casebook.pdf --pages 1-50 -o chapter1.md

# Convert multiple ranges
python pdf_to_markdown.py casebook.pdf --pages 1-10,25-35 -o sections.md
```

### Page Number Markers (for AI/RAG)

**✓ NOW FULLY SUPPORTED** for both single-page and multi-page documents!

Page markers are added automatically as HTML comments (`<!-- Page N -->`) throughout the markdown output, preserving the original PDF page numbers. This is essential for legal citations and RAG (Retrieval-Augmented Generation) applications.

```bash
# Page markers enabled by default
python pdf_to_markdown.py casebook.pdf -o output.md
# Output includes: <!-- Page 1 -->, <!-- Page 2 -->, etc.

# Disable page markers if needed
python pdf_to_markdown.py document.pdf --no-page-markers -o output.md
```

**How it works - Hybrid Approach:**

The tool uses a sophisticated hybrid approach combining two methods for maximum accuracy:

1. **PyMuPDF Text Extraction**: Extracts text from each PDF page separately to know exact page boundaries
2. **Fuzzy Text Matching**: Matches PyMuPDF's page-specific text against Docling's markdown using multiple strategies
3. **Table-Aware Detection**: For pages with tables (where text structure differs significantly), uses Docling's table provenance information to locate page breaks
4. **Positional Estimation**: For blank/image-only pages, estimates position based on document structure

**Accuracy:**
- Tested on 40-page legal casebook: **100% of pages accurately marked**
- Tested on 122-page chapter (pages 41-162): **100% accuracy**
- Handles complex multi-page tables
- Preserves original PDF page numbers (critical for citations)
- **Automatically detects PDF page labels** (e.g., Chapter 2 starting at page 41)
- Works with blank pages and image-heavy documents

### Extract Images

```bash
python pdf_to_markdown.py input.pdf --images
```

### OCR for Scanned Documents

```bash
# Enable OCR for image-based PDFs (slower but necessary for scanned docs)
python pdf_to_markdown.py scanned_casebook.pdf --ocr
```

## Verification

Verify conversion completeness to ensure all content was captured:

### Single File Verification

```bash
python verify_conversion.py source.pdf output.md
```

### Batch Verification

```bash
# Verify all markdown files against source PDFs
python verify_conversion.py --batch output_dir/ --pdf-dir source_dir/
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
python pdf_to_markdown.py input.pdf

# Custom log file location
python pdf_to_markdown.py input.pdf --log-file /path/to/custom.log

# Verbose logging (includes DEBUG messages)
python pdf_to_markdown.py input.pdf -v
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

Each conversion displays detailed statistics:
```
============================================================
📄 CONVERSION REPORT
============================================================
✓ Status:     SUCCESS
⏱  Time:       9.47s
📊 Statistics:
   Pages:     40
   Words:     18,092
   Characters: 125,506
   Headings:  33
   Tables:    4
📁 Output:    chapter.md
============================================================
```

### Batch Report

Batch conversions provide comprehensive summaries:
```
============================================================
📊 BATCH CONVERSION SUMMARY
============================================================
Files processed:  6
✓ Successful:     6
✗ Failed:         0
⏱  Total time:     219.7s
📄 Total pages:    1,309
📝 Total words:    646,140
⚡ Avg speed:      36.6s per file
============================================================
```

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
- Docling (includes all necessary dependencies)

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
