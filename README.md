# PDF to Markdown Converter

Convert PDF files (especially large legal casebooks) to markdown format optimized for AI tools using **Docling** - IBM Research's state-of-the-art PDF conversion tool.

## Features

- **Advanced layout analysis** - Preserves document structure with high fidelity
- **Table structure recognition** - Maintains complex table formatting
- **Heading hierarchy detection** - Proper markdown heading levels
- **Formula extraction** - Handles mathematical notation (if present)
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
```

### Page Range Selection

```bash
# Convert specific pages (1-indexed)
python pdf_to_markdown.py casebook.pdf --pages 1-50 -o chapter1.md

# Convert multiple ranges
python pdf_to_markdown.py casebook.pdf --pages 1-10,25-35 -o sections.md
```

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
