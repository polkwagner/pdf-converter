# PDF to Markdown Converter

Convert PDF files (especially large legal casebooks) to markdown format optimized for AI tools.

## Features

- Intelligent text extraction with structure preservation
- Batch processing of multiple PDFs
- Page range selection
- Optimized for LLM consumption (Claude, GPT-4, etc.)

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

## Output Format

The generated markdown includes:
- Preserved heading hierarchy
- Properly formatted paragraphs
- Tables (where detected)
- Footnotes
- Metadata header with source file information

## Requirements

- Python 3.8+
- pymupdf4llm
- PyMuPDF (fitz)

## Use Cases

- Preparing legal casebooks for AI analysis
- Converting academic papers for LLM context
- Creating searchable markdown archives
- RAG (Retrieval-Augmented Generation) pipelines

## License

MIT
