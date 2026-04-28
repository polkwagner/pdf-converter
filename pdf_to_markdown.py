#!/usr/bin/env python3
"""
pdf_to_markdown.py

Convert PDF files (especially large casebooks) to markdown format optimized for AI tools.
Uses Docling (IBM Research) for state-of-the-art high-fidelity PDF conversion with advanced
layout analysis, table recognition, and structure preservation.

Usage:
    python pdf_to_markdown.py <input_pdf> [-o <output_file>]
    python pdf_to_markdown.py <input_directory> [--batch]

Examples:
    # Single file conversion
    python pdf_to_markdown.py chapter1.pdf -o chapter1.md

    # Batch conversion of all PDFs in a directory
    python pdf_to_markdown.py ./casebooks/ --batch

    # With page range
    python pdf_to_markdown.py casebook.pdf --pages 1-50 -o chapter1.md
"""

import argparse
import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from difflib import SequenceMatcher
import re

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("ERROR: Docling not installed.")
    print("Install with: pip install docling")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed.")
    print("Install with: pip install PyMuPDF")
    sys.exit(1)

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False

# Rich console for visual output
try:
    from console import (
        console, ConversionProgress, print_header, print_conversion_report as rich_print_report,
        print_batch_summary, print_success, print_warning, print_error, suppress_docling_logging
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Pre-compile regex patterns for performance (used in normalize_text)
_WHITESPACE_PATTERN = re.compile(r'\s+')

# Cache for normalized markdown text (reused across page searches)
_normalized_text_cache = {}


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF files to markdown optimized for AI tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s chapter1.pdf -o chapter1.md
  %(prog)s ./casebooks/ --batch
  %(prog)s casebook.pdf --pages 1-50 -o excerpt.md
        """
    )

    parser.add_argument(
        'input',
        help='Input PDF file or directory'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output markdown file or directory'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch convert all PDFs in input directory'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process subdirectories recursively (with --batch)'
    )

    parser.add_argument(
        '--pages',
        help='Page range to convert (e.g., "1-10" or "1,3,5-8"). 1-indexed.'
    )

    parser.add_argument(
        '--images',
        action='store_true',
        help='Extract and save images from PDF'
    )

    parser.add_argument(
        '--ocr',
        action='store_true',
        help='Enable OCR for scanned documents (slower but handles image-based PDFs)'
    )

    parser.add_argument(
        '--no-page-markers',
        dest='page_markers',
        action='store_false',
        help='Disable page number markers (enabled by default)'
    )

    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed conversion report to JSON file (batch mode only)'
    )

    parser.add_argument(
        '--log-file',
        help='Path to log file (default: conversion.log in output directory)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Setup logging
    if args.log_file:
        log_path = args.log_file
    elif args.batch or os.path.isdir(args.input):
        # Default log file for batch mode - in 'converted' subfolder
        if args.output:
            output_location = Path(args.output)
        else:
            output_location = Path(args.input) / 'converted'
        output_location.mkdir(parents=True, exist_ok=True)
        log_path = output_location / 'conversion.log'
    else:
        # Default log file for single file mode - in 'converted' subfolder
        if args.output:
            output_location = Path(args.output).parent
        else:
            output_location = Path(args.input).parent / 'converted'
        output_location.mkdir(parents=True, exist_ok=True)
        log_path = output_location / 'conversion.log'

    logger = setup_logging(str(log_path), verbose=args.verbose, use_rich=RICH_AVAILABLE)
    logger.debug(f"Log file: {log_path}")

    # Parse page range if provided
    pages = None
    if args.pages:
        try:
            pages = parse_page_range(args.pages)
            logger.debug(f"Converting pages: {args.pages} (0-indexed: {pages})")
        except Exception as e:
            logger.error(f"Error parsing page range: {e}")
            sys.exit(1)

    # Batch or single file mode
    if args.batch or os.path.isdir(args.input):
        batch_convert_directory(
            args.input,
            output_dir=args.output,
            recursive=args.recursive,
            save_report=args.save_report,
            page_markers=args.page_markers,
            logger=logger
        )
    else:
        convert_pdf_to_markdown(
            args.input,
            output_path=args.output,
            pages=pages,
            extract_images=args.images,
            ocr=args.ocr,
            page_markers=args.page_markers,
            logger=logger
        )

    logger.debug("\n" + "="*60)
    logger.debug("Session completed")
    logger.debug("="*60)


if __name__ == '__main__':
    main()
