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


def setup_logging(log_file: Optional[str] = None, verbose: bool = False, use_rich: bool = False):
    """
    Configure logging to file (and optionally console).

    Args:
        log_file: Optional path to log file. If None, uses default location.
        verbose: If True, show DEBUG messages
        use_rich: If True, suppress console logging (rich handles console output)
    """
    # Create logger
    logger = logging.getLogger('pdf_converter')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Only add console handler if not using rich (rich handles all console output)
    if not use_rich:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler - DEBUG and above
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Log session start to file only
        file_handler.emit(logging.LogRecord(
            'pdf_converter', logging.INFO, '', 0,
            "="*60, (), None
        ))
        file_handler.emit(logging.LogRecord(
            'pdf_converter', logging.INFO, '', 0,
            f"PDF to Markdown Converter - Session Started", (), None
        ))
        file_handler.emit(logging.LogRecord(
            'pdf_converter', logging.INFO, '', 0,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (), None
        ))
        file_handler.emit(logging.LogRecord(
            'pdf_converter', logging.INFO, '', 0,
            "="*60, (), None
        ))

    return logger


def print_conversion_report(report: Dict, logger: logging.Logger = None):
    """Print and log formatted conversion report."""
    if logger is None:
        logger = logging.getLogger('pdf_converter')

    output = []
    output.append("\n" + "="*60)
    output.append("CONVERSION REPORT")
    output.append("="*60)

    if report['status'] == 'success':
        stats = report['statistics']
        output.append(f"Status:       SUCCESS")
        output.append(f"Time:         {report['conversion_time']}s")
        output.append(f"")
        output.append(f"Statistics:")
        output.append(f"  Pages:      {stats['pages']}")
        output.append(f"  Words:      {stats['words']:,}")
        output.append(f"  Characters: {stats['characters']:,}")
        output.append(f"  Headings:   {stats['headings']}")
        output.append(f"  Tables:     {stats['tables']}")
        output.append(f"")
        output.append(f"Output:       {Path(report['output_file']).name}")
    else:
        output.append(f"Status:       FAILED")
        output.append(f"Time:         {report['conversion_time']}s")
        output.append(f"Error:        {report['error']}")

    output.append("="*60)

    # Log only (console handler will print it)
    report_text = '\n'.join(output)
    logger.info(report_text)


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None,
    pages: Optional[List[int]] = None,
    extract_images: bool = False,
    ocr: bool = False,
    report: bool = True,
    page_markers: bool = True,
    logger: logging.Logger = None,
    quiet: bool = False,
    converter: Optional['DocumentConverter'] = None,
    pdf_info: Optional[Dict] = None
) -> Dict:
    """
    Convert a PDF file to markdown format using Docling.

    Args:
        pdf_path: Path to the input PDF file
        output_path: Optional path for output markdown file. If None, uses same name as PDF.
        pages: Optional list of page numbers to convert (0-indexed)
        extract_images: Whether to extract and save images (default: False for text-only)
        ocr: Whether to use OCR for scanned documents (default: False)
        page_markers: Whether to add page number markers to output (default: True)
        quiet: Suppress visual output (default: False)
        converter: Optional pre-initialized DocumentConverter (for batch efficiency)
        pdf_info: Optional pre-computed PDF info dict with keys: page_count, first_page,
                  last_page, blank_pages, file_size (for batch efficiency)

    Returns:
        Dict: Conversion report with status and statistics
    """
    if logger is None:
        logger = logging.getLogger('pdf_converter')

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Determine output path
    if output_path is None:
        # Create 'converted' subfolder in same directory as PDF
        pdf_dir = Path(pdf_path).parent
        output_dir = pdf_dir / 'converted'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / Path(pdf_path).with_suffix('.md').name

    # Get file info - use pre-computed if available (batch mode), otherwise compute
    if pdf_info:
        file_size = pdf_info.get('file_size', os.path.getsize(pdf_path))
        page_count_preview = pdf_info.get('page_count')
        first_page_preview = pdf_info.get('first_page')
        last_page_preview = pdf_info.get('last_page')
        blank_pages_count = pdf_info.get('blank_pages', 0)
    else:
        file_size = os.path.getsize(pdf_path)
        page_count_preview = None
        first_page_preview = None
        last_page_preview = None
        blank_pages_count = 0
        try:
            doc = fitz.open(pdf_path)
            page_count_preview = len(doc)

            # Check for page labels to get actual page range
            page_labels = doc.get_page_labels()
            if page_labels:
                first_page_preview = get_actual_page_number(0, page_labels)
                last_page_preview = get_actual_page_number(len(doc) - 1, page_labels)

            # Count blank pages (pages with minimal text)
            for i in range(len(doc)):
                page_text = doc[i].get_text().strip()
                if len(page_text) < 20:  # Consider pages with <20 chars as blank
                    blank_pages_count += 1

            doc.close()
        except:
            pass

    # Log to file
    logger.debug(f"Converting: {pdf_path}")
    logger.debug(f"Output: {output_path}")

    # Track conversion time
    start_time = time.time()

    # Use rich visual output if available and not quiet
    use_rich = RICH_AVAILABLE and not quiet

    if use_rich:
        suppress_docling_logging()

    try:
        # Use provided converter or create new one
        # In batch mode, reusing the converter avoids reloading ML models for each file
        if converter is None:
            converter = DocumentConverter()

        # Phase 1: Convert PDF with Docling
        if use_rich:
            with ConversionProgress(pdf_path, str(output_path), page_count_preview, file_size,
                                   first_page_preview, last_page_preview, quiet) as progress:
                with progress.phase("Converting PDF with Docling..."):
                    result = converter.convert(pdf_path)
                    md_text = result.document.export_to_markdown()

                # Phase 2: Add page markers
                if page_markers:
                    page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0
                    with progress.phase(f"Adding page markers...", total=page_count) as update:
                        # Pass progress callback to add_page_markers
                        md_text = add_page_markers(md_text, pdf_path, result.document, logger,
                                                  progress_callback=update if page_count > 100 else None)

                # Phase 3: Write output
                with progress.phase("Writing output..."):
                    pdf_name = Path(pdf_path).name
                    header = f"# {Path(pdf_path).stem}\n\n"
                    header += f"*Converted from: {pdf_name}*\n\n"
                    header += f"*Conversion tool: Docling (IBM Research)*\n\n"
                    header += "---\n\n"
                    full_content = header + md_text

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_content)
        else:
            # Non-rich fallback
            result = converter.convert(pdf_path)
            md_text = result.document.export_to_markdown()

            if page_markers:
                md_text = add_page_markers(md_text, pdf_path, result.document, logger)

            pdf_name = Path(pdf_path).name
            header = f"# {Path(pdf_path).stem}\n\n"
            header += f"*Converted from: {pdf_name}*\n\n"
            header += f"*Conversion tool: Docling (IBM Research)*\n\n"
            header += "---\n\n"
            full_content = header + md_text

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_content)

        # Calculate conversion time
        conversion_time = time.time() - start_time

        # Analyze content
        lines = full_content.split('\n')
        headings = [line for line in lines if line.strip().startswith('#')]
        tables = [i for i in range(len(lines)-2)
                 if lines[i].startswith('|') and lines[i+1].startswith('|') and '---' in lines[i+1]]

        # Get document statistics
        char_count = len(full_content)
        word_count = len(full_content.split())
        page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0

        # Count page markers and extract page range
        pages_marked = full_content.count('<!-- Page')

        # Extract first and last page numbers from markers
        page_marker_matches = re.findall(r'<!-- Page (\d+) -->', full_content)
        if page_marker_matches:
            page_numbers = [int(p) for p in page_marker_matches]
            first_page = min(page_numbers)
            last_page = max(page_numbers)
        else:
            first_page = 1
            last_page = page_count

        # Build report
        conversion_report = {
            'status': 'success',
            'input_file': pdf_path,
            'output_file': str(output_path),
            'conversion_time': round(conversion_time, 2),
            'statistics': {
                'pages': page_count,
                'first_page': first_page,
                'last_page': last_page,
                'blank_pages': blank_pages_count,
                'characters': char_count,
                'words': word_count,
                'headings': len(headings),
                'tables': len(tables),
                'pages_marked': pages_marked
            }
        }

        if report:
            if use_rich:
                rich_print_report(conversion_report)
            else:
                print_conversion_report(conversion_report, logger)

        return conversion_report

    except Exception as e:
        error_report = {
            'status': 'error',
            'input_file': pdf_path,
            'error': str(e),
            'conversion_time': round(time.time() - start_time, 2)
        }
        logger.error(f"✗ Error converting {pdf_path}: {e}")
        return error_report


def get_pdf_info(pdf_path: str) -> Dict:
    """
    Get PDF metadata without full conversion.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict with page_count, first_page, last_page, blank_pages, file_size
    """
    info = {
        'file_size': os.path.getsize(pdf_path),
        'page_count': None,
        'first_page': None,
        'last_page': None,
        'blank_pages': 0
    }

    try:
        doc = fitz.open(pdf_path)
        info['page_count'] = len(doc)

        # Check for page labels
        page_labels = doc.get_page_labels()
        if page_labels:
            info['first_page'] = get_actual_page_number(0, page_labels)
            info['last_page'] = get_actual_page_number(len(doc) - 1, page_labels)

        # Count blank pages
        for i in range(len(doc)):
            page_text = doc[i].get_text().strip()
            if len(page_text) < 20:
                info['blank_pages'] += 1

        doc.close()
    except:
        pass

    return info


def batch_convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    save_report: bool = False,
    page_markers: bool = True,
    logger: logging.Logger = None
) -> Dict:
    """
    Convert all PDF files in a directory to markdown.

    Optimized for batch processing:
    - Reuses DocumentConverter across all files (avoids reloading ML models)
    - Pre-scans PDFs for metadata
    - Shows rich progress feedback

    Args:
        input_dir: Directory containing PDF files
        output_dir: Optional output directory. If None, outputs alongside PDFs.
        recursive: Whether to process subdirectories
        save_report: Whether to save JSON report
        page_markers: Whether to add page number markers to output (default: True)
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger('pdf_converter')

    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # Find all PDFs
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(input_path.glob(pattern))

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return {'success_count': 0, 'error_count': 0, 'reports': []}

    # If no output_dir specified, create 'converted' subfolder in input directory
    if output_dir is None:
        output_dir = input_path / 'converted'

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if rich is available for visual feedback
    use_rich = RICH_AVAILABLE

    if use_rich:
        suppress_docling_logging()
        from console import console, print_batch_summary
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        from rich.panel import Panel
        from rich.table import Table

        # Print batch header
        console.print()
        console.print(Panel(
            f"[bold]Directory:[/bold] {input_path}\n[bold]Output:[/bold] {output_path}\n[bold]Files:[/bold] {len(pdf_files)} PDFs",
            title="[bold blue]Batch Conversion[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        ))
        console.print()
    else:
        logger.info(f"Found {len(pdf_files)} PDF file(s) to convert")
        logger.info(f"Output directory: {output_path}")
        logger.info("-" * 60)

    success_count = 0
    error_count = 0
    total_time = 0
    total_words = 0
    total_pages = 0
    reports = []

    # Initialize DocumentConverter ONCE for all files (major performance optimization)
    # This avoids reloading ML models for each file
    converter = DocumentConverter()

    # Pre-scan all PDFs for metadata (enables better progress estimation)
    pdf_info_map = {}
    if use_rich:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Scanning PDFs...[/bold]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning", total=len(pdf_files))
            for pdf_path in pdf_files:
                pdf_info_map[str(pdf_path)] = get_pdf_info(str(pdf_path))
                progress.update(task, advance=1)
    else:
        for pdf_path in pdf_files:
            pdf_info_map[str(pdf_path)] = get_pdf_info(str(pdf_path))

    # Calculate total pages for progress estimation
    total_pages_estimate = sum(
        info.get('page_count', 0) or 0 for info in pdf_info_map.values()
    )

    # Convert each PDF with progress tracking
    if use_rich:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Converting 0/{len(pdf_files)}...", total=len(pdf_files))

            for i, pdf_path in enumerate(pdf_files, 1):
                progress.update(task, description=f"Converting {i}/{len(pdf_files)}: {pdf_path.name[:30]}...")

                # Determine output path
                out_dir = Path(output_dir)
                if recursive:
                    rel_path = pdf_path.relative_to(input_path)
                    out_path = out_dir / rel_path.with_suffix('.md')
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_path = out_dir / pdf_path.with_suffix('.md').name

                try:
                    report = convert_pdf_to_markdown(
                        str(pdf_path),
                        str(out_path),
                        extract_images=False,
                        ocr=False,
                        report=False,
                        page_markers=page_markers,
                        logger=logger,
                        quiet=True,  # Suppress individual file output in batch mode
                        converter=converter,  # Reuse converter
                        pdf_info=pdf_info_map.get(str(pdf_path))  # Pre-computed info
                    )
                    reports.append(report)

                    if report['status'] == 'success':
                        success_count += 1
                        total_time += report['conversion_time']
                        total_words += report['statistics']['words']
                        total_pages += report['statistics']['pages']
                    else:
                        error_count += 1

                except Exception as e:
                    logger.debug(f"Error converting {pdf_path}: {e}")
                    error_count += 1
                    reports.append({
                        'status': 'error',
                        'input_file': str(pdf_path),
                        'error': str(e)
                    })

                progress.update(task, advance=1)

        # Print batch summary using rich
        print_batch_summary(success_count, error_count, total_time, total_pages, total_words)

    else:
        # Non-rich fallback
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

            out_dir = Path(output_dir)
            if recursive:
                rel_path = pdf_path.relative_to(input_path)
                out_path = out_dir / rel_path.with_suffix('.md')
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = out_dir / pdf_path.with_suffix('.md').name

            try:
                report = convert_pdf_to_markdown(
                    str(pdf_path),
                    str(out_path),
                    extract_images=False,
                    ocr=False,
                    report=False,
                    page_markers=page_markers,
                    logger=logger,
                    quiet=True,
                    converter=converter,
                    pdf_info=pdf_info_map.get(str(pdf_path))
                )
                reports.append(report)

                if report['status'] == 'success':
                    success_count += 1
                    total_time += report['conversion_time']
                    total_words += report['statistics']['words']
                    total_pages += report['statistics']['pages']
                    logger.info(f"   OK {report['statistics']['pages']} pages, {report['statistics']['words']:,} words, {report['conversion_time']}s")
                else:
                    error_count += 1
                    logger.error(f"   FAILED: {report.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"FAILED: {e}")
                error_count += 1
                reports.append({
                    'status': 'error',
                    'input_file': str(pdf_path),
                    'error': str(e)
                })
                continue

    # Print batch summary (only for non-rich mode, rich mode already printed it)
    if not use_rich:
        summary = [
            "\n" + "=" * 60,
            "BATCH CONVERSION SUMMARY",
            "=" * 60,
            f"Files processed:  {len(pdf_files)}",
            f"  Successful:     {success_count}",
            f"  Failed:         {error_count}",
            f"",
            f"Totals:",
            f"  Time:           {total_time:.1f}s",
            f"  Pages:          {total_pages:,}",
            f"  Words:          {total_words:,}"
        ]

        if success_count > 0:
            summary.append(f"")
            summary.append(f"Average:          {total_time/success_count:.1f}s per file")

        summary.append("=" * 60)

        summary_text = '\n'.join(summary)
        logger.info(summary_text)

    # Save report to JSON if requested
    if save_report:
        report_path = Path(output_dir if output_dir else input_dir) / 'conversion_report.json'
        batch_report = {
            'summary': {
                'files_processed': len(pdf_files),
                'successful': success_count,
                'failed': error_count,
                'total_time': round(total_time, 2),
                'total_pages': total_pages,
                'total_words': total_words
            },
            'files': reports
        }
        with open(report_path, 'w') as f:
            json.dump(batch_report, f, indent=2)

        logger.info(f"\nDetailed report saved to: {report_path}")

    return {
        'success_count': success_count,
        'error_count': error_count,
        'reports': reports
    }


def parse_page_range(page_range: str) -> List[int]:
    """
    Parse page range string like '1-10' or '1,3,5-8' into list of page numbers.
    Note: Converts to 0-indexed.
    """
    pages = []
    for part in page_range.split(','):
        if '-' in part:
            start, end = part.split('-')
            pages.extend(range(int(start) - 1, int(end)))  # 0-indexed
        else:
            pages.append(int(part) - 1)  # 0-indexed
    return pages


def extract_page_text_with_pymupdf(pdf_path: str, logger: logging.Logger = None) -> List[Dict]:
    """
    Extract text from each page separately using PyMuPDF for accurate page tracking.

    Respects PDF page labels if present (e.g., Chapter 2 starting at page 41).

    Args:
        pdf_path: Path to PDF file
        logger: Optional logger for debugging

    Returns:
        List of dicts with page_number (actual page number) and text for each page
    """
    if logger is None:
        logger = logging.getLogger('pdf_converter')

    doc = fitz.open(pdf_path)
    pages = []

    # Check for PDF page labels (metadata that defines actual page numbers)
    page_labels = None
    try:
        page_labels = doc.get_page_labels()
        if page_labels:
            logger.debug(f"PDF has page labels: {page_labels}")
    except:
        pass

    # Determine actual page numbers for each PDF page
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text()

        # Calculate actual page number
        if page_labels:
            # Use PDF page labels to get the real page number
            actual_page_num = get_actual_page_number(page_idx, page_labels)
        else:
            # No labels - use sequential numbering starting at 1
            actual_page_num = page_idx + 1

        pages.append({
            'page_number': actual_page_num,
            'text': text.strip()
        })

    if page_labels:
        logger.debug(f"Page numbering: PDF pages 0-{len(doc)-1} → actual pages {pages[0]['page_number']}-{pages[-1]['page_number']}")

    doc.close()
    return pages


def get_actual_page_number(page_idx: int, page_labels: List[Dict]):
    """
    Convert PDF page index to actual page label using page labels.

    Supports different numbering styles (arabic, roman, letters) and prefixes.

    Args:
        page_idx: 0-based index of page in PDF
        page_labels: List of page label rules from PyMuPDF

    Returns:
        Actual page label for this page (e.g., "41", "iv", "A-3")
    """
    # Page labels format: [{'startpage': 0, 'prefix': '', 'firstpagenum': 41, 'style': 'D'}, ...]
    # Styles: D=decimal, r=roman lower, R=roman upper, a=letters lower, A=letters upper

    # Find the applicable label rule for this page
    applicable_rule = None
    for label in reversed(page_labels):  # Check from end to find last applicable rule
        if page_idx >= label.get('startpage', 0):
            applicable_rule = label
            break

    if not applicable_rule:
        # No rule found - use default sequential numbering
        return page_idx + 1

    # Calculate offset from start of this numbering section
    start_page = applicable_rule.get('startpage', 0)
    first_num = applicable_rule.get('firstpagenum', 1)
    offset = page_idx - start_page
    page_num = first_num + offset

    # Get style and prefix
    style = applicable_rule.get('style', 'D')
    prefix = applicable_rule.get('prefix', '')

    # Format the page number based on style
    if style == 'D':
        # Decimal (1, 2, 3...)
        formatted_num = str(page_num)
    elif style == 'r':
        # Lowercase Roman (i, ii, iii...)
        formatted_num = to_roman(page_num).lower()
    elif style == 'R':
        # Uppercase Roman (I, II, III...)
        formatted_num = to_roman(page_num)
    elif style == 'a':
        # Lowercase letters (a, b, c...)
        formatted_num = to_letters(page_num).lower()
    elif style == 'A':
        # Uppercase letters (A, B, C...)
        formatted_num = to_letters(page_num)
    else:
        # Unknown style - fall back to decimal
        formatted_num = str(page_num)

    # Combine prefix and formatted number
    return f"{prefix}{formatted_num}"


def to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


def to_letters(num: int) -> str:
    """Convert integer to letters (A, B, C, ... Z, AA, AB, ...)."""
    result = ""
    while num > 0:
        num -= 1
        result = chr(65 + (num % 26)) + result
        num //= 26
    return result


def normalize_text(text: str) -> str:
    """
    Normalize text for matching by removing extra whitespace and lowercasing.

    Uses pre-compiled regex pattern for better performance.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Replace multiple whitespace with single space (using pre-compiled pattern)
    text = _WHITESPACE_PATTERN.sub(' ', text)
    # Lowercase for case-insensitive matching
    text = text.lower()
    # Strip leading/trailing whitespace
    return text.strip()


def find_text_position(md_text: str, search_text: str, start_pos: int = 0, min_words: int = 10,
                       md_normalized: Optional[str] = None, estimated_pos: Optional[int] = None,
                       search_window_chars: int = 200000) -> int:
    """
    Find the position of search_text in md_text using optimized fuzzy matching.

    Performance optimizations:
    - Uses RapidFuzz (10-100x faster than difflib if available)
    - Smart position estimation to limit search window
    - Early termination on excellent matches

    Args:
        md_text: The markdown text to search in
        search_text: The text to find
        start_pos: Position to start searching from
        min_words: Minimum number of words to use as signature
        md_normalized: Pre-normalized markdown (optional, for performance)
        estimated_pos: Estimated position of match (enables windowed search)
        search_window_chars: Size of search window around estimated position

    Returns:
        Position where search_text was found, or -1 if not found
    """
    search_normalized = normalize_text(search_text)

    if not search_normalized:
        return -1

    # Use pre-normalized text if provided, otherwise normalize on demand
    if md_normalized is None:
        md_normalized = normalize_text(md_text)

    # Smart windowing: if we have an estimated position, search only nearby
    if estimated_pos is not None and estimated_pos > start_pos:
        # Search within window around estimated position
        window_start = max(start_pos, estimated_pos - search_window_chars // 2)
        window_end = min(len(md_normalized), estimated_pos + search_window_chars // 2)
        search_region = md_normalized[window_start:window_end]
        search_offset = window_start
    else:
        # No estimate - search from start_pos onward
        search_region = md_normalized[start_pos:]
        search_offset = start_pos

    # Try exact match first (C-optimized string search)
    exact_pos = search_region.find(search_normalized)
    if exact_pos != -1:
        return search_offset + exact_pos

    # Prepare signature for fuzzy matching
    words = search_normalized.split()

    # Adaptively choose signature size based on available text
    max_signature_words = min(50, len(words))
    signature_words = max(min_words, max_signature_words)

    if len(words) < min_words:
        # Very short page - use all available words
        signature_words = len(words)

    if signature_words == 0:
        return -1

    signature = ' '.join(words[:signature_words])
    window_size = len(signature)
    threshold = 65  # RapidFuzz uses 0-100 scale

    # Fuzzy search with early termination
    best_ratio = 0.0
    best_pos = -1
    step_size = max(5, window_size // 20)

    if RAPIDFUZZ_AVAILABLE:
        # Use RapidFuzz (10-100x faster than difflib)
        for i in range(0, max(1, len(search_region) - window_size + 1), step_size):
            window = search_region[i:i + window_size]
            ratio = fuzz.ratio(signature, window)

            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

                # Early termination if excellent match found
                if ratio > 90:
                    break
    else:
        # Fallback to difflib
        for i in range(0, max(1, len(search_region) - window_size + 1), step_size):
            window = search_region[i:i + window_size]
            ratio = SequenceMatcher(None, signature, window).ratio() * 100

            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

                # Early termination if excellent match found
                if ratio > 90:
                    break

    if best_ratio >= threshold:
        return search_offset + best_pos

    return -1


def get_table_page_mapping(doc) -> Dict[int, List]:
    """
    Extract mapping of page numbers to tables from Docling document.

    Args:
        doc: DoclingDocument object

    Returns:
        Dict mapping page number (1-indexed) to list of table indices
    """
    table_pages = {}

    if hasattr(doc, 'tables'):
        for table_idx, table in enumerate(doc.tables):
            if hasattr(table, 'prov') and table.prov:
                # prov is a list of ProvenanceItem objects
                for prov_item in table.prov:
                    if hasattr(prov_item, 'page_no'):
                        page_no = prov_item.page_no
                        if page_no not in table_pages:
                            table_pages[page_no] = []
                        table_pages[page_no].append(table_idx)

    return table_pages


def find_table_in_markdown(md_text: str, table_idx: int, doc, start_pos: int = 0,
                           table_md_cache: Optional[Dict] = None, md_normalized: Optional[str] = None) -> int:
    """
    Find where a specific table appears in the markdown text.

    Optimized with caching and two-phase search.

    Args:
        md_text: The markdown text
        table_idx: Index of table in doc.tables
        doc: DoclingDocument object
        start_pos: Position to start searching from
        table_md_cache: Optional cache for table markdown exports
        md_normalized: Pre-normalized markdown (optional, for performance)

    Returns:
        Position where table starts, or -1 if not found
    """
    if not hasattr(doc, 'tables') or table_idx >= len(doc.tables):
        return -1

    # Use cache to avoid re-exporting same table multiple times
    if table_md_cache is None:
        table_md_cache = {}

    table = doc.tables[table_idx]

    # Export table markdown (with caching)
    if table_idx not in table_md_cache:
        try:
            table_md_cache[table_idx] = table.export_to_markdown(doc)
        except Exception:
            return -1

    table_md = table_md_cache[table_idx]

    # Use pre-normalized text if available
    if md_normalized is None:
        md_normalized = normalize_text(md_text[start_pos:])
    else:
        md_normalized = md_normalized[start_pos:]

    # Normalize table text
    table_normalized = normalize_text(table_md[:200])  # Use first 200 chars

    # Try exact match first
    pos = md_normalized.find(table_normalized)
    if pos != -1:
        return start_pos + pos

    # If exact match fails, try fuzzy matching with two-phase approach
    table_words = table_normalized.split()[:20]
    if not table_words:
        return -1

    signature = ' '.join(table_words)
    window_size = len(signature)
    threshold = 0.7

    # Fuzzy search with RapidFuzz or difflib fallback
    best_ratio = 0.0
    best_pos = -1

    if RAPIDFUZZ_AVAILABLE:
        for i in range(0, max(1, len(md_normalized) - window_size + 1), 20):
            window = md_normalized[i:i + window_size]
            ratio = fuzz.ratio(signature, window)
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
                if ratio > 85:
                    break
        # RapidFuzz uses 0-100 scale
        threshold = 70
    else:
        for i in range(0, max(1, len(md_normalized) - window_size + 1), 20):
            window = md_normalized[i:i + window_size]
            ratio = SequenceMatcher(None, signature, window).ratio() * 100
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
                if ratio > 85:
                    break
        threshold = 70

    if best_ratio >= threshold:
        return start_pos + best_pos

    return -1


def insert_page_markers_hybrid(md_text: str, pdf_path: str, doc=None, logger: logging.Logger = None) -> str:
    """
    Insert accurate page markers using hybrid PyMuPDF + text matching approach.

    Uses PyMuPDF to extract text from each page, then matches that text against
    Docling's markdown output to find accurate page boundaries. For pages with
    tables, uses Docling's table provenance information.

    Args:
        md_text: Markdown text from Docling
        pdf_path: Path to original PDF for page extraction
        doc: DoclingDocument object (optional, but enables table-aware matching)
        logger: Optional logger for debugging

    Returns:
        Markdown text with accurate page markers inserted
    """
    if logger is None:
        logger = logging.getLogger('pdf_converter')

    # Extract page-by-page text using PyMuPDF
    pages = extract_page_text_with_pymupdf(pdf_path, logger)

    if not pages:
        return md_text

    # Single page document - simple case
    if len(pages) == 1:
        return f"<!-- Page 1 -->\n\n{md_text}"

    # Get table-to-page mapping from Docling document
    table_pages = {}
    if doc:
        table_pages = get_table_page_mapping(doc)
        if table_pages:
            logger.debug(f"Found tables on pages: {sorted(table_pages.keys())}")

    # Multi-page document - need to find page boundaries
    marked_text = md_text
    insertions = []  # Track (position, marker_text) tuples
    failed_pages = []  # Track pages that couldn't be matched

    # OPTIMIZATION: Pre-normalize the entire markdown text once
    # This avoids normalizing it 500+ times (once per page search)
    md_normalized = normalize_text(marked_text)
    logger.debug(f"Pre-normalized markdown text ({len(md_normalized)} chars) for efficient searching")

    current_search_pos = 0
    pages_found = 0
    blank_pages_skipped = []

    # Calculate average chars per page for position estimation
    total_md_chars = len(marked_text)
    total_pages = len(pages)
    avg_chars_per_page = total_md_chars // total_pages if total_pages > 0 else 0

    # Progress tracking for large documents
    show_progress = total_pages > 100
    if show_progress:
        logger.info(f"Processing {total_pages} pages with position estimation...")
        progress_interval = max(10, total_pages // 20)  # Show progress every 5%

    for i, page_data in enumerate(pages):
        page_num = page_data['page_number']
        page_text = page_data['text']

        # Progress indicator for large documents
        if show_progress and i > 0 and i % progress_interval == 0:
            percent = (i / total_pages) * 100
            logger.info(f"  Progress: {i}/{total_pages} pages ({percent:.0f}%) - {pages_found} matched")

        if i == 0:
            # First page - marker goes at the beginning
            insertions.append((0, f"<!-- Page {page_num} -->\n\n"))
            pages_found += 1
            continue

        # Estimate where this page should be (for smart windowing)
        estimated_pos = current_search_pos + avg_chars_per_page

        # For subsequent pages, try multiple strategies to find the page boundary
        pos = -1

        # Strategy 1: Try first 500 chars with position estimation
        signature_length = min(500, len(page_text))
        if signature_length > 0:
            signature = page_text[:signature_length]
            pos = find_text_position(marked_text, signature, current_search_pos, min_words=5,
                                    md_normalized=md_normalized, estimated_pos=estimated_pos)

        # Strategy 2: If that failed, try first 200 chars with lower threshold
        if pos == -1 and len(page_text) >= 100:
            signature_length = min(200, len(page_text))
            signature = page_text[:signature_length]
            pos = find_text_position(marked_text, signature, current_search_pos, min_words=3,
                                    md_normalized=md_normalized, estimated_pos=estimated_pos)

        # Strategy 3: Try middle portion of page (skip headers/footers that might be missing)
        if pos == -1 and len(page_text) >= 200:
            # Take text from 10% to 60% of the page
            start_offset = len(page_text) // 10
            end_offset = min(start_offset + 300, len(page_text) * 6 // 10)
            signature = page_text[start_offset:end_offset]
            pos = find_text_position(marked_text, signature, current_search_pos, min_words=3,
                                    md_normalized=md_normalized, estimated_pos=estimated_pos)

        # Strategy 4: Extract distinctive words/phrases and search for them
        # This helps with tables where formatting is very different
        if pos == -1 and len(page_text) >= 50:
            # Split into words and take every 5th word (skip common words, get content words)
            words = page_text.split()
            if len(words) >= 10:
                # Create a minimal signature from scattered distinctive words
                distinctive_words = [words[i] for i in range(0, min(50, len(words)), 5)]
                signature = ' '.join(distinctive_words[:10])  # Use 10 words max
                pos = find_text_position(marked_text, signature, current_search_pos, min_words=2,
                                        md_normalized=md_normalized, estimated_pos=estimated_pos)

        # Strategy 5: For blank pages, skip them entirely
        # If a PDF page is blank and we can't find it, it means Docling didn't include it
        # in the markdown output. Don't fabricate a position - just skip it.
        if pos == -1 and len(page_text) == 0:
            logger.debug(f"Page {page_num} is blank - skipping (no content in markdown)")
            blank_pages_skipped.append(page_num)
            # Don't add to failed_pages, and don't insert a marker
            continue

        if pos != -1:
            # Found the page boundary - record insertion point
            insertions.append((pos, f"\n\n<!-- Page {page_num} -->\n\n"))
            current_search_pos = pos + len(signature if 'signature' in locals() and signature else 100)
            pages_found += 1

            # Update average chars per page estimate based on actual findings
            if pages_found > 1:
                avg_chars_per_page = current_search_pos // pages_found
        else:
            # Couldn't find this page's text - track for table-based recovery
            failed_pages.append((i, page_num, page_text))
            logger.debug(f"Could not locate page {page_num} in markdown (page has {len(page_text)} chars)")

    logger.debug(f"Successfully matched {pages_found}/{len(pages)} pages")

    if blank_pages_skipped:
        logger.info(f"Skipped {len(blank_pages_skipped)} blank pages: {blank_pages_skipped[:20]}" +
                   (f" ... and {len(blank_pages_skipped)-20} more" if len(blank_pages_skipped) > 20 else ""))

    # Second pass: Try to recover failed pages using table information
    if failed_pages and table_pages and doc:
        logger.debug(f"Attempting table-based recovery for {len(failed_pages)} pages...")

        # Create cache for table markdown exports (avoid re-exporting same table)
        table_md_cache = {}

        for i, page_num, page_text in failed_pages:
            # Check if this page has a table
            if page_num in table_pages:
                table_indices = table_pages[page_num]
                logger.debug(f"  Page {page_num} has {len(table_indices)} table(s): {table_indices}")

                # Try to find each table in the markdown
                for table_idx in table_indices:
                    # Find where this table appears (with caching and pre-normalized text)
                    table_pos = find_table_in_markdown(
                        marked_text, table_idx, doc,
                        start_pos=0,
                        table_md_cache=table_md_cache,
                        md_normalized=md_normalized
                    )

                    if table_pos != -1:
                        # Found the table! Place page marker before it
                        insertions.append((table_pos, f"\n\n<!-- Page {page_num} -->\n\n"))
                        pages_found += 1
                        logger.debug(f"  ✓ Recovered page {page_num} using table {table_idx} at position {table_pos}")
                        break
                    else:
                        logger.debug(f"  ✗ Could not find table {table_idx} in markdown")

        logger.debug(f"After table recovery: {pages_found}/{len(pages)} pages")

    # Apply insertions using list-based approach (O(n) instead of O(n²))
    # Sort in forward order (not reverse)
    insertions.sort(key=lambda x: x[0])

    # Build result efficiently using list parts
    result_parts = []
    last_pos = 0

    for pos, marker in insertions:
        # Add text segment since last insertion
        result_parts.append(marked_text[last_pos:pos])
        # Add marker
        result_parts.append(marker)
        last_pos = pos

    # Add remaining text after last insertion
    result_parts.append(marked_text[last_pos:])

    # Join all parts once at the end (much faster than repeated concatenation)
    return ''.join(result_parts)


def insert_page_markers_provenance(md_text: str, doc, pdf_path: str = None,
                                    logger: logging.Logger = None,
                                    progress_callback=None) -> str:
    """
    Insert page markers using Docling's element provenance information.

    This approach uses Docling's internal page tracking for each document element,
    which is far more accurate than trying to match PDF text against markdown.

    Args:
        md_text: Markdown text from Docling
        doc: DoclingDocument object with element provenance
        pdf_path: Path to original PDF (for page label support)
        logger: Optional logger for debugging
        progress_callback: Optional callback for progress updates

    Returns:
        Markdown text with accurate page markers inserted
    """
    if logger is None:
        logger = logging.getLogger('pdf_converter')

    if not doc:
        logger.warning("No Docling document provided for provenance-based markers")
        return md_text

    # Get PDF page labels if available (for non-sequential page numbering)
    page_labels = None
    if pdf_path:
        try:
            pdf_doc = fitz.open(pdf_path)
            page_labels = pdf_doc.get_page_labels()
            if page_labels:
                logger.debug(f"PDF has page labels: {page_labels}")
            pdf_doc.close()
        except Exception as e:
            logger.debug(f"Could not read PDF page labels: {e}")

    # Build mapping of page -> first text item on that page
    first_items_by_page = {}

    for item, level in doc.iterate_items():
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    page_no = prov.page_no
                    if page_no not in first_items_by_page:
                        # Get text from item
                        text = getattr(item, 'text', None)
                        if text and len(text.strip()) > 10:  # Meaningful text
                            first_items_by_page[page_no] = text.strip()
                    break

    if not first_items_by_page:
        logger.warning("No page provenance found in document")
        return md_text

    total_pages = max(first_items_by_page.keys())
    logger.debug(f"Found provenance for {len(first_items_by_page)} pages (max page: {total_pages})")

    # Normalize markdown for searching
    md_lower = md_text.lower()

    # Find positions for each page
    insertions = []
    pages_found = 0
    last_pos = 0  # Track last found position to ensure forward progress
    sorted_pages = sorted(first_items_by_page.keys())

    for i, page_no in enumerate(sorted_pages):
        item_text = first_items_by_page[page_no]

        # Search for item text in markdown (case-insensitive)
        search_text = item_text[:80].lower()  # Use first 80 chars

        # Search from last position forward
        pos = md_lower.find(search_text, last_pos)

        if pos == -1:
            # Try shorter match
            search_text = item_text[:40].lower()
            pos = md_lower.find(search_text, last_pos)

        if pos == -1:
            # Try even shorter match
            search_text = item_text[:20].lower()
            pos = md_lower.find(search_text, last_pos)

        if pos != -1:
            insertions.append((pos, page_no))
            last_pos = pos + len(search_text)
            pages_found += 1
        else:
            logger.debug(f"Could not locate page {page_no} text in markdown")

        # Update progress
        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1)

    # Final progress update
    if progress_callback:
        progress_callback(len(sorted_pages))

    logger.debug(f"Located {pages_found}/{len(first_items_by_page)} pages using provenance")

    if not insertions:
        return md_text

    # Sort by position
    insertions.sort(key=lambda x: x[0])

    # Build result with markers
    result_parts = []
    last_insert_pos = 0

    for pos, page_no in insertions:
        # Add text before this marker
        result_parts.append(md_text[last_insert_pos:pos])

        # Convert Docling's 1-indexed page to actual page label
        # Docling uses 1-indexed pages, PDF page labels use 0-indexed
        pdf_page_idx = page_no - 1  # Convert to 0-indexed for page label lookup
        if page_labels:
            actual_page = get_actual_page_number(pdf_page_idx, page_labels)
        else:
            actual_page = page_no

        # Add marker with actual page number
        result_parts.append(f"\n\n<!-- Page {actual_page} -->\n\n")
        last_insert_pos = pos

    # Add remaining text
    result_parts.append(md_text[last_insert_pos:])

    return ''.join(result_parts)


def add_page_markers(md_text: str, pdf_path: str, doc=None, logger: logging.Logger = None,
                     progress_callback=None) -> str:
    """
    Add accurate page number markers to markdown content.

    Uses Docling's element provenance for accurate page tracking. Falls back to
    PyMuPDF text matching if provenance is unavailable.

    Args:
        md_text: Markdown text from Docling
        pdf_path: Path to original PDF file (needed for fallback)
        doc: DoclingDocument object with page information
        logger: Optional logger for debugging
        progress_callback: Optional callback for progress updates (called with page count)

    Returns:
        Markdown text with accurate page markers inserted
    """
    import re

    if logger is None:
        logger = logging.getLogger('pdf_converter')

    # First check for Docling's internal page break markers (rare but possible)
    pattern = r"#_#_DOCLING_DOC_PAGE_BREAK_(\d+)_(\d+)_#_#"
    matches = list(re.finditer(pattern, md_text))

    if matches:
        # We have internal markers with accurate page numbers - use them
        first_match = matches[0]
        first_page = int(first_match.group(1)) + 1  # Convert 0-indexed to 1-indexed

        # Replace all markers with page-specific HTML comments
        def replacement(match):
            next_page = int(match.group(2))
            return f"\n\n<!-- Page {next_page + 1} -->\n\n"

        md_text = re.sub(pattern, replacement, md_text)
        md_text = f"<!-- Page {first_page} -->\n\n{md_text}"
        return md_text

    # Try provenance-based approach first (most accurate)
    if doc:
        try:
            result = insert_page_markers_provenance(md_text, doc, pdf_path, logger, progress_callback)
            # Check if we got reasonable results
            marker_count = result.count('<!-- Page')
            if marker_count > 0:
                logger.debug(f"Provenance-based markers: {marker_count} pages marked")
                return result
        except Exception as e:
            logger.warning(f"Provenance-based marker insertion failed: {e}")

    # Fall back to PyMuPDF + text matching approach
    try:
        return insert_page_markers_hybrid(md_text, pdf_path, doc, logger)
    except Exception as e:
        logger.warning(f"Hybrid page marker insertion failed: {e}")
        # If both approaches fail, fall back to simple single-page marker
        if doc and hasattr(doc, 'pages') and len(doc.pages) == 1:
            return f"<!-- Page 1 -->\n\n{md_text}"

        # Multi-page conversion failed - return without markers
        # Better to have no markers than inaccurate markers
        return md_text


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
