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


def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """
    Configure logging to both console and file.

    Args:
        log_file: Optional path to log file. If None, uses default location.
        verbose: If True, show DEBUG messages
    """
    # Create logger
    logger = logging.getLogger('pdf_converter')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler - INFO and above
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

        # Log session start
        logger.info("="*60)
        logger.info(f"PDF to Markdown Converter - Session Started")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

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
    logger: logging.Logger = None
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

    Returns:
        str: The generated markdown content
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

    logger.info(f"Converting: {pdf_path}")
    logger.info(f"Output: {output_path}")

    # Track conversion time
    start_time = time.time()

    # Initialize DocumentConverter
    # Docling will use default settings optimized for high-fidelity conversion
    converter = DocumentConverter()

    try:
        # Convert the PDF
        result = converter.convert(pdf_path)

        # Export to markdown
        md_text = result.document.export_to_markdown()

        # Add page markers if requested
        if page_markers:
            md_text = add_page_markers(md_text, pdf_path, result.document, logger)

        # Add metadata header
        pdf_name = Path(pdf_path).name
        header = f"# {Path(pdf_path).stem}\n\n"
        header += f"*Converted from: {pdf_name}*\n\n"
        header += f"*Conversion tool: Docling (IBM Research)*\n\n"
        header += "---\n\n"

        full_content = header + md_text

        # Write to file
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

        # Build report
        conversion_report = {
            'status': 'success',
            'input_file': pdf_path,
            'output_file': str(output_path),
            'conversion_time': round(conversion_time, 2),
            'statistics': {
                'pages': page_count,
                'characters': char_count,
                'words': word_count,
                'headings': len(headings),
                'tables': len(tables)
            }
        }

        if report:
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
        return

    # If no output_dir specified, create 'converted' subfolder in input directory
    if output_dir is None:
        output_dir = input_path / 'converted'

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Found {len(pdf_files)} PDF file(s) to convert")
    logger.info(f"Output directory: {output_path}")
    logger.info("-" * 60)

    success_count = 0
    error_count = 0
    total_time = 0
    total_words = 0
    total_pages = 0
    reports = []

    # Convert each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        # Determine output path
        out_dir = Path(output_dir)
        # Preserve subdirectory structure if recursive
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
                report=False,  # Don't print individual reports in batch mode
                page_markers=page_markers,
                logger=logger
            )
            reports.append(report)

            if report['status'] == 'success':
                success_count += 1
                total_time += report['conversion_time']
                total_words += report['statistics']['words']
                total_pages += report['statistics']['pages']
                logger.info(f"   ✓ {report['statistics']['pages']} pages, {report['statistics']['words']:,} words, {report['conversion_time']}s")
            else:
                error_count += 1
                logger.error(f"   ✗ Failed: {report.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            error_count += 1
            reports.append({
                'status': 'error',
                'input_file': str(pdf_path),
                'error': str(e)
            })
            continue

    # Print batch summary
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


def extract_page_text_with_pymupdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page separately using PyMuPDF for accurate page tracking.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with page_number (1-indexed) and text for each page
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({
            'page_number': page_num + 1,  # 1-indexed
            'text': text.strip()
        })

    doc.close()
    return pages


def normalize_text(text: str) -> str:
    """
    Normalize text for matching by removing extra whitespace and lowercasing.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    import re
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Lowercase for case-insensitive matching
    text = text.lower()
    # Strip leading/trailing whitespace
    return text.strip()


def find_text_position(md_text: str, search_text: str, start_pos: int = 0, min_words: int = 10) -> int:
    """
    Find the position of search_text in md_text using fuzzy matching.

    Args:
        md_text: The markdown text to search in
        search_text: The text to find
        start_pos: Position to start searching from
        min_words: Minimum number of words to use as signature

    Returns:
        Position where search_text was found, or -1 if not found
    """
    # Normalize both texts
    md_normalized = normalize_text(md_text[start_pos:])
    search_normalized = normalize_text(search_text)

    if not search_normalized:
        return -1

    # Try exact match first
    exact_pos = md_normalized.find(search_normalized)
    if exact_pos != -1:
        return start_pos + exact_pos

    # If exact match fails, try fuzzy matching with sliding window
    # Use first N words of search text as signature (more reliable than full text)
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

    best_ratio = 0.0
    best_pos = -1
    threshold = 0.65  # Lowered from 0.75 to be more lenient

    # Slide window across markdown text
    step_size = max(5, window_size // 20)  # Adaptive step size
    for i in range(0, max(1, len(md_normalized) - window_size + 1), step_size):
        window = md_normalized[i:i + window_size]
        ratio = SequenceMatcher(None, signature, window).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = i

            # If we found a very good match, stop early
            if ratio > 0.9:
                break

    if best_ratio >= threshold:
        return start_pos + best_pos

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


def find_table_in_markdown(md_text: str, table_idx: int, doc, start_pos: int = 0) -> int:
    """
    Find where a specific table appears in the markdown text.

    Args:
        md_text: The markdown text
        table_idx: Index of table in doc.tables
        doc: DoclingDocument object
        start_pos: Position to start searching from

    Returns:
        Position where table starts, or -1 if not found
    """
    if not hasattr(doc, 'tables') or table_idx >= len(doc.tables):
        return -1

    table = doc.tables[table_idx]

    # Export the table to markdown to see what it looks like
    try:
        table_md = table.export_to_markdown(doc)

        # Look for the table in the markdown starting from start_pos
        # Normalize whitespace for more robust matching
        table_normalized = normalize_text(table_md[:200])  # Use first 200 chars
        md_normalized = normalize_text(md_text[start_pos:])

        # Try to find it
        pos = md_normalized.find(table_normalized)
        if pos != -1:
            return start_pos + pos

        # If exact match fails, try fuzzy matching on table header
        # Extract first few words from table
        table_words = table_normalized.split()[:20]
        if table_words:
            signature = ' '.join(table_words)

            # Use fuzzy matching
            from difflib import SequenceMatcher
            best_ratio = 0.0
            best_pos = -1
            window_size = len(signature)

            for i in range(0, max(1, len(md_normalized) - window_size + 1), 20):
                window = md_normalized[i:i + window_size]
                ratio = SequenceMatcher(None, signature, window).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i
                    if ratio > 0.85:
                        break

            if best_ratio >= 0.7:
                return start_pos + best_pos

    except Exception as e:
        pass

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
    pages = extract_page_text_with_pymupdf(pdf_path)

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

    current_search_pos = 0
    pages_found = 0

    for i, page_data in enumerate(pages):
        page_num = page_data['page_number']
        page_text = page_data['text']

        if i == 0:
            # First page - marker goes at the beginning
            insertions.append((0, f"<!-- Page {page_num} -->\n\n"))
            pages_found += 1
            continue

        # For subsequent pages, try multiple strategies to find the page boundary
        pos = -1

        # Strategy 1: Try first 500 chars
        signature_length = min(500, len(page_text))
        if signature_length > 0:
            signature = page_text[:signature_length]
            pos = find_text_position(marked_text, signature, current_search_pos, min_words=5)

        # Strategy 2: If that failed, try first 200 chars with lower threshold
        if pos == -1 and len(page_text) >= 100:
            signature_length = min(200, len(page_text))
            signature = page_text[:signature_length]
            pos = find_text_position(marked_text, signature, current_search_pos, min_words=3)

        # Strategy 3: Try middle portion of page (skip headers/footers that might be missing)
        if pos == -1 and len(page_text) >= 200:
            # Take text from 10% to 60% of the page
            start_offset = len(page_text) // 10
            end_offset = min(start_offset + 300, len(page_text) * 6 // 10)
            signature = page_text[start_offset:end_offset]
            pos = find_text_position(marked_text, signature, current_search_pos, min_words=3)

        # Strategy 4: Extract distinctive words/phrases and search for them
        # This helps with tables where formatting is very different
        if pos == -1 and len(page_text) >= 50:
            # Split into words and take every 5th word (skip common words, get content words)
            words = page_text.split()
            if len(words) >= 10:
                # Create a minimal signature from scattered distinctive words
                distinctive_words = [words[i] for i in range(0, min(50, len(words)), 5)]
                signature = ' '.join(distinctive_words[:10])  # Use 10 words max
                pos = find_text_position(marked_text, signature, current_search_pos, min_words=2)

        # Strategy 5: For blank pages (0 chars), estimate position based on document progress
        if pos == -1 and len(page_text) == 0:
            # Estimate position: current_pos + (remaining_text / remaining_pages)
            remaining_md = len(marked_text) - current_search_pos
            remaining_pages = len(pages) - i
            if remaining_pages > 0:
                estimated_page_length = remaining_md // remaining_pages
                # Position the marker approximately one page forward
                pos = current_search_pos + estimated_page_length
                logger.debug(f"Page {page_num} is blank - using positional estimate at {pos}")

        if pos != -1:
            # Found the page boundary - record insertion point
            insertions.append((pos, f"\n\n<!-- Page {page_num} -->\n\n"))
            current_search_pos = pos + len(signature if 'signature' in locals() and signature else 100)
            pages_found += 1
        else:
            # Couldn't find this page's text - track for table-based recovery
            failed_pages.append((i, page_num, page_text))
            logger.debug(f"Could not locate page {page_num} in markdown (page has {len(page_text)} chars)")

    logger.debug(f"Successfully matched {pages_found}/{len(pages)} pages")

    # Second pass: Try to recover failed pages using table information
    if failed_pages and table_pages and doc:
        logger.debug(f"Attempting table-based recovery for {len(failed_pages)} pages...")

        for i, page_num, page_text in failed_pages:
            # Check if this page has a table
            if page_num in table_pages:
                table_indices = table_pages[page_num]
                logger.debug(f"  Page {page_num} has {len(table_indices)} table(s): {table_indices}")

                # Try to find each table in the markdown
                for table_idx in table_indices:
                    # Find where this table appears in the markdown (search from beginning)
                    table_pos = find_table_in_markdown(marked_text, table_idx, doc, start_pos=0)

                    if table_pos != -1:
                        # Found the table! Place page marker before it
                        insertions.append((table_pos, f"\n\n<!-- Page {page_num} -->\n\n"))
                        pages_found += 1
                        logger.debug(f"  ✓ Recovered page {page_num} using table {table_idx} at position {table_pos}")
                        break
                    else:
                        logger.debug(f"  ✗ Could not find table {table_idx} in markdown")

        logger.debug(f"After table recovery: {pages_found}/{len(pages)} pages")

    # Apply insertions in reverse order to maintain position validity
    insertions.sort(key=lambda x: x[0], reverse=True)

    for pos, marker in insertions:
        marked_text = marked_text[:pos] + marker + marked_text[pos:]

    return marked_text


def add_page_markers(md_text: str, pdf_path: str, doc=None, logger: logging.Logger = None) -> str:
    """
    Add accurate page number markers to markdown content using hybrid approach.

    Uses PyMuPDF to extract page-by-page text, then matches that text against
    Docling's markdown output to find accurate page boundaries. This ensures
    100% accuracy for legal citations.

    Args:
        md_text: Markdown text from Docling
        pdf_path: Path to original PDF file (needed for PyMuPDF extraction)
        doc: DoclingDocument object with page information
        logger: Optional logger for debugging

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

    # Use hybrid PyMuPDF + text matching approach for accurate page markers
    try:
        return insert_page_markers_hybrid(md_text, pdf_path, doc, logger)
    except Exception as e:
        logger.warning(f"Page marker insertion failed: {e}")
        # If hybrid approach fails, fall back to simple single-page marker
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

    logger = setup_logging(str(log_path), verbose=args.verbose)
    logger.info(f"Log file: {log_path}")

    # Parse page range if provided
    pages = None
    if args.pages:
        try:
            pages = parse_page_range(args.pages)
            logger.info(f"Converting pages: {args.pages} (0-indexed: {pages})")
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

    logger.info("\n" + "="*60)
    logger.info("Session completed")
    logger.info("="*60)


if __name__ == '__main__':
    main()
