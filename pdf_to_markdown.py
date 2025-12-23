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

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("ERROR: Docling not installed.")
    print("Install with: pip install docling")
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
    page_markers: bool = False,
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
        page_markers: Whether to add page number markers to output (default: False)

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

        # Export to markdown (with internal page break markers if multi-page)
        md_text = result.document.export_to_markdown()

        # Add page markers if requested
        if page_markers:
            md_text = add_page_markers(md_text)

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
    page_markers: bool = False,
    logger: logging.Logger = None
) -> Dict:
    """
    Convert all PDF files in a directory to markdown.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Optional output directory. If None, outputs alongside PDFs.
        recursive: Whether to process subdirectories
        save_report: Whether to save JSON report
        page_markers: Whether to add page number markers to output
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


def add_page_markers(md_text: str) -> str:
    """
    Add page number markers to markdown content.

    Replaces Docling's internal page break markers with formatted HTML comments
    like <!-- Page N --> where N is the actual PDF page number (1-indexed).

    Args:
        md_text: Markdown text potentially containing internal markers

    Returns:
        Markdown text with page markers added
    """
    import re

    # Pattern to match Docling's internal page break markers
    # Format: #_#_DOCLING_DOC_PAGE_BREAK_<prev_page>_<next_page>_#_#
    # where prev_page and next_page are 0-indexed
    pattern = r"#_#_DOCLING_DOC_PAGE_BREAK_(\d+)_(\d+)_#_#"

    # Find all markers to determine first page
    matches = list(re.finditer(pattern, md_text))

    if not matches:
        # No page breaks found, assume single page document
        return f"<!-- Page 1 -->\n\n{md_text}"

    # Extract first page number from first marker's prev_page
    first_match = matches[0]
    first_page = int(first_match.group(1)) + 1  # Convert to 1-indexed

    # Replace all markers with page-specific comments
    def replacement(match):
        next_page = int(match.group(2))
        # next_page is 0-indexed, convert to 1-indexed PDF page number
        return f"<!-- Page {next_page + 1} -->"

    md_text = re.sub(pattern, replacement, md_text)

    # Add marker for first page at document start
    md_text = f"<!-- Page {first_page} -->\n\n{md_text}"

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
        '--page-markers',
        action='store_true',
        help='Add page number markers (<!-- Page N -->) to output'
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
