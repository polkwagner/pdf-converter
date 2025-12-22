#!/usr/bin/env python3
"""
verify_conversion.py

Verify that PDF to Markdown conversion captured all content by comparing:
- Page counts
- Word counts
- Character counts
- Table detection
- Image detection

Usage:
    python verify_conversion.py <pdf_file> <markdown_file>
    python verify_conversion.py --batch <directory>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Dict
import re

try:
    import fitz  # PyMuPDF for verification
except ImportError:
    print("ERROR: PyMuPDF not installed.")
    print("Install with: pip install PyMuPDF")
    sys.exit(1)


def extract_pdf_stats(pdf_path: str) -> Dict:
    """
    Extract statistics from PDF for verification.

    Returns dict with:
        - page_count
        - char_count (approximate)
        - word_count (approximate)
        - table_count (rough estimate)
        - image_count
    """
    doc = fitz.open(pdf_path)

    stats = {
        'page_count': len(doc),
        'char_count': 0,
        'word_count': 0,
        'table_count': 0,
        'image_count': 0,
        'has_scanned_pages': False
    }

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        stats['char_count'] += len(text)
        stats['word_count'] += len(text.split())

        # Detect potential tables (rough heuristic: look for grid-like structures)
        # This is a simple check - Docling does better detection
        try:
            tables = page.find_tables()
            if tables and hasattr(tables, 'tables'):
                stats['table_count'] += len(tables.tables)
        except:
            # Table detection not available in all PyMuPDF versions
            pass

        # Count images
        images = page.get_images()
        stats['image_count'] += len(images)

        # Check for scanned pages (image-heavy, text-light)
        if len(images) > 0 and len(text.strip()) < 100:
            stats['has_scanned_pages'] = True

    doc.close()
    return stats


def extract_markdown_stats(md_path: str) -> Dict:
    """
    Extract statistics from markdown file.

    Returns dict with:
        - char_count
        - word_count
        - heading_count
        - table_count (by counting markdown tables)
        - code_block_count
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count markdown tables (lines starting with |)
    table_lines = [line for line in content.split('\n') if line.strip().startswith('|')]
    # Rough estimate: each table has at least 3 lines (header, separator, data)
    table_count = len([i for i in range(len(table_lines)-2)
                      if table_lines[i].startswith('|') and
                      table_lines[i+1].startswith('|') and
                      '---' in table_lines[i+1]])

    # Count headings
    heading_count = len([line for line in content.split('\n')
                        if line.strip().startswith('#')])

    # Count code blocks (might include tables or other special content)
    code_blocks = content.count('```')

    stats = {
        'char_count': len(content),
        'word_count': len(content.split()),
        'heading_count': heading_count,
        'table_count': table_count,
        'code_block_count': code_blocks // 2  # Opening and closing
    }

    return stats


def verify_conversion(pdf_path: str, md_path: str, verbose: bool = True) -> Dict:
    """
    Verify conversion completeness by comparing PDF and markdown stats.

    Returns verification report dict with status and metrics.
    """
    if not os.path.exists(pdf_path):
        return {'status': 'error', 'message': f'PDF not found: {pdf_path}'}

    if not os.path.exists(md_path):
        return {'status': 'error', 'message': f'Markdown not found: {md_path}'}

    # Extract stats
    pdf_stats = extract_pdf_stats(pdf_path)
    md_stats = extract_markdown_stats(md_path)

    # Calculate ratios
    char_ratio = md_stats['char_count'] / pdf_stats['char_count'] if pdf_stats['char_count'] > 0 else 0
    word_ratio = md_stats['word_count'] / pdf_stats['word_count'] if pdf_stats['word_count'] > 0 else 0

    # Determine status
    issues = []
    warnings = []

    # Check word count (should be within reasonable range)
    # Markdown often has more words due to formatting, but shouldn't be drastically less
    if word_ratio < 0.70:
        issues.append(f"Word count suspiciously low: {word_ratio:.1%} of PDF")
    elif word_ratio < 0.85:
        warnings.append(f"Word count lower than expected: {word_ratio:.1%} of PDF")

    # Check character count
    if char_ratio < 0.60:
        issues.append(f"Character count very low: {char_ratio:.1%} of PDF")

    # Check for scanned pages
    if pdf_stats['has_scanned_pages']:
        warnings.append("PDF contains image-heavy pages (possibly scanned). Consider using --ocr flag.")

    # Check table conversion
    if pdf_stats['table_count'] > 0 and md_stats['table_count'] == 0:
        warnings.append(f"PDF has {pdf_stats['table_count']} tables, but none found in markdown")

    # Determine overall status
    if issues:
        status = 'FAILED'
    elif warnings:
        status = 'WARNING'
    else:
        status = 'PASSED'

    report = {
        'status': status,
        'pdf_stats': pdf_stats,
        'md_stats': md_stats,
        'ratios': {
            'char_ratio': char_ratio,
            'word_ratio': word_ratio
        },
        'issues': issues,
        'warnings': warnings
    }

    if verbose:
        print_report(pdf_path, md_path, report)

    return report


def print_report(pdf_path: str, md_path: str, report: Dict):
    """Print formatted verification report."""
    status = report['status']
    status_symbol = {
        'PASSED': '✓',
        'WARNING': '⚠',
        'FAILED': '✗'
    }.get(status, '?')

    print(f"\n{'='*70}")
    print(f"Verification Report: {status_symbol} {status}")
    print(f"{'='*70}")
    print(f"PDF:      {Path(pdf_path).name}")
    print(f"Markdown: {Path(md_path).name}")
    print()

    # PDF Stats
    pdf_stats = report['pdf_stats']
    print("PDF Statistics:")
    print(f"  Pages:      {pdf_stats['page_count']}")
    print(f"  Characters: {pdf_stats['char_count']:,}")
    print(f"  Words:      {pdf_stats['word_count']:,}")
    print(f"  Tables:     {pdf_stats['table_count']}")
    print(f"  Images:     {pdf_stats['image_count']}")
    print()

    # Markdown Stats
    md_stats = report['md_stats']
    print("Markdown Statistics:")
    print(f"  Characters: {md_stats['char_count']:,}")
    print(f"  Words:      {md_stats['word_count']:,}")
    print(f"  Headings:   {md_stats['heading_count']}")
    print(f"  Tables:     {md_stats['table_count']}")
    print()

    # Ratios
    ratios = report['ratios']
    print("Conversion Ratios:")
    print(f"  Character retention: {ratios['char_ratio']:.1%}")
    print(f"  Word retention:      {ratios['word_ratio']:.1%}")
    print()

    # Issues and Warnings
    if report['issues']:
        print("❌ ISSUES:")
        for issue in report['issues']:
            print(f"  - {issue}")
        print()

    if report['warnings']:
        print("⚠️  WARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning}")
        print()

    if status == 'PASSED':
        print("✅ Conversion appears complete and accurate!")
    print(f"{'='*70}\n")


def batch_verify(directory: str, pdf_dir: str = None, pattern: str = "*.md") -> Dict:
    """
    Verify all markdown files in a directory against their source PDFs.
    Assumes PDF has same name as markdown file.

    Args:
        directory: Directory containing markdown files
        pdf_dir: Optional directory containing source PDFs (if different from markdown dir)
        pattern: Glob pattern for markdown files
    """
    md_dir = Path(directory)
    md_files = list(md_dir.glob(pattern))

    if not md_files:
        print(f"No markdown files found in {directory}")
        return {}

    results = {}
    passed = 0
    warnings = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"Batch Verification: {len(md_files)} file(s)")
    print(f"{'='*70}\n")

    for md_file in md_files:
        # Try to find corresponding PDF
        if pdf_dir:
            # Look in specified PDF directory
            pdf_file = Path(pdf_dir) / f"{md_file.stem}.pdf"
        else:
            # Look in same directory as markdown
            pdf_file = md_file.with_suffix('.pdf')

        # If not found, check parent dir or common source locations
        if not pdf_file.exists():
            # Check if there's a common pattern (e.g., output_dir vs input_dir)
            possible_paths = [
                Path(md_file.parent.parent) / md_file.stem / f"{md_file.stem}.pdf",
                Path(md_file.parent.parent) / f"{md_file.stem}.pdf",
            ]
            for p in possible_paths:
                if p.exists():
                    pdf_file = p
                    break

        if not pdf_file.exists():
            print(f"⚠️  Skipping {md_file.name}: PDF not found")
            continue

        report = verify_conversion(str(pdf_file), str(md_file), verbose=False)
        results[str(md_file)] = report

        # Print summary line
        status = report['status']
        symbol = {'PASSED': '✓', 'WARNING': '⚠', 'FAILED': '✗'}.get(status, '?')
        word_ratio = report['ratios']['word_ratio']
        print(f"{symbol} {md_file.name:50s} | Words: {word_ratio:5.1%} | {status}")

        if status == 'PASSED':
            passed += 1
        elif status == 'WARNING':
            warnings += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"Summary: {passed} passed, {warnings} warnings, {failed} failed")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Verify PDF to Markdown conversion completeness',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input',
        nargs='?',
        help='PDF file, Markdown file, or directory (with --batch)'
    )

    parser.add_argument(
        'markdown',
        nargs='?',
        help='Markdown file to verify against PDF'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Verify all markdown files in directory'
    )

    parser.add_argument(
        '--pdf-dir',
        help='Directory containing source PDF files (for batch mode)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    if args.batch:
        if not args.input:
            print("Error: Directory required for batch verification")
            sys.exit(1)
        batch_verify(args.input, pdf_dir=args.pdf_dir)
    else:
        if not args.input or not args.markdown:
            print("Error: Both PDF and Markdown files required")
            print("Usage: python verify_conversion.py <pdf_file> <markdown_file>")
            sys.exit(1)

        report = verify_conversion(args.input, args.markdown, verbose=not args.quiet)

        # Exit with error code if verification failed
        if report['status'] == 'FAILED':
            sys.exit(1)


if __name__ == '__main__':
    main()
