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
from pathlib import Path
from typing import Optional, List, Dict

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("ERROR: Docling not installed.")
    print("Install with: pip install docling")
    sys.exit(1)


def print_conversion_report(report: Dict):
    """Print formatted conversion report."""
    print("\n" + "="*60)
    print("📄 CONVERSION REPORT")
    print("="*60)

    if report['status'] == 'success':
        stats = report['statistics']
        print(f"✓ Status:     SUCCESS")
        print(f"⏱  Time:       {report['conversion_time']}s")
        print(f"📊 Statistics:")
        print(f"   Pages:     {stats['pages']}")
        print(f"   Words:     {stats['words']:,}")
        print(f"   Characters: {stats['characters']:,}")
        print(f"   Headings:  {stats['headings']}")
        print(f"   Tables:    {stats['tables']}")
        print(f"📁 Output:    {Path(report['output_file']).name}")
    else:
        print(f"✗ Status:     FAILED")
        print(f"⏱  Time:       {report['conversion_time']}s")
        print(f"❌ Error:     {report['error']}")

    print("="*60 + "\n")


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None,
    pages: Optional[List[int]] = None,
    extract_images: bool = False,
    ocr: bool = False,
    report: bool = True
) -> Dict:
    """
    Convert a PDF file to markdown format using Docling.

    Args:
        pdf_path: Path to the input PDF file
        output_path: Optional path for output markdown file. If None, uses same name as PDF.
        pages: Optional list of page numbers to convert (0-indexed)
        extract_images: Whether to extract and save images (default: False for text-only)
        ocr: Whether to use OCR for scanned documents (default: False)

    Returns:
        str: The generated markdown content
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Determine output path
    if output_path is None:
        output_path = Path(pdf_path).with_suffix('.md')

    print(f"Converting: {pdf_path}")
    print(f"Output: {output_path}")

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
            print_conversion_report(conversion_report)

        return conversion_report

    except Exception as e:
        error_report = {
            'status': 'error',
            'input_file': pdf_path,
            'error': str(e),
            'conversion_time': round(time.time() - start_time, 2)
        }
        print(f"✗ Error converting {pdf_path}: {e}")
        return error_report


def batch_convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    save_report: bool = False
) -> Dict:
    """
    Convert all PDF files in a directory to markdown.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Optional output directory. If None, outputs alongside PDFs.
        recursive: Whether to process subdirectories
    """
    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # Find all PDFs
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(input_path.glob(pattern))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to convert")
    print("-" * 60)

    success_count = 0
    error_count = 0
    total_time = 0
    total_words = 0
    total_pages = 0
    reports = []

    # Convert each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        # Determine output path
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Preserve subdirectory structure if recursive
            if recursive:
                rel_path = pdf_path.relative_to(input_path)
                out_path = out_dir / rel_path.with_suffix('.md')
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = out_dir / pdf_path.with_suffix('.md').name
        else:
            out_path = pdf_path.with_suffix('.md')

        try:
            report = convert_pdf_to_markdown(
                str(pdf_path),
                str(out_path),
                extract_images=False,
                ocr=False,
                report=False  # Don't print individual reports in batch mode
            )
            reports.append(report)

            if report['status'] == 'success':
                success_count += 1
                total_time += report['conversion_time']
                total_words += report['statistics']['words']
                total_pages += report['statistics']['pages']
                print(f"   ✓ {report['statistics']['pages']} pages, {report['statistics']['words']:,} words, {report['conversion_time']}s")
            else:
                error_count += 1
                print(f"   ✗ Failed: {report.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            error_count += 1
            reports.append({
                'status': 'error',
                'input_file': str(pdf_path),
                'error': str(e)
            })
            continue

    # Print batch summary
    print("\n" + "=" * 60)
    print("📊 BATCH CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Files processed:  {len(pdf_files)}")
    print(f"✓ Successful:     {success_count}")
    print(f"✗ Failed:         {error_count}")
    print(f"⏱  Total time:     {total_time:.1f}s")
    print(f"📄 Total pages:    {total_pages:,}")
    print(f"📝 Total words:    {total_words:,}")
    if success_count > 0:
        print(f"⚡ Avg speed:      {total_time/success_count:.1f}s per file")
    print("=" * 60)

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
        print(f"\n📄 Detailed report saved to: {report_path}")

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
        '--save-report',
        action='store_true',
        help='Save detailed conversion report to JSON file (batch mode only)'
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Parse page range if provided
    pages = None
    if args.pages:
        try:
            pages = parse_page_range(args.pages)
            print(f"Converting pages: {args.pages} (0-indexed: {pages})")
        except Exception as e:
            print(f"Error parsing page range: {e}")
            sys.exit(1)

    # Batch or single file mode
    if args.batch or os.path.isdir(args.input):
        batch_convert_directory(
            args.input,
            output_dir=args.output,
            recursive=args.recursive,
            save_report=args.save_report
        )
    else:
        convert_pdf_to_markdown(
            args.input,
            output_path=args.output,
            pages=pages,
            extract_images=args.images,
            ocr=args.ocr
        )


if __name__ == '__main__':
    main()
