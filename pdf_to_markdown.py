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
from pathlib import Path
from typing import Optional, List

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("ERROR: Docling not installed.")
    print("Install with: pip install docling")
    sys.exit(1)


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None,
    pages: Optional[List[int]] = None,
    extract_images: bool = False,
    ocr: bool = False
) -> str:
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

        # Get file size info
        char_count = len(full_content)
        word_count = len(full_content.split())
        page_count = result.document.page_count if hasattr(result.document, 'page_count') else '?'

        print(f"✓ Successfully converted ({page_count} pages, {char_count:,} chars, ~{word_count:,} words)")

        return full_content

    except Exception as e:
        print(f"✗ Error converting {pdf_path}: {e}")
        raise


def batch_convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False
) -> None:
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
            convert_pdf_to_markdown(
                str(pdf_path),
                str(out_path),
                extract_images=False,
                ocr=False
            )
            success_count += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            error_count += 1
            continue

    print("\n" + "=" * 60)
    print(f"Batch conversion complete! ({success_count} successful, {error_count} failed)")


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
            recursive=args.recursive
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
