#!/usr/bin/env python3
"""convert.py — unified materials-md CLI.

Auto-detects format from the input file's extension and dispatches to the
appropriate converter. Stage 2 supports PDF and HTML; subsequent stages add
DOCX and PPTX.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from materials.core.base import ConversionOptions
from materials.formats.pdf import (
    PDFConverter,
    RICH_AVAILABLE,
    parse_page_range,
    setup_logging,
)
from materials.formats.html import HTMLConverter

# Extension → converter instance. Stage 2-4 register more entries here.
REGISTRY = {}
for converter in (PDFConverter(), HTMLConverter()):
    for ext in converter.extensions:
        REGISTRY[ext] = converter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert documents to markdown optimized for AI ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s chapter1.pdf -o chapter1.md\n"
            "  %(prog)s ./casebooks/ --batch\n"
            "  %(prog)s casebook.pdf --pages 1-50 -o excerpt.md\n"
        ),
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output markdown file or directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch convert PDF files in a directory "
                             "(HTML/DOCX/PPTX batch support coming in stage 5)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into subdirectories (with --batch)")
    parser.add_argument("--pages",
                        help='Page range (e.g., "1-10" or "1,3,5-8"). PDF only. 1-indexed.')
    parser.add_argument("--images", action="store_true",
                        help="Extract embedded images (PDF only in stage 1)")
    parser.add_argument("--ocr", action="store_true",
                        help="OCR for scanned PDFs (slow; PDF only)")
    parser.add_argument("--no-page-markers", dest="page_markers", action="store_false",
                        help="Disable position markers (enabled by default)")
    parser.add_argument(
        "--strip-html-noise",
        action="store_true",
        help="Strip <script>, <style>, <nav>, <footer>, <aside> from HTML before "
             "conversion. Requires beautifulsoup4. (HTML only.)",
    )
    parser.add_argument("--save-report", action="store_true",
                        help="Save detailed conversion report JSON (batch mode)")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose (DEBUG) logging")
    return parser


def _dispatch_single(input_path: str, args: argparse.Namespace) -> int:
    ext = Path(input_path).suffix.lower()
    converter = REGISTRY.get(ext)
    if converter is None:
        supported = ", ".join(sorted(REGISTRY.keys()))
        print(
            f"Error: unsupported extension {ext!r}. Supported: {supported}",
            file=sys.stderr,
        )
        return 2

    pages = parse_page_range(args.pages) if args.pages else None

    options = ConversionOptions(
        output_path=args.output,
        page_markers=args.page_markers,
        pages=pages,
        ocr=args.ocr,
        extract_images=args.images,
        quiet=False,
        verbose=args.verbose,
        strip_html_noise=args.strip_html_noise,
    )
    result = converter.convert(input_path, options)
    return 0 if result.status == "success" else 1


def _dispatch_batch(input_dir: str, args: argparse.Namespace) -> int:
    # Stage 1 batch is PDF-only — same as legacy.
    converter = PDFConverter()
    converter.convert_directory(
        input_dir,
        output_dir=args.output,
        recursive=args.recursive,
        save_report=args.save_report,
        page_markers=args.page_markers,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return 1

    # Logging setup mirrors the legacy main() exactly so output paths match.
    if args.log_file:
        log_path = args.log_file
    elif args.batch or os.path.isdir(args.input):
        out_loc = Path(args.output) if args.output else Path(args.input) / "converted"
        out_loc.mkdir(parents=True, exist_ok=True)
        log_path = out_loc / "conversion.log"
    else:
        out_loc = Path(args.output).parent if args.output else Path(args.input).parent / "converted"
        out_loc.mkdir(parents=True, exist_ok=True)
        log_path = out_loc / "conversion.log"

    setup_logging(str(log_path), verbose=args.verbose, use_rich=RICH_AVAILABLE)

    if args.batch or os.path.isdir(args.input):
        return _dispatch_batch(args.input, args)
    return _dispatch_single(args.input, args)


if __name__ == "__main__":
    sys.exit(main())
