#!/usr/bin/env python3
"""convert.py — unified materials-md CLI.

Auto-detects format from the input file's extension and dispatches to the
appropriate converter. Stage 2 supports PDF and HTML; subsequent stages add
DOCX and PPTX.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from materials.core.base import ConversionOptions
from materials.core.logging import RICH_AVAILABLE, setup_logging
from materials.formats.docx import DOCXConverter
from materials.formats.html import HTMLConverter
from materials.formats.pdf import PDFConverter, parse_page_range
from materials.formats.pptx import PPTXConverter

# Extension → converter instance. Stage 2-4 register more entries here.
REGISTRY = {}
for converter in (PDFConverter(), HTMLConverter(), DOCXConverter(), PPTXConverter()):
    for ext in converter.extensions:
        REGISTRY[ext] = converter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert documents to markdown optimized for AI ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s chapter1.pdf -o chapter1.md\n"
            "  %(prog)s memo.docx -o memo.md\n"
            "  %(prog)s memo.docx --full -o memo-with-comments.md\n"
            "  %(prog)s deck.pptx -o deck.md\n"
            "  %(prog)s deck.pptx --notes-only -o lecture-transcript.md\n"
            "  %(prog)s article.html -o article.md\n"
            "  %(prog)s ./casebooks/ --batch\n"
            "  %(prog)s casebook.pdf --pages 1-50 -o excerpt.md\n"
            "\n"
            "Replaces pdf_to_markdown.py, which now forwards here.\n"
        ),
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output markdown file or directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch convert every supported file in a directory")
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="DOCX: include comments appendix and inline anchors. Default lean mode "
             "drops comments (footnotes are always preserved).",
    )
    parser.add_argument(
        "--show-revisions",
        action="store_true",
        help="DOCX: render tracked changes as [+ added +] / [- removed -] inline. "
             "Without this flag, revisions are accepted-final.",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="DOCX: extract embedded images to <output>_files/. "
             "Without this flag, images become <!-- image --> placeholders.",
    )
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="PPTX: emit a clean lecture transcript — slide numbers + speaker "
             "notes text only, dropping bullet content. Slides without notes "
             "are skipped.",
    )
    parser.add_argument("--save-report", action="store_true",
                        help="Save detailed conversion report JSON (batch mode)")
    parser.add_argument("--continue-on-error", dest="continue_on_error",
                        action="store_true", default=True,
                        help="In batch mode, continue past file failures (default: True)")
    parser.add_argument("--no-continue-on-error", dest="continue_on_error",
                        action="store_false",
                        help="In batch mode, stop on the first file failure")
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Batch mode: parallel worker processes (default 1 = serial). "
             "Each worker pays per-process model-warmup cost; useful for large batches.",
    )
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose (DEBUG) logging")
    return parser


def _dispatch_single(input_path: str, args: argparse.Namespace) -> int:
    logger = logging.getLogger("pdf_converter")
    ext = Path(input_path).suffix.lower()
    converter = REGISTRY.get(ext)
    if converter is None:
        supported = ", ".join(sorted(REGISTRY.keys()))
        print(
            f"Error: unsupported extension {ext!r}. Supported: {supported}",
            file=sys.stderr,
        )
        return 2

    logger.debug(f"dispatch: {ext!r} → {type(converter).__name__}")

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
        full=args.full,
        show_revisions=args.show_revisions,
        keep_images=args.keep_images,
        notes_only=args.notes_only,
        workers=args.workers,
    )
    result = converter.convert(input_path, options)
    if result.status != "success":
        print(f"Error: {result.error or 'conversion failed'}", file=sys.stderr)
        return 1
    return 0


def _dispatch_batch(input_dir: str, args: argparse.Namespace) -> int:
    """Iterate the REGISTRY: each registered converter processes the files in
    `input_dir` whose extensions it owns. Exits non-zero if any file failed
    (per spec §6.4)."""
    logger = logging.getLogger("pdf_converter")

    # Group converters by identity so we don't double-call when a converter
    # owns multiple extensions (e.g., HTMLConverter handles .html and .htm).
    seen_converters = []
    for converter in REGISTRY.values():
        if converter not in seen_converters:
            seen_converters.append(converter)

    pages = parse_page_range(args.pages) if args.pages else None
    batch_options = ConversionOptions(
        output_path=args.output,
        page_markers=args.page_markers,
        pages=pages,
        ocr=args.ocr,
        extract_images=args.images,
        quiet=False,
        verbose=args.verbose,
        save_report=args.save_report,
        full=args.full,
        show_revisions=args.show_revisions,
        keep_images=args.keep_images,
        notes_only=args.notes_only,
        strip_html_noise=args.strip_html_noise,
        workers=args.workers,
    )

    total_success = 0
    total_errors = 0
    for converter in seen_converters:
        logger.debug(f"batch: invoking {type(converter).__name__}")
        result = converter.convert_directory(
            input_dir,
            output_dir=args.output,
            recursive=args.recursive,
            save_report=args.save_report,
            page_markers=args.page_markers,
            workers=args.workers,
            options=batch_options,
        )
        total_success += result.get("success_count", 0)
        total_errors += result.get("error_count", 0)
        if total_errors > 0 and not args.continue_on_error:
            logger.warning("Stopping batch on first failure (--no-continue-on-error)")
            return 1

    return 0 if total_errors == 0 else 1


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
