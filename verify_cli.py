#!/usr/bin/env python3
"""verify_cli.py — consolidated entry point for verification.

Two subcommands:
  - content : compare PDF and markdown by page count, word/char retention,
              tables, image-heavy pages. (Was: verify_conversion.py.)
  - markers : sample page markers and fuzzy-match against PyMuPDF page
              text to verify position accuracy. (Was: verify_page_markers.py.)

Stage 5 of the materials-md refactor consolidates the two scripts under one
CLI entry. The original scripts are kept as deprecation shims; they will be
removed in a follow-up cycle.
"""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify conversion output against source files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_content = sub.add_parser(
        "content",
        help="Coarse content verification: page/word/char retention, "
             "table/image counts.",
    )
    p_content.add_argument("source", help="Source file (PDF) or directory")
    p_content.add_argument("output", nargs="?", help="Output markdown file")
    p_content.add_argument("--batch", action="store_true",
                           help="Batch mode: source is a directory of markdown files")
    p_content.add_argument("--pdf-dir", help="Directory of source PDFs for --batch")
    p_content.add_argument("--pattern", default="*.md",
                           help="Glob pattern for batch (default: *.md)")
    p_content.add_argument("-v", "--verbose", action="store_true")

    p_markers = sub.add_parser(
        "markers",
        help="Page-marker accuracy: sample markers, extract surrounding text, "
             "fuzzy-match against PyMuPDF page text.",
    )
    p_markers.add_argument("pdf", help="Source PDF file")
    p_markers.add_argument("md", help="Output markdown file")
    p_markers.add_argument("--sample-size", type=int, default=50,
                           help="Number of markers to sample (default 50)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "content":
        from verify_conversion import verify_conversion, batch_verify, print_report
        if args.batch:
            result = batch_verify(args.source, pdf_dir=args.pdf_dir, pattern=args.pattern)
            return 0 if result.get("status") != "fail" else 1
        if args.output is None:
            print("Error: content mode requires both source and output paths "
                  "(or --batch with a directory)", file=sys.stderr)
            return 2
        report = verify_conversion(args.source, args.output, verbose=args.verbose)
        print_report(args.source, args.output, report)
        return 0 if report.get("status") in ("PASSED", "WARNING") else 1

    if args.command == "markers":
        from verify_page_markers import verify_page_markers
        verify_page_markers(args.pdf, args.md, sample_size=args.sample_size)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
