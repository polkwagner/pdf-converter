"""PDFConverter — wraps PDF→markdown logic.

Stage 1 Phase A: this module imports from pdf_to_markdown.py (the existing
top-level script, untouched) so we can wire up the new dispatcher and prove
byte-identical output before moving definitions.
Phase B (Task 11) moves the function bodies into this file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Phase A: delegate to the legacy module. Phase B replaces these imports with
# locally-defined functions.
import pdf_to_markdown as _legacy

from materials.core.base import BaseConverter, ConversionOptions, ConversionResult


class PDFConverter(BaseConverter):
    """Convert PDF files to markdown using Docling + PyMuPDF + RapidFuzz."""

    extensions = (".pdf",)

    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        logger = logging.getLogger("pdf_converter")
        report = _legacy.convert_pdf_to_markdown(
            input_path,
            output_path=options.output_path,
            pages=options.pages,
            extract_images=options.extract_images,
            ocr=options.ocr,
            page_markers=options.page_markers,
            quiet=options.quiet,
            logger=logger,
        )
        return ConversionResult.from_legacy(report)

    def convert_directory(
        self,
        input_dir: str,
        output_dir: Optional[str],
        recursive: bool,
        save_report: bool,
        page_markers: bool,
    ) -> None:
        logger = logging.getLogger("pdf_converter")
        _legacy.batch_convert_directory(
            input_dir,
            output_dir=output_dir,
            recursive=recursive,
            save_report=save_report,
            page_markers=page_markers,
            logger=logger,
        )


# Convenience accessors for legacy helpers used by convert.py during Phase A.
parse_page_range = _legacy.parse_page_range
setup_logging = _legacy.setup_logging
