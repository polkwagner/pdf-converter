"""Base types for converter modules.

Each format-specific converter (PDF, DOCX, PPTX, HTML) subclasses BaseConverter
and consumes a ConversionOptions, producing a ConversionResult. Stage 1
introduces this abstraction; stages 2-4 add subclasses.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConversionOptions:
    """Options that apply to any conversion. Format-specific extras live here too;
    each converter ignores options it doesn't understand. Lean default = the AI-
    ingestion case from the spec."""

    output_path: Optional[str] = None
    page_markers: bool = True
    pages: Optional[List[int]] = None
    ocr: bool = False
    extract_images: bool = False
    quiet: bool = False
    verbose: bool = False
    save_report: bool = False
    log_path: Optional[str] = None
    # Format-specific (used in later stages — declared here so options are uniform)
    full: bool = False  # DOCX: include comments appendix
    show_revisions: bool = False  # DOCX: render tracked changes
    keep_images: bool = False  # DOCX: extract images to disk
    notes_only: bool = False  # PPTX: emit speaker-notes transcript only
    strip_html_noise: bool = False  # HTML: bs4 nav/script pre-strip


@dataclass
class ConversionResult:
    """Outcome of a single-file conversion."""

    status: str  # "success" or "error"
    output_file: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    conversion_time: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_legacy(cls, report: Dict[str, Any]) -> "ConversionResult":
        """Adapt the legacy convert_pdf_to_markdown report dict into this shape."""
        return cls(
            status=report.get("status", "error"),
            output_file=report.get("output_file"),
            statistics=report.get("statistics", {}) or {},
            conversion_time=report.get("conversion_time", 0.0),
            error=report.get("error"),
        )


class BaseConverter(ABC):
    """Abstract base for format converters. One subclass per supported format."""

    extensions: tuple[str, ...] = ()  # e.g., (".pdf",) — used for dispatch

    @abstractmethod
    def convert(self, input_path: str, options: ConversionOptions) -> ConversionResult:
        """Convert one input file. Implementations must respect options.output_path
        (writing markdown there) and return a ConversionResult."""

    def supports(self, path: str | Path) -> bool:
        """True iff this converter handles the file's extension."""
        ext = Path(path).suffix.lower()
        return ext in self.extensions
