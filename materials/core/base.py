"""Base types for converter modules.

Each format-specific converter (PDF, DOCX, PPTX, HTML) subclasses BaseConverter
and consumes a ConversionOptions, producing a ConversionResult. Stage 1
introduces this abstraction; stages 2-4 add subclasses.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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
    workers: int = 1  # Batch mode: parallel worker processes (1 = serial)


@dataclass
class ConversionResult:
    """Outcome of a single-file conversion.

    The `statistics` dict has a canonical schema across formats so batch
    summaries don't KeyError when mixing PDF, DOCX, PPTX, and HTML results.
    Format-inapplicable fields are left absent (not None) so callers can use
    `.get(key, default)`. Canonical keys:

    Cross-format
    ------------
    - words: int — output word count
    - characters: int — output character count
    - source_words: int — input word count (after any pre-cleaning)
    - verify_status: str — "PASS" | "WARN" | "FAIL" (cheap-check overall)

    Format-specific (present only when applicable)
    ----------------------------------------------
    - pages: int — PDF page count
    - first_page, last_page: str — PDF page-label range
    - blank_pages: int — PDF blank-page count
    - pages_marked: int — PDF pages with position markers
    - sections: int — HTML/DOCX section-marker count
    - slides: int — PPTX slide count
    - notes_slides: int — PPTX slides bearing speaker notes
    - headings: int — heading count
    - tables: int — table count
    - comments: int — DOCX comment count (when --full)
    - footnotes: int — DOCX footnote count
    """

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
        (writing markdown there) and return a ConversionResult.

        Implementations MUST NOT raise on user-input errors (missing file,
        unparseable content); return `ConversionResult(status="error", error=...)`
        instead so callers can aggregate failures across batches."""

    def supports(self, path: str | Path) -> bool:
        """True iff this converter handles the file's extension."""
        ext = Path(path).suffix.lower()
        return ext in self.extensions

    def convert_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        recursive: bool = False,
        save_report: bool = False,
        page_markers: bool = True,
        workers: int = 1,
        options: Optional["ConversionOptions"] = None,
    ) -> Dict[str, Any]:
        """Convert every file in `input_dir` whose extension this converter
        supports. Returns `{success_count, error_count, reports}`. Set
        `workers > 1` for ProcessPoolExecutor parallelism (each worker pays
        per-process model-warmup cost; useful for large batches).

        `options`: when provided, all format-specific flags are forwarded to
        workers verbatim (with output_path and workers overridden). When
        None, a minimal ConversionOptions is constructed from the scalar
        parameters (backward-compatible).
        """
        from materials.core.output import default_output_path  # local import to avoid cycle

        in_dir = Path(input_dir)
        if not in_dir.is_dir():
            return {"success_count": 0, "error_count": 1, "reports": [],
                    "error": f"Not a directory: {input_dir}"}

        glob = "**/*" if recursive else "*"
        candidates = [
            str(p) for p in in_dir.glob(glob)
            if p.is_file() and p.suffix.lower() in self.extensions
        ]

        if workers > 1 and len(candidates) > 1:
            from materials.core.parallel import parallel_convert_files
            if options is None:
                # Backward-compat: build minimal options from scalar params
                opts = ConversionOptions(
                    output_path=output_dir,
                    page_markers=page_markers,
                    save_report=save_report,
                    workers=workers,
                )
            else:
                # Caller provided full options; pin output_path and workers
                opts = replace(options, output_path=output_dir, workers=workers)
            return parallel_convert_files(
                self.__class__, candidates, opts, workers,
                input_dir=str(in_dir), recursive=recursive,
            )

        success_count = 0
        error_count = 0
        reports: List[Dict[str, Any]] = []
        for src in candidates:
            out_override = None
            if output_dir:
                out_override = str(Path(output_dir) / Path(src).with_suffix(".md").name)
            opts = ConversionOptions(output_path=out_override, page_markers=page_markers)
            result = self.convert(src, opts)
            if result.status == "success":
                success_count += 1
            else:
                error_count += 1
            reports.append({
                "input": src,
                "status": result.status,
                "output_file": result.output_file,
                "error": result.error,
                "statistics": result.statistics,
            })

        return {
            "success_count": success_count,
            "error_count": error_count,
            "reports": reports,
        }
