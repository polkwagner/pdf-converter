"""Logging setup and console-availability detection — shared across the CLI.

Lives in `materials/core/` because logging is CLI infrastructure, not a
property of any one format. Previously these symbols re-exported from
`materials/formats/pdf.py`; that inversion is gone post Rex review.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional


# Detect whether the Rich UX layer is importable. The `console.py` module at
# the repo root provides Rich-formatted progress and panels; if Rich isn't
# installed (or `console.py` is unavailable for any reason), converters fall
# back to plain logging output. The detection happens once at import time.
try:
    import console  # noqa: F401 — only imported to test availability
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(
    log_file: Optional[str] = None,
    verbose: bool = False,
    use_rich: bool = False,
) -> logging.Logger:
    """Configure the `pdf_converter` logger for a CLI session.

    Args:
        log_file: Optional path to a log file (DEBUG-level entries appended).
        verbose: If True, the logger captures DEBUG messages; otherwise INFO.
        use_rich: If True, suppress the stdout console handler so Rich panels
                  own the terminal. If False, INFO messages stream to stdout.
    """
    logger = logging.getLogger("pdf_converter")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []

    if not use_rich:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

        # Session header — written to the log file only, not the console.
        for line in (
            "=" * 60,
            "materials-md converter — session started",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
        ):
            file_handler.emit(logging.LogRecord(
                "pdf_converter", logging.INFO, "", 0, line, (), None,
            ))

    return logger
