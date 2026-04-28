"""Output path helpers and shared formatting utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Spec §5.2 — sanitize heading text before embedding in <!-- Section K: ... --> markers.
_DASH_RUN = re.compile(r"-{2,}")
_WS_RUN = re.compile(r"\s+")
_MAX_HEADING_LEN = 80


def sanitize_heading_text(text: str) -> str:
    """Make a heading safe for inclusion inside an HTML comment marker.

    Rules:
      1. Collapse runs of two-or-more `-` to a single `-`.
      2. Strip backticks.
      3. Strip newlines/tabs and collapse internal whitespace.
      4. Truncate to 80 characters (append U+2026 if truncated).
      5. If empty (or only punctuation like a lone `-`) after sanitization,
         return "(untitled)".
    """
    s = text.replace("`", "")
    s = s.replace("\n", " ").replace("\t", " ")
    s = _WS_RUN.sub(" ", s).strip()
    s = _DASH_RUN.sub("-", s)
    if len(s) > _MAX_HEADING_LEN:
        s = s[:_MAX_HEADING_LEN].rstrip() + "…"
    # Treat lone or all-dash-after-collapse strings as empty: an input of
    # "---" or "-----" carries no semantic heading content, just a divider.
    if not s or s.strip("-").strip() == "":
        return "(untitled)"
    return s


def default_output_path(input_path: str, override: Optional[str]) -> Path:
    """Stage 1 PDF behavior preserved exactly: if no override, write to
    `<input_dir>/converted/<input_stem>.md`."""
    if override:
        return Path(override)
    src = Path(input_path)
    out_dir = src.parent / "converted"
    out_dir.mkdir(exist_ok=True)
    return out_dir / src.with_suffix(".md").name


def default_log_path(output_location: Path) -> Path:
    """Stage 1 PDF behavior preserved exactly: log file lives next to the markdown."""
    output_location.mkdir(parents=True, exist_ok=True)
    return output_location / "conversion.log"
