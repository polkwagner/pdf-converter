"""HTML conversion tests."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample_article.html"


def _isolated_fixture(tmpdir: Path) -> Path:
    dest = tmpdir / "fixture.html"
    shutil.copy(FIXTURE, dest)
    return dest


def _run_convert(tmpdir: Path, *extra_args: str) -> tuple[Path, subprocess.CompletedProcess]:
    fixture = _isolated_fixture(tmpdir)
    out = tmpdir / "out.md"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "convert.py"),
            str(fixture),
            "-o",
            str(out),
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )
    return out, result


def test_html_converts_without_error(tmp_path):
    """Basic smoke: convert.py handles .html files."""
    out, result = _run_convert(tmp_path)
    assert result.returncode == 0, (
        f"convert.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert out.exists()
    md = out.read_text(encoding="utf-8")
    assert len(md) > 0


def test_html_emits_section_markers(tmp_path):
    """H1 and H2 headings should each get a numbered <!-- Section K: ... --> marker.
    H3 should NOT get a marker."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")

    # The fixture has 1 H1 and 3 H2s, so 4 markers total.
    section_lines = [line for line in md.splitlines() if line.startswith("<!-- Section ")]
    assert len(section_lines) == 4, f"Expected 4 section markers, got {len(section_lines)}: {section_lines}"

    # Numbering must be sequential: 1, 2, 3, 4.
    for i, line in enumerate(section_lines, start=1):
        assert line.startswith(f"<!-- Section {i}:"), (
            f"Marker {i} has wrong number: {line!r}"
        )


def test_html_section_marker_escapes_dashes(tmp_path):
    """Heading 'Why we use --strict mode' must be sanitized to single dash."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "<!-- Section 3: Why we use -strict mode -->" in md, (
        f"Dash escape failed; markers in output:\n"
        + "\n".join(line for line in md.splitlines() if "Section " in line)
    )


def test_html_section_marker_strips_backticks(tmp_path):
    """Heading 'Code example with `backticks` in title' must have backticks stripped."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "<!-- Section 4: Code example with backticks in title -->" in md, (
        f"Backtick strip failed; markers in output:\n"
        + "\n".join(line for line in md.splitlines() if "Section " in line)
    )


def test_html_strip_noise_removes_nav_and_footer(tmp_path):
    """With --strip-html-noise, the <nav> and <footer> content should not appear in output."""
    out, _ = _run_convert(tmp_path, "--strip-html-noise")
    md = out.read_text(encoding="utf-8")
    assert "Home" not in md, "nav was not stripped"
    assert "Site footer" not in md, "footer was not stripped"


def test_html_no_page_markers_suppresses_sections(tmp_path):
    """--no-page-markers must produce output with zero <!-- Section markers,
    even though the headings themselves are still rendered as markdown."""
    out, _ = _run_convert(tmp_path, "--no-page-markers")
    md = out.read_text(encoding="utf-8")
    assert "<!-- Section " not in md, (
        f"--no-page-markers should suppress section markers; output had:\n{md[:300]}"
    )
    # The heading itself is still rendered (we suppress the marker, not the
    # heading line).
    assert "Article Title" in md, "Heading text should still appear in markdown"


def test_html_strip_noise_without_bs4_returns_actionable_error(tmp_path):
    """If bs4 isn't installed, --strip-html-noise must fail with the actionable
    'pip install beautifulsoup4' error rather than crashing or silently doing
    nothing. We simulate the missing dependency by patching sys.modules."""
    import sys as _sys
    from materials.core.base import ConversionOptions
    from materials.formats.html import HTMLConverter

    fixture = _isolated_fixture(tmp_path)
    out = tmp_path / "out.md"

    saved_bs4 = _sys.modules.pop("bs4", None)
    _sys.modules["bs4"] = None  # makes `from bs4 import ...` raise ImportError
    try:
        converter = HTMLConverter()
        opts = ConversionOptions(
            output_path=str(out),
            page_markers=True,
            strip_html_noise=True,
        )
        result = converter.convert(str(fixture), opts)
    finally:
        if saved_bs4 is not None:
            _sys.modules["bs4"] = saved_bs4
        else:
            _sys.modules.pop("bs4", None)

    assert result.status == "error", f"Expected error status, got {result.status}"
    assert "beautifulsoup4" in (result.error or ""), (
        f"Error should mention beautifulsoup4: {result.error!r}"
    )
    assert "pip install" in (result.error or ""), (
        f"Error should give an install hint: {result.error!r}"
    )
