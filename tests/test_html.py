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


def test_html_default_keeps_nav_content(tmp_path):
    """Without --strip-html-noise, nav/footer content goes through Docling's default handling.
    This test documents behavior — Docling may or may not include nav text by default."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    # Just verify the test runs and produces non-empty output. Default behavior
    # depends on Docling's HTML pipeline; we don't enforce nav presence/absence here.
    assert len(md) > 100
