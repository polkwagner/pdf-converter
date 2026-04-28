"""DOCX conversion tests."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample_with_comments.docx"


def _isolated_fixture(tmpdir: Path) -> Path:
    dest = tmpdir / FIXTURE.name
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


def test_docx_converts_without_error(tmp_path):
    """Smoke: convert.py handles .docx files."""
    out, result = _run_convert(tmp_path)
    assert result.returncode == 0, (
        f"convert.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    md = out.read_text(encoding="utf-8")
    assert len(md) > 0


def test_docx_emits_section_markers(tmp_path):
    """1 H1 + 1 H2 in the source DOCX → 2 numbered section markers in output.
    (Docling shifts heading levels: H1 source becomes ## in markdown,
    H2 source becomes ###. The DOCX converter detects this and treats
    the two highest-level headings present as section boundaries.)"""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    section_lines = [line for line in md.splitlines() if line.startswith("<!-- Section ")]
    assert len(section_lines) == 2, f"Expected 2 section markers, got {len(section_lines)}: {section_lines}"
    assert section_lines[0].startswith("<!-- Section 1:")
    assert section_lines[1].startswith("<!-- Section 2:")


def test_docx_lean_default_drops_comments(tmp_path):
    """Default mode (lean): comment text 'Consider citing Bilski' must NOT appear."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "Consider citing Bilski" not in md, (
        f"Default mode should strip comments; output contained the comment text:\n{md[:500]}"
    )
    assert "## Reviewer Comments" not in md, (
        "Default mode should not include the Reviewer Comments appendix"
    )


def test_docx_full_includes_reviewer_comments_appendix(tmp_path):
    """--full appends a `## Reviewer Comments` section with the comment text."""
    out, _ = _run_convert(tmp_path, "--full")
    md = out.read_text(encoding="utf-8")
    assert "## Reviewer Comments" in md, (
        f"--full should include Reviewer Comments appendix; output:\n{md[-600:]}"
    )
    assert "Sarah Chen" in md, "Comment author should appear in the appendix"
    assert "Consider citing Bilski" in md, "Comment body should appear in the appendix"


def test_docx_footnotes_always_preserved(tmp_path):
    """Footnotes are body content (legal citations etc.) and survive in lean mode.
    The footnote text must appear in the output, either inline as `[^1]` or in
    a `## Footnotes` section. (Docling's DOCX backend omits footnotes entirely;
    the converter extracts them from `footnotes.xml` and appends a section.)"""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "Bilski v. Kappos" in md, (
        f"Footnote text should appear in default output; got:\n{md[:800]}"
    )


def test_docx_show_revisions_renders_inline_changes(tmp_path):
    """--show-revisions renders ins/del as `[+ added +]` / `[- removed -]`.
    Both spaced (`[+ ADDED-TEXT +]`) and unspaced (`[+ADDED-TEXT+]`) are
    accepted to give the implementation flexibility."""
    out, _ = _run_convert(tmp_path, "--show-revisions")
    md = out.read_text(encoding="utf-8")
    assert "[+ ADDED-TEXT +]" in md or "[+ADDED-TEXT+]" in md or "[+ ADDED-TEXT+]" in md or "[+ADDED-TEXT +]" in md, (
        f"Expected insertion marker [+ ADDED-TEXT +]; got:\n{md}"
    )
    assert "[- REMOVED-TEXT -]" in md or "[-REMOVED-TEXT-]" in md or "[- REMOVED-TEXT-]" in md or "[-REMOVED-TEXT -]" in md, (
        f"Expected deletion marker [- REMOVED-TEXT -]; got:\n{md}"
    )


def test_docx_default_accepts_revisions(tmp_path):
    """Without --show-revisions, the inserted text appears as plain prose
    and the deleted text is gone — i.e., the 'accepted' state."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "ADDED-TEXT" in md, "Insertion should appear in accepted-final output"
    assert "REMOVED-TEXT" not in md, "Deletion should be dropped in accepted-final output"


def test_docx_no_page_markers_suppresses_sections(tmp_path):
    """--no-page-markers should produce zero <!-- Section markers."""
    out, _ = _run_convert(tmp_path, "--no-page-markers")
    md = out.read_text(encoding="utf-8")
    assert "<!-- Section " not in md
