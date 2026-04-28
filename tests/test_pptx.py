"""PPTX conversion tests."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample_with_notes.pptx"


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


def test_pptx_converts_without_error(tmp_path):
    """Smoke: convert.py handles .pptx files."""
    out, result = _run_convert(tmp_path)
    assert result.returncode == 0, (
        f"convert.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    md = out.read_text(encoding="utf-8")
    assert len(md) > 0


def test_pptx_emits_slide_markers(tmp_path):
    """Each of the 3 slides gets a numbered <!-- Slide K --> marker."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    slide_lines = [line for line in md.splitlines() if line.startswith("<!-- Slide ")]
    assert len(slide_lines) == 3, f"Expected 3 slide markers, got {len(slide_lines)}: {slide_lines}"
    for i, line in enumerate(slide_lines, start=1):
        assert line.startswith(f"<!-- Slide {i}"), f"Marker {i} wrong: {line!r}"


def test_pptx_speaker_notes_present_for_notes_slides(tmp_path):
    """Slides 1 and 2 have notes; their text must appear in output."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "Warner-Jenkinson" in md, (
        f"Speaker note for slide 1 should appear; got:\n{md}"
    )
    assert "Festo cabined the doctrine" in md, (
        f"Speaker note for slide 2 should appear; got:\n{md}"
    )


def test_pptx_speaker_notes_marker_only_for_notes_slides(tmp_path):
    """The <!-- Speaker notes --> marker appears only for slides that have
    notes. Fixture has 2 such slides."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    notes_markers = [line for line in md.splitlines() if line.startswith("<!-- Speaker notes")]
    assert len(notes_markers) == 2, (
        f"Expected 2 Speaker notes markers (slides 1, 2), got {len(notes_markers)}: {notes_markers}"
    )


def test_pptx_slide_body_content_present_by_default(tmp_path):
    """Default mode includes slide bullet content in addition to notes."""
    out, _ = _run_convert(tmp_path)
    md = out.read_text(encoding="utf-8")
    assert "Literal infringement" in md, "Slide 1 bullet should appear"
    assert "Festo Corp." in md, "Slide 2 bullet should appear"


def test_pptx_notes_only_skips_bullet_content(tmp_path):
    """--notes-only emits a clean transcript: slide numbers + notes text,
    dropping bullet content. Slides without notes are skipped."""
    out, _ = _run_convert(tmp_path, "--notes-only")
    md = out.read_text(encoding="utf-8")
    # Notes from slides 1 and 2 should appear
    assert "Warner-Jenkinson" in md
    assert "Festo cabined the doctrine" in md
    # Bullet content should NOT appear
    assert "Literal infringement" not in md, (
        f"--notes-only should drop bullet content; got bullet text:\n{md}"
    )
    # Slide 3 has no notes; bullets and title for slide 3 are absent.
    assert "Three takeaways" not in md
    assert "Summary and Discussion" not in md, (
        "--notes-only should skip slide 3 (no notes) entirely"
    )


def test_pptx_no_page_markers_suppresses_slide_markers(tmp_path):
    """--no-page-markers should suppress slide markers (and speaker-notes
    markers, since both are 'position markers')."""
    out, _ = _run_convert(tmp_path, "--no-page-markers")
    md = out.read_text(encoding="utf-8")
    assert "<!-- Slide " not in md
    assert "<!-- Speaker notes" not in md
