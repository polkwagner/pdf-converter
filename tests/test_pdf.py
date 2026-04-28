"""PDF conversion tests — guards against regressions in the byte-identical
output that Stage 1's refactor was specified to preserve.

Pinned-golden pattern: the canonical reference is `tests/fixtures/sample.golden.md`,
generated once on Docling 2.65.0 at Stage 1 completion and committed. Both the
new `convert.py` path and the frozen legacy snapshot must produce output that
matches the golden. When Docling upgrades and produces different output, the
golden is regenerated as a deliberate, reviewable acceptance — turning a
silent identity comparison into an explicit regression guard.

Regenerate the golden with:
    ./venv/bin/python convert.py tests/fixtures/sample.pdf -o tests/fixtures/sample.golden.md
"""
import difflib
import hashlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample.pdf"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "sample.golden.md"


def _isolated_fixture(tmpdir: Path) -> Path:
    """Copy the source fixture into tmpdir so PyMuPDF/Docling's incidental
    writes during reading don't mutate the committed fixture. Preserves the
    original filename so document-title and "Converted from:" lines in the
    output match the pinned golden."""
    dest = tmpdir / FIXTURE.name
    shutil.copy(FIXTURE, dest)
    return dest


def _convert_via_legacy(tmpdir: Path) -> Path:
    """Run the frozen legacy code path against an isolated fixture copy."""
    from tests.fixtures import legacy_pdf_to_markdown as legacy

    fixture = _isolated_fixture(tmpdir)
    out = tmpdir / "legacy.md"
    logger = logging.getLogger("legacy_test")
    logger.addHandler(logging.NullHandler())
    legacy.convert_pdf_to_markdown(
        str(fixture),
        output_path=str(out),
        page_markers=True,
        report=False,
        quiet=True,
        logger=logger,
    )
    return out


def _convert_via_new(tmpdir: Path) -> Path:
    """Run the new convert.py CLI against an isolated fixture copy."""
    fixture = _isolated_fixture(tmpdir)
    out = tmpdir / "new.md"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "convert.py"), str(fixture), "-o", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"convert.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    return out


def _diff_against_golden(produced_path: Path, label: str) -> None:
    """Assert produced output matches the golden, with a useful diff on failure."""
    produced = produced_path.read_bytes()
    golden = GOLDEN.read_bytes()
    if produced == golden:
        return
    produced_lines = produced.decode("utf-8", errors="replace").splitlines()
    golden_lines = golden.decode("utf-8", errors="replace").splitlines()
    diff = "\n".join(difflib.unified_diff(
        golden_lines, produced_lines,
        fromfile="golden", tofile=label, lineterm="", n=2,
    ))
    pytest.fail(
        f"{label} diverged from golden. If this is intentional (Docling upgrade or "
        f"deliberate behavior change), regenerate with:\n"
        f"  ./venv/bin/python convert.py tests/fixtures/sample.pdf "
        f"-o tests/fixtures/sample.golden.md\n\n"
        f"Diff (first 60 lines):\n{diff[:6000]}"
    )


def test_new_matches_golden(tmp_path):
    """convert.py must produce output byte-equal to the pinned golden file.
    This is the primary regression guard for the Stage 1 refactor: any change
    in PDF→markdown output (whether from a Docling upgrade, a refactor, or a
    bug) trips this test and forces an explicit review of whether to accept
    the new output by regenerating the golden."""
    produced = _convert_via_new(tmp_path)
    _diff_against_golden(produced, "convert.py")


def test_legacy_matches_golden(tmp_path):
    """The frozen legacy snapshot must also match the golden. This is a sanity
    check against accidental drift in the snapshot itself: if someone edits
    `tests/fixtures/legacy_pdf_to_markdown.py`, this test catches it. Together
    with test_new_matches_golden, both code paths are pinned to the same
    reference output, so the migration claim of byte-identity is durable
    even when Docling versions change."""
    produced = _convert_via_legacy(tmp_path)
    _diff_against_golden(produced, "legacy snapshot")


def test_fixture_unchanged_after_conversion(tmp_path):
    """Regression guard: the source fixture must not mutate during a
    conversion. Catches the Stage 1 flake where PyMuPDF wrote back to
    incremental-save PDFs during read paths."""
    before = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    _convert_via_new(tmp_path)
    after = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert before == after, "Source fixture was mutated during a conversion"
