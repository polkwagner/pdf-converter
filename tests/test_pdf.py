"""PDF conversion tests, including the byte-identical migration test that
guards stage 1's "zero behavior change" promise."""
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample.pdf"


def _isolated_fixture(tmpdir: Path) -> Path:
    """Copy the source fixture into tmpdir so PyMuPDF/Docling's incidental
    writes during reading don't mutate the committed fixture."""
    dest = tmpdir / "fixture.pdf"
    shutil.copy(FIXTURE, dest)
    return dest


def _run_legacy(tmpdir: Path) -> Path:
    """Run the frozen legacy code path against an isolated fixture copy, return md path."""
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


def _run_new(tmpdir: Path) -> Path:
    """Run the new convert.py CLI against an isolated fixture copy, return md path."""
    fixture = _isolated_fixture(tmpdir)
    out = tmpdir / "new.md"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "convert.py"),
            str(fixture),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"convert.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    return out


def test_migration_byte_identical(tmp_path):
    """Stage 1's exit criterion: convert.py output must be byte-identical
    to the frozen legacy pdf_to_markdown.py output for the sample fixture."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    legacy_md = _run_legacy(legacy_dir)
    new_md = _run_new(new_dir)

    legacy_bytes = legacy_md.read_bytes()
    new_bytes = new_md.read_bytes()

    if legacy_bytes != new_bytes:
        legacy_lines = legacy_bytes.decode("utf-8", errors="replace").splitlines()
        new_lines = new_bytes.decode("utf-8", errors="replace").splitlines()
        import difflib
        diff = "\n".join(difflib.unified_diff(legacy_lines, new_lines, lineterm="", n=2)[:60])
        pytest.fail(f"Output diverged:\n{diff}")


def test_fixture_unchanged_after_test(tmp_path):
    """Regression guard: running the migration test must not mutate the source fixture.
    This catches a stage-1 flake where PyMuPDF wrote back to the fixture during read."""
    import hashlib

    before = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    test_migration_byte_identical(tmp_path)
    after = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert before == after, "Source fixture was mutated during the test run"
