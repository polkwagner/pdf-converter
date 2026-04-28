"""PDF conversion tests, including the byte-identical migration test that
guards stage 1's "zero behavior change" promise."""
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample.pdf"


def _run_legacy(tmpdir: Path) -> Path:
    """Run the frozen legacy code path against the fixture, return md path."""
    from tests.fixtures import legacy_pdf_to_markdown as legacy

    out = tmpdir / "legacy.md"
    logger = logging.getLogger("legacy_test")
    logger.addHandler(logging.NullHandler())
    legacy.convert_pdf_to_markdown(
        str(FIXTURE),
        output_path=str(out),
        page_markers=True,
        report=False,
        quiet=True,
        logger=logger,
    )
    return out


def _run_new(tmpdir: Path) -> Path:
    """Run the new convert.py CLI against the fixture, return md path."""
    out = tmpdir / "new.md"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "convert.py"),
            str(FIXTURE),
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
    legacy_md = _run_legacy(tmp_path)
    new_md = _run_new(tmp_path)

    legacy_bytes = legacy_md.read_bytes()
    new_bytes = new_md.read_bytes()

    if legacy_bytes != new_bytes:
        # Surface a useful diff snippet
        legacy_lines = legacy_bytes.decode("utf-8", errors="replace").splitlines()
        new_lines = new_bytes.decode("utf-8", errors="replace").splitlines()
        import difflib
        diff = "\n".join(difflib.unified_diff(legacy_lines, new_lines, lineterm="", n=2)[:60])
        pytest.fail(f"Output diverged:\n{diff}")
