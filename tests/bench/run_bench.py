#!/usr/bin/env python3
"""Benchmark serial vs parallel batch PDF conversion.

Usage:
    ./venv/bin/python tests/bench/run_bench.py <pdf_directory>

Runs the same batch conversion serially (workers=1) and in parallel
(workers=2, 4, 8 — capped at min(8, file_count // 2)). Reports wall-clock
time and per-file mean for each. The corpus is the user's choice; for
casebook-scale files, parallelism wins past ~8 files. For small batches,
serial wins because per-worker DocumentConverter warmup dominates.

Pastes its output to stdout — copy into a stage-5 PR comment or commit
message to record the threshold for the user's actual hardware/corpus.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path


def _bench_one(pdf_dir: str, workers: int) -> tuple[float, int, int]:
    """Returns (wall_clock_seconds, success_count, error_count)."""
    import sys as _sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))

    from materials.formats.pdf import PDFConverter

    converter = PDFConverter()
    with tempfile.TemporaryDirectory() as out_dir:
        start = time.time()
        result = converter.convert_directory(
            pdf_dir,
            output_dir=out_dir,
            recursive=False,
            save_report=False,
            page_markers=True,
            workers=workers,
        )
        elapsed = time.time() - start
    return elapsed, result.get("success_count", 0), result.get("error_count", 0)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: run_bench.py <pdf_directory>", file=sys.stderr)
        return 2
    pdf_dir = argv[0]
    if not Path(pdf_dir).is_dir():
        print(f"Not a directory: {pdf_dir}", file=sys.stderr)
        return 1

    file_count = len([p for p in Path(pdf_dir).glob("*.pdf")])
    if file_count == 0:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        return 1

    print(f"=== materials-md batch benchmark ===")
    print(f"Corpus: {pdf_dir} ({file_count} PDFs)")
    print(f"")

    # Warmup pass: load the Docling model into the OS page cache so subsequent
    # measurements aren't biased toward "first run is slow because of cold load."
    # Discard the result.
    print(f"  warmup (workers=1, discarded): ", end="", flush=True)
    warmup_elapsed, _ok, _err = _bench_one(pdf_dir, 1)
    print(f"{warmup_elapsed:.1f}s")
    print()

    worker_counts = [1, 2, 4, 8]
    worker_counts = [w for w in worker_counts if w == 1 or w * 2 <= file_count]
    if not worker_counts:
        worker_counts = [1]

    results = []
    for workers in worker_counts:
        elapsed, succeeded, failed = _bench_one(pdf_dir, workers)
        per_file = elapsed / max(succeeded, 1)
        results.append((workers, elapsed, per_file, succeeded, failed))
        print(f"  workers={workers}: {elapsed:6.1f}s  ({per_file:5.2f}s/file)  "
              f"{succeeded} ok, {failed} err")

    print()
    base = results[0][1]
    for workers, elapsed, _per, _ok, _err in results[1:]:
        speedup = base / elapsed if elapsed > 0 else 0
        print(f"  workers={workers} speedup vs serial: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
