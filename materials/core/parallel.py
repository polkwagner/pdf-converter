"""Parallel batch conversion via ProcessPoolExecutor.

Each worker process runs its own DocumentConverter; per-worker model warmup
is the trade-off cost. For small batches (<= 4-8 files) the serial path
(BaseConverter.convert_directory's loop, or PDFConverter's warmed-model trick)
is faster. For larger batches, parallelism wins.

Tested empirically via tests/bench/run_bench.py — see CLAUDE.md for
benchmark guidance.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from materials.core.base import ConversionOptions, ConversionResult


def _convert_one_in_worker(
    converter_class_path: str,
    input_path: str,
    options_kwargs: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Worker entry point. Imports the converter class lazily so each worker
    pays the model-warmup cost on first task only — subsequent tasks in the
    same worker reuse the cached import.

    Returns (input_path, serialized_result_dict). We serialize the result so
    ConversionResult dataclass instances don't have to traverse the pickle
    boundary if the calling Python version differs (defensive).
    """
    module_name, class_name = converter_class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    converter_class = getattr(module, class_name)

    converter = converter_class()
    options = ConversionOptions(**options_kwargs)
    result = converter.convert(input_path, options)
    return (
        input_path,
        {
            "status": result.status,
            "output_file": result.output_file,
            "statistics": result.statistics,
            "conversion_time": result.conversion_time,
            "error": result.error,
        },
    )


def parallel_convert_files(
    converter_class: type,
    input_paths: List[str],
    options: ConversionOptions,
    workers: int,
    input_dir: Optional[str] = None,
    recursive: bool = False,
) -> Dict[str, Any]:
    """Convert each path in `input_paths` in parallel using `workers` worker
    processes. Returns the same shape as BaseConverter.convert_directory's
    serial path: {success_count, error_count, reports}.

    `converter_class` is a class object (e.g., PDFConverter); we serialize its
    dotted path for the workers to re-import.

    `input_dir` and `recursive`: when both are provided and recursive is True,
    output paths preserve subdirectory structure relative to input_dir,
    matching the serial path's behavior and avoiding filename collisions.
    """
    logger = logging.getLogger("pdf_converter")
    converter_class_path = f"{converter_class.__module__}.{converter_class.__name__}"

    options_kwargs = {
        "output_path": options.output_path,
        "page_markers": options.page_markers,
        "pages": options.pages,
        "ocr": options.ocr,
        "extract_images": options.extract_images,
        "quiet": True,
        "verbose": options.verbose,
        "save_report": options.save_report,
        "log_path": options.log_path,
        "full": options.full,
        "show_revisions": options.show_revisions,
        "keep_images": options.keep_images,
        "notes_only": options.notes_only,
        "strip_html_noise": options.strip_html_noise,
        "workers": 1,
    }

    success_count = 0
    error_count = 0
    reports: List[Dict[str, Any]] = []

    logger.info(f"Parallel batch: {len(input_paths)} files, {workers} workers")

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for path in input_paths:
                per_file_kwargs = dict(options_kwargs)
                if options.output_path:
                    out_root = Path(options.output_path)
                    out_root.mkdir(parents=True, exist_ok=True)
                    src_path = Path(path)
                    if input_dir and recursive:
                        try:
                            rel = src_path.relative_to(input_dir)
                            target = out_root / rel.with_suffix(".md")
                        except ValueError:
                            target = out_root / src_path.with_suffix(".md").name
                    else:
                        target = out_root / src_path.with_suffix(".md").name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    per_file_kwargs["output_path"] = str(target)
                else:
                    per_file_kwargs["output_path"] = None
                futures.append(executor.submit(
                    _convert_one_in_worker,
                    converter_class_path,
                    path,
                    per_file_kwargs,
                ))

            for future in as_completed(futures):
                try:
                    input_path, result_dict = future.result()
                except Exception as exc:
                    error_count += 1
                    reports.append({
                        "input": "<unknown — worker raised>",
                        "status": "error",
                        "error": str(exc),
                    })
                    logger.warning(f"Worker raised: {exc}")
                    continue

                if result_dict["status"] == "success":
                    success_count += 1
                else:
                    error_count += 1
                reports.append({
                    "input": input_path,
                    "status": result_dict["status"],
                    "output_file": result_dict["output_file"],
                    "error": result_dict["error"],
                    "statistics": result_dict["statistics"],
                })

    except BrokenProcessPool as exc:
        unstarted = len(input_paths) - success_count - error_count
        logger.error(
            f"Worker pool died (BrokenProcessPool): {exc}. "
            f"Returning partial results: {success_count} succeeded, "
            f"{error_count} failed before crash, "
            f"{unstarted} unstarted."
        )
        error_count += unstarted

    logger.info(
        f"Parallel batch complete: {success_count} succeeded, {error_count} failed"
    )
    return {
        "success_count": success_count,
        "error_count": error_count,
        "reports": reports,
    }
