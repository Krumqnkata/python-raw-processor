"""Filesystem discovery and sequential background jobs with UI-neutral events."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import os
import threading
import time

from raw_engine import ProcessingCancelled, ProcessingParams, RAW_EXTENSIONS, RawEngine, check_cancel

Emit = Callable[[str, dict], None]


@dataclass(frozen=True)
class BatchSummary:
    total: int
    completed: int
    succeeded: int
    failed: int
    cancelled: bool
    seconds: float


def unique_inputs(paths: Iterable[str | Path]) -> list[Path]:
    """Preserve selection order while removing aliases and duplicates."""
    seen: set[str] = set()
    result: list[Path] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        key = os.path.normcase(str(path))
        if path.suffix.lower() in RAW_EXTENSIONS and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def discover_raws(folder: str | Path, recursive: bool = False,
                  cancel: threading.Event | None = None) -> tuple[list[Path], list[str]]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    files: list[Path] = []
    errors: list[str] = []
    # os.walk doesn't follow directory symlinks; avoids recursion cycles.
    for directory, subdirs, names in os.walk(root, onerror=lambda error: errors.append(str(error))):
        check_cancel(cancel)
        if not recursive:
            subdirs.clear()
        for name in names:
            check_cancel(cancel)
            path = Path(directory) / name
            if path.suffix.lower() in RAW_EXTENSIONS:
                files.append(path)
    return unique_inputs(sorted(files, key=lambda p: str(p).casefold())), errors


def plan_outputs(files: list[Path], directory: str | Path, output_format: str,
                 overwrite: bool = False) -> list[tuple[Path, Path]]:
    """Reserve distinct stems even when different folders/extensions share a name."""
    if output_format not in {"jpg", "png"}:
        raise ValueError("Неподдържан изходен формат.")
    root = Path(directory).expanduser().resolve()
    used: set[str] = set()
    planned = []
    for source in files:
        candidate = root / f"{source.stem}.{output_format}"
        number = 1
        # Casefold also protects exports moved to case-insensitive filesystems.
        while str(candidate).casefold() in used or (candidate.exists() and not overwrite):
            candidate = root / f"{source.stem}_{number:03d}.{output_format}"
            number += 1
        used.add(str(candidate).casefold())
        planned.append((source, candidate))
    return planned


def run_batch(engine: RawEngine, files: list[Path], directory: Path,
              params: ProcessingParams, overwrite: bool,
              cancel: threading.Event, emit: Emit) -> BatchSummary:
    """One photo at a time limits RAM; all events can be forwarded to a Queue."""
    started = time.perf_counter()
    completed = succeeded = failed = 0
    try:
        check_cancel(cancel)
        jobs = plan_outputs(files, directory, params.output_format, overwrite)
        directory.mkdir(parents=True, exist_ok=True)
        for source, destination in jobs:
            check_cancel(cancel)
            emit("file_started", {"name": source.name, "index": completed + 1, "total": len(jobs)})
            try:
                result = engine.process_file(
                    source, destination, params, overwrite=overwrite, cancel=cancel,
                    on_stage=lambda message: emit("stage", {"message": message}),
                    include_preview=True,
                )
            except ProcessingCancelled:
                raise
            except Exception as error:
                # A broken RAW should not prevent the remaining photographs from exporting.
                failed += 1
                emit("file_error", {"name": source.name, "error": str(error) or type(error).__name__})
            else:
                succeeded += 1
                emit("file_saved", {"result": result, "name": source.name})
            completed += 1
            emit("progress", {"completed": completed, "total": len(jobs),
                              "succeeded": succeeded, "failed": failed})
    except ProcessingCancelled:
        pass
    except Exception as error:
        emit("fatal_error", {"error": str(error) or type(error).__name__})
    summary = BatchSummary(len(files), completed, succeeded, failed,
                           cancel.is_set(), time.perf_counter() - started)
    emit("batch_done", {"summary": summary})
    return summary
