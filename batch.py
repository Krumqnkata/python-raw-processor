"""Filesystem discovery and sequential background jobs with UI-neutral events."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import os
import hashlib
import json
import re
from datetime import datetime
from dataclasses import asdict
from studio import atomic_json, read_json, params_from_dict
from metadata import read_metadata
import threading
import time

from raw_engine import ProcessingCancelled, ProcessingParams, RAW_EXTENSIONS, RawEngine, check_cancel, validate_template

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
                 overwrite: bool = False, template: str = '{stem}') -> list[tuple[Path, Path]]:
    """Reserve distinct stems even when different folders/extensions share a name."""
    if output_format not in {"jpg", "png"}:
        raise ValueError("Неподдържан изходен формат.")
    root = Path(directory).expanduser().resolve()
    validate_template(template)
    used: set[str] = set()
    planned = []
    for index, source in enumerate(files,1):
        date, camera = '', ''
        if '{date}' in template or '{camera}' in template:
            try: metadata = read_metadata(source)
            except Exception: metadata = {}
            date = str(metadata.get('EXIF DateTimeOriginal',''))[:10].replace(':','')
            if not date:
                date = datetime.fromtimestamp(source.stat().st_mtime).strftime('%Y%m%d') if source.exists() else 'unknown'
            camera = re.sub(r'[\\/<>:"|?*\x00-\x1f]', '_', str(metadata.get('Image Model','camera')))
        stem = template.format(stem=source.stem,index=index,date=date,camera=camera).strip(' .')[:180] or 'photo'
        candidate = root / f"{stem}.{output_format}"
        number = 1
        # Casefold also protects exports moved to case-insensitive filesystems.
        while str(candidate).casefold() in used or (candidate.exists() and not overwrite):
            candidate = root / f"{stem}_{number:03d}.{output_format}"
            number += 1
        used.add(str(candidate).casefold())
        planned.append((source, candidate))
    return planned


def wait_pause(pause, cancel):
    while pause is not None and pause.is_set():
        check_cancel(cancel)
        cancel.wait(.1)
    check_cancel(cancel)


def file_signature(path):
    try:
        stat = Path(path).stat()
        return [stat.st_size,stat.st_mtime_ns]
    except OSError:
        return None


def run_batch(engine: RawEngine, files: list[Path], directory: Path,
              params: ProcessingParams, overwrite: bool,
              cancel: threading.Event, emit: Emit, *, per_file: dict | None = None,
              pause: threading.Event | None = None, resume: bool = False) -> BatchSummary:
    """Persist successful jobs; pause cooperatively and resume matching snapshots."""
    started = time.perf_counter()
    completed = succeeded = failed = 0
    journal_path = directory/'raw-studio-batch.json'
    try:
        check_cancel(cancel)
        settings = [per_file.get(str(source),params) if per_file else params for source in files]
        snapshot = [{'source':str(p),'signature':file_signature(p),'params':asdict(v)} for p,v in zip(files,settings)]
        fingerprint = hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
        journal = None
        if resume and journal_path.is_file():
            old = read_json(journal_path)
            if old.get('version')==1 and old.get('fingerprint')==fingerprint:
                journal = old
            else:
                emit('stage',{'message':'Настройките/файловете са променени; започвам нова серия.'})
        if journal is None:
            # Export format/name policy is common, editing is per photo.
            jobs = plan_outputs(files,directory,params.output_format,overwrite,params.filename_template)
            journal = {'version':1,'fingerprint':fingerprint,'overwrite':overwrite,
                       'jobs':[dict(row,destination=str(dest),status='pending') for row,(_,dest) in zip(snapshot,jobs)]}
        directory.mkdir(parents=True,exist_ok=True)
        atomic_json(journal_path,journal)
        for row, setting in zip(journal['jobs'],settings):
            wait_pause(pause,cancel)
            source,destination = Path(row['source']),Path(row['destination'])
            if destination.parent.resolve()!=directory.resolve() or destination.suffix.lower() not in {'.jpg','.png'}:
                raise ValueError('Невалиден изходен път в дневника на серията.')
            emit('file_started',{'name':source.name,'index':completed+1,'total':len(files)})
            if row.get('status')=='done' and row.get('output_signature') == file_signature(destination) and destination.is_file():
                succeeded += 1
                emit('stage',{'message':'Вече обработено: '+source.name})
            else:
                if destination.exists() and not overwrite:
                    number = 1
                    original = destination
                    while destination.exists():
                        destination = original.with_name(f'{original.stem}_{number:03d}{original.suffix}')
                        number += 1
                    row['destination'] = str(destination)
                    atomic_json(journal_path,journal)
                def stage(message):
                    wait_pause(pause,cancel)
                    emit('stage',{'message':message})
                try:
                    result = engine.process_file(source,destination,setting,overwrite=overwrite,cancel=cancel,
                                                 on_stage=stage,include_preview=True)
                except ProcessingCancelled:
                    raise
                except Exception as error:
                    failed += 1
                    row['status'] = 'failed'
                    row['error'] = str(error) or type(error).__name__
                    emit('file_error',{'name':source.name,'error':row['error']})
                else:
                    succeeded += 1
                    row['status'] = 'done'
                    row['output_signature'] = file_signature(destination)
                    emit('file_saved',{'result':result,'name':source.name,'source':source})
                atomic_json(journal_path,journal)
            completed += 1
            emit('progress',{'completed':completed,'total':len(files),'succeeded':succeeded,'failed':failed})
    except ProcessingCancelled:
        pass
    except Exception as error:
        emit('fatal_error',{'error':str(error) or type(error).__name__})
    summary = BatchSummary(len(files),completed,succeeded,failed,cancel.is_set(),time.perf_counter()-started)
    emit('batch_done',{'summary':summary})
    return summary


class FolderWatcher:
    """Report new/changed RAW files only after two unchanged polling intervals."""
    def __init__(self,folder,recursive=False,interval=2.):
        self.folder = Path(folder).resolve()
        self.recursive = recursive
        self.interval = interval
        self.seen = {}
        self.pending = {}
        self.initialized = False

    def poll(self):
        paths,_ = discover_raws(self.folder,self.recursive)
        signatures = {str(p):file_signature(p) for p in paths}
        if not self.initialized:
            self.seen = signatures
            self.initialized = True
            return []
        ready = []
        for path, signature in signatures.items():
            if signature is None or signature[0] == 0 or self.seen.get(path)==signature:
                continue
            old,count = self.pending.get(path,(None,0))
            count = count+1 if old==signature else 0
            self.pending[path] = (signature,count)
            if count >= 2:
                ready.append(Path(path))
                self.seen[path] = signature
                self.pending.pop(path,None)
        self.pending = {p:v for p,v in self.pending.items() if p in signatures}
        self.seen = {p:v for p,v in self.seen.items() if p in signatures}
        return ready

    def run(self,cancel,emit):
        while not cancel.is_set():
            try:
                ready = self.poll()
                if ready: emit('watch_files',{'files':ready})
            except OSError as error:
                emit('watch_error',{'error':str(error)})
            cancel.wait(self.interval)
