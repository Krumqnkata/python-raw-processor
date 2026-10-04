import threading
from pathlib import Path

from batch import discover_raws, plan_outputs, run_batch, unique_inputs
from raw_engine import ProcessingParams, RawEngine


def test_folder_discovery_case_and_recursion(tmp_path):
    (tmp_path / 'A.CR2').touch()
    (tmp_path / 'b.nef').touch()
    (tmp_path / 'not-raw.jpg').touch()
    sub = tmp_path / 'nested'
    sub.mkdir()
    (sub / 'c.ARW').touch()
    flat, errors = discover_raws(tmp_path)
    recursive, _ = discover_raws(tmp_path, recursive=True)
    assert len(flat) == 2 and len(recursive) == 3 and errors == []


def test_duplicates_and_output_collisions(tmp_path):
    a = tmp_path / 'a' / 'photo.CR2'
    b = tmp_path / 'b' / 'photo.NEF'
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'photo.jpg').write_bytes(b'existing')
    assert unique_inputs([a, a, b]) == [a, b]
    jobs = plan_outputs([a, b], out, 'jpg')
    assert [p.name for _, p in jobs] == ['photo_001.jpg', 'photo_002.jpg']
    jobs = plan_outputs([a, b], out, 'jpg', overwrite=True)
    assert [p.name for _, p in jobs] == ['photo.jpg', 'photo_001.jpg']


def test_corrupt_raw_does_not_abort_batch(dng_path, tmp_path):
    broken = tmp_path / 'broken.CR2'
    broken.write_text('This is not a RAW photograph.')
    events = []
    summary = run_batch(RawEngine(), [broken, dng_path], tmp_path / 'out',
                        ProcessingParams(), False, threading.Event(),
                        lambda kind, data: events.append((kind, data)))
    assert summary.completed == 2 and summary.failed == 1 and summary.succeeded == 1
    assert (tmp_path / 'out' / 'synthetic.jpg').is_file()
    progress = [data['completed'] for kind, data in events if kind == 'progress']
    assert progress == [1, 2]
    assert events[-1][0] == 'batch_done'


def test_cancel_between_files(dng_path, tmp_path):
    event = threading.Event()
    events = []
    def emit(kind, data):
        events.append(kind)
        if kind == 'file_saved':
            event.set()
    summary = run_batch(RawEngine(), [dng_path, dng_path], tmp_path / 'out',
                        ProcessingParams(), False, event, emit)
    assert summary.cancelled and summary.completed == 1 and summary.succeeded == 1
    assert len(list((tmp_path / 'out').glob('*.jpg'))) == 1


def test_pre_cancelled_batch_does_no_work(dng_path, tmp_path):
    event = threading.Event()
    event.set()
    summary = run_batch(RawEngine(), [dng_path], tmp_path / 'out',
                        ProcessingParams(), False, event, lambda *_: None)
    assert summary.cancelled and summary.completed == 0
    assert not (tmp_path / 'out').exists()
