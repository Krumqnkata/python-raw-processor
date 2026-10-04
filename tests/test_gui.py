"""Actual Tk smoke/concurrency checks; use xvfb-run on Linux CI."""
import os
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='Requires a graphical display')


def pump(app, condition, timeout=15):
    deadline = time.monotonic() + timeout
    while not condition():
        app.update()
        if time.monotonic() > deadline:
            raise AssertionError('GUI worker did not finish in time')
        time.sleep(.01)
    app.update()


@pytest.fixture
def app():
    import customtkinter as ctk
    from app_gui import AppGUI
    ctk.set_appearance_mode('Dark')
    app = AppGUI()
    errors = []
    app.report_callback_exception = lambda *error: errors.append(error)
    yield app
    app._on_close()
    if app._closing and app._worker and app._worker.is_alive():
        app._worker.join(timeout=10)
    try:
        if app.winfo_exists():
            app.update()
    except Exception:
        pass
    assert not errors, errors


def test_gui_real_preview_export_and_themes(app, dng_path, tmp_path):
    import customtkinter as ctk
    from PIL import Image
    app._add_files([dng_path])
    pump(app, lambda: app._preview is not None and app._busy is None)
    assert app._preview.before.size == app._preview.after.size
    for mode in ['Преди', 'След', 'Сравнение']:
        app.compare_var.set(mode)
        app._render_preview()
        app.update()
    for theme in ['Светла тема', 'Тъмна тема']:
        app._change_theme(theme)
        app.update()
    assert ctk.get_appearance_mode() == 'Dark'
    app.format_var.set('PNG')
    app._update_format()
    assert app.quality_slider.cget('state') == 'disabled'
    assert app.compression_slider.cget('state') == 'normal'
    app.output_var.set(str(tmp_path / 'export'))
    app._start_batch()
    assert app.wb_menu.cget('state') == 'disabled'
    pump(app, lambda: app._busy is None)
    assert (tmp_path / 'export' / 'synthetic.png').is_file()
    assert app.progress.get() == 1
    assert app._batch_preview is not None
    assert app.compare_var.get() == 'След'
    assert 'Завършено' in app.status_label.cget('text')


def test_worker_never_blocks_tk_or_updates_it_from_background(app, dng_path, tmp_path):
    from raw_engine import RawEngine
    main_thread = threading.get_ident()
    worker_threads = []
    handler_threads = []
    release = threading.Event()
    entered = threading.Event()

    class HeldEngine(RawEngine):
        def process_file(self, *args, **kwargs):
            worker_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test barrier timed out')
            return super().process_file(*args, **kwargs)

    app.engine = HeldEngine()
    original_handle = app._handle_event
    def record_handle(*args):
        handler_threads.append(threading.get_ident())
        original_handle(*args)
    app._handle_event = record_handle
    app.files = [dng_path]  # Export-only: no initial automatic preview.
    app.output_var.set(str(tmp_path / 'out'))
    app._start_batch()
    pump(app, entered.is_set)
    heartbeat = []
    app.after(10, lambda: heartbeat.append(True))
    try:
        pump(app, lambda: bool(heartbeat), timeout=2)
        assert app._busy == 'batch'  # UI responded while the worker was held.
    finally:
        release.set()
    pump(app, lambda: app._busy is None)
    assert worker_threads and all(t != main_thread for t in worker_threads)
    assert handler_threads and all(t == main_thread for t in handler_threads)


def test_preview_changes_coalesce_without_stale_results(app, dng_path):
    import numpy as np
    from raw_engine import ProcessingParams
    app._add_files([dng_path])
    app.exposure_slider.set(1)
    app._schedule_preview()
    app.exposure_slider.set(-1)
    app._schedule_preview()
    pump(app, lambda: app._preview is not None and app._busy is None and not app._pending_preview)
    assert app._params().exposure_ev == -1
    assert app._preview.after.size == app._preview.before.size
    expected = app.engine.preview(dng_path, ProcessingParams(exposure_ev=-1)).after
    np.testing.assert_allclose(np.asarray(app._preview.after).astype(float),
                               np.asarray(expected).astype(float), atol=1)
