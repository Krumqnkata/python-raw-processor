"""Actual Tk smoke/concurrency checks; use xvfb-run on Linux CI."""
import os
import gc
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != 'win32' and not os.environ.get('DISPLAY'),
    reason='Requires a graphical display',
)


def pump(app, condition, timeout=15):
    deadline = time.monotonic() + timeout
    while not condition():
        app.update()
        if time.monotonic() > deadline:
            raise AssertionError('GUI worker did not finish in time')
        time.sleep(.01)
    app.update()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('RAW_STUDIO_CONFIG_DIR',str(tmp_path/'preferences'))
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


@pytest.mark.parametrize('clear_action', ['_selection_changed', '_clear_files'])
def test_cleared_preview_can_display_another_image(app, clear_action):
    from PIL import Image
    before = app.before_image_label
    after = app.after_image_label
    app._display_image(before, Image.new('RGB', (120, 80), 'red'))
    app._display_image(after, Image.new('RGB', (120, 80), 'green'))
    app.update()
    getattr(app, clear_action)()
    gc.collect()
    # Reset must detach the native Tk image too, before its Python owner dies.
    assert str(before._label.cget('image')) == ''
    assert str(after._label.cget('image')) == ''
    app._display_image(before, Image.new('RGB', (120, 80), 'blue'))
    app._display_image(after, Image.new('RGB', (120, 80), 'yellow'))
    app.update()
    for label in [before, after]:
        assert str(label._label.cget('image')) in app.tk.call('image', 'names')


def test_individual_edits_presets_history_and_clipboard(app,dng_path,tmp_path):
    from pathlib import Path
    second = tmp_path/'second.DNG'
    second.write_bytes(dng_path.read_bytes())
    app._add_files([dng_path,second])
    pump(app,lambda:app._preview is not None and app._busy is None)
    app.exposure_slider.set(.8)
    app._schedule_preview()
    app._select_photo(second)
    pump(app,lambda:app._preview is not None and app._busy is None)
    assert app._params().exposure_ev==0
    app._apply_builtin('Черно-бяло')
    pump(app,lambda:app._busy is None and not app._pending_preview)
    assert app._params().monochrome
    app._undo()
    assert not app._params().monochrome
    app._redo()
    assert app._params().monochrome
    app._select_photo(dng_path)
    pump(app,lambda:app._preview is not None and app._busy is None)
    assert app._params().exposure_ev==.8
    assert not app._params().monochrome
    app._copy_settings()
    app._paste_settings()
    assert app.session.photos[str(second)].params.exposure_ev==.8
    assert not app.session.photos[str(second)].params.monochrome
    pump(app,lambda:app._busy is None and not app._pending_preview)
    app.max_edge_var.set('173')
    app.template_var.set('event_{index:04d}_{stem}')
    app.output_var.set(str(tmp_path/'out'))
    app._start_batch()
    pump(app,lambda:app._busy is None)
    outputs = sorted((tmp_path/'out').glob('*.jpg'))
    assert len(outputs)==2
    from PIL import Image
    for path in outputs:
        with Image.open(path) as image: assert max(image.size)==173
    assert outputs[0].name.startswith('event_0001_')


def test_canvas_tools_zoom_clipping_and_thumbnails(app,dng_path):
    from types import SimpleNamespace
    app.geometry('1100x720')  # Exercise the minimum supported window too.
    app._add_files([dng_path])
    pump(app,lambda:app._preview is not None and app._busy is None)
    app.compare_var.set('Плъзгач')
    app.split_slider.set(.35)
    app._render_preview()
    app._set_zoom(1)
    pump(app,lambda:app._preview is not None and not app._preview.draft and app._busy is None)
    app.clip_warning_var.set(True)
    app._render_preview()
    app._set_tool('Четка')
    label = app.after_image_label
    def event(u,v):
        box,size,full = app._view_rects[label]
        scaling = label._get_widget_scaling()
        # Coordinates are fractions of the visible viewport, not the full RAW.
        # A 100% image may extend beyond a small/high-DPI Windows screen.
        x = u*size[0]*scaling
        y = v*size[1]*scaling
        return SimpleNamespace(x_root=label.winfo_rootx()+(label.winfo_width()-size[0]*scaling)/2+x,
                               y_root=label.winfo_rooty()+(label.winfo_height()-size[1]*scaling)/2+y)
    app._mouse_down(event(.45,.45),label)
    app._mouse_drag(event(.55,.55),label)
    app._mouse_up(event(.55,.55),label)
    assert len(app._params().masks)==1
    pump(app,lambda:app._busy is None and not app._pending_preview)
    app._set_tool('Градиент')
    app._mouse_down(event(.45,.45),label)
    app._mouse_drag(event(.6,.6),label)
    app._mouse_up(event(.6,.6),label)
    assert len(app._params().masks)==2
    pump(app,lambda:app._busy is None and not app._pending_preview)
    app._set_tool('Изрязване')
    app._mouse_down(event(.6,.6),label)  # Reverse drag must also work.
    app._mouse_drag(event(.4,.4),label)
    app._mouse_up(event(.4,.4),label)
    assert app._params().crop!=(0,0,1,1)
    pump(app,lambda:app._busy is None and not app._pending_preview)
    app._set_zoom(None)
    pump(app,lambda:bool(app._thumb_cache),timeout=20)
    assert str(dng_path) in app._thumb_cache


def test_gui_project_roundtrip(app,dng_path,tmp_path,monkeypatch):
    from app_gui import filedialog
    project = tmp_path/'test.rawstudio'
    app._add_files([dng_path])
    pump(app,lambda:app._preview is not None and app._busy is None)
    app.exposure_slider.set(-.7)
    app._commit_current()
    app.session.photos[str(dng_path)].rating=5
    monkeypatch.setattr(filedialog,'asksaveasfilename',lambda **_:str(project))
    app._save_project()
    app._clear_files()
    monkeypatch.setattr(filedialog,'askopenfilename',lambda **_:str(project))
    app._load_project()
    pump(app,lambda:app._preview is not None and app._busy is None)
    assert app._params().exposure_ev==-.7
    assert app.session.photos[str(dng_path)].rating==5
    app._undo()
    assert app._params().exposure_ev==0


def test_pause_resume_and_watcher_export_in_gui(app,dng_path,tmp_path,monkeypatch):
    from app_gui import filedialog
    from batch import FolderWatcher
    from raw_engine import RawEngine
    entered,release = threading.Event(),threading.Event()
    class BarrierEngine(RawEngine):
        def process_file(self,*args,**kwargs):
            entered.set()
            assert release.wait(5)
            return super().process_file(*args,**kwargs)
    app.engine=BarrierEngine()
    app.files=[dng_path]
    app.output_var.set(str(tmp_path/'export'))
    app._start_batch()
    pump(app,entered.is_set)
    app._toggle_pause()
    assert app._pause.is_set() and app.pause_button.cget('text')=='Продължи'
    release.set()
    heartbeat=[]
    app.after(10,lambda:heartbeat.append(True))
    pump(app,lambda:bool(heartbeat))
    assert app._busy=='batch'
    app._toggle_pause()
    pump(app,lambda:app._busy is None)
    assert (tmp_path/'export'/'synthetic.jpg').is_file()
    journal=tmp_path/'export'/'raw-studio-batch.json'
    monkeypatch.setattr(filedialog,'askopenfilename',lambda **_:str(journal))
    app.engine=RawEngine()
    app._resume_batch()
    pump(app,lambda:app._busy is None)
    assert len(list((tmp_path/'export').glob('*.jpg')))==1
    watched=tmp_path/'watched'
    watched.mkdir()
    # Short interval for the real watcher thread, not a mocked arrival event.
    original=FolderWatcher.__init__
    watchers=[]
    def fast(self,folder,recursive=False,interval=2):
        original(self,folder,recursive,.05)
        watchers.append(self)
    monkeypatch.setattr(FolderWatcher,'__init__',fast)
    monkeypatch.setattr(filedialog,'askdirectory',lambda **_:str(watched))
    app._start_watch()
    pump(app,lambda:bool(watchers) and watchers[0].initialized)
    incoming=watched/'incoming.DNG'
    incoming.write_bytes(dng_path.read_bytes())
    pump(app,lambda:(tmp_path/'export'/'incoming.jpg').is_file() and app._busy is None,timeout=20)
    app._stop_watch()
    assert app._watch_cancel.is_set()


def test_preview_100_percent_respects_display_scaling(app,dng_path):
    import customtkinter as ctk
    ctk.set_widget_scaling(1.5)
    try:
        app._add_files([dng_path])
        pump(app,lambda:app._preview is not None and app._busy is None)
        app._set_zoom(1)
        pump(app,lambda:app._preview is not None and not app._preview.draft and app._busy is None)
        app.compare_var.set('След')
        app._render_preview()
        box,size,full=app._view_rects[app.after_image_label]
        scale=app.after_image_label._get_widget_scaling()
        assert abs(size[0]*scale-(box[2]-box[0]))<=2
        assert abs(size[1]*scale-(box[3]-box[1]))<=2
    finally:
        ctk.set_widget_scaling(1)


def test_white_balance_eyedropper_applies_neutral_rgb_gains(app,dng_path):
    from PIL import Image
    from types import SimpleNamespace
    app._add_files([dng_path])
    pump(app,lambda:app._preview is not None and app._busy is None)
    app.wb_var.set('Gray World')
    app._preview.before=Image.new('RGB',(400,300),(140,130,120))
    app._set_tool('Пипетка')
    app.update_idletasks()
    label=app.before_image_label
    event=SimpleNamespace(x_root=label.winfo_rootx()+label.winfo_width()/2,
                          y_root=label.winfo_rooty()+label.winfo_height()/2)
    app._mouse_down(event,label)
    params=app._params()
    assert params.white_balance=='camera'
    assert params.wb_gains[0]<1<params.wb_gains[2]
    assert params.temperature==0 and params.tint==0
    pump(app,lambda:app._busy is None and not app._pending_preview)


def test_invalid_export_values_show_message_and_preserve_selection(app,dng_path,tmp_path,monkeypatch):
    from app_gui import messagebox
    second=tmp_path/'second.DNG'
    second.write_bytes(dng_path.read_bytes())
    app._add_files([dng_path,second])
    pump(app,lambda:app._preview is not None and app._busy is None)
    errors=[]
    monkeypatch.setattr(messagebox,'showerror',lambda title,message,**_:errors.append(message))
    app.max_edge_var.set('not a number')
    app._ui_action(app._save_project)
    assert errors and 'цяло число' in errors[-1]
    original=app.selected_var.get()
    app.selected_var.set(next(k for k,p in app._file_choices.items() if p==second))
    app._ui_action(app._selection_changed)
    assert app._active_path==str(dng_path) and app.selected_var.get()==original
    app.max_edge_var.set('0')
    app.template_var.set('../{stem}')
    app._ui_action(app._apply_builtin,'Естествено')
    assert len(errors)==3 and 'шаблон' in errors[-1]


def test_easy_import_auto_correct_export_and_modes(app,dng_path,tmp_path):
    second=tmp_path/'second.DNG';second.write_bytes(dng_path.read_bytes())
    app.exposure_slider.set(1.2)
    app._add_files([dng_path,second])
    assert app.mode_var.get()=='Лесен'
    assert all(p.params.auto_exposure and p.params.exposure_ev==0 for p in app.session.photos.values())
    pump(app,lambda:app._preview is not None and app._busy is None)
    app._simple_changed('shadows',.25)
    app._set_mode('Разширен');assert app._params().shadows==.25
    app._set_mode('Лесен');assert app._params().shadows==.25
    app._apply_export_preset('За споделяне');assert app._params().quality==90
    app.output_var.set(str(tmp_path/'result'));app._request_export()
    pump(app,lambda:app._busy is None and not app._export_pending)
    assert len(list((tmp_path/'result').glob('*.jpg')))==2
    assert '2 готови' in app.report_label.cget('text')
    assert not app._details_visible


def test_export_requested_during_preview_is_completed(app,dng_path,tmp_path):
    from raw_engine import RawEngine
    entered,release=threading.Event(),threading.Event()
    class Held(RawEngine):
        def preview(self,*args,**kwargs):
            entered.set();assert release.wait(5)
            return super().preview(*args,**kwargs)
    app.engine=Held();app._add_files([dng_path]);pump(app,entered.is_set)
    app.output_var.set(str(tmp_path/'export'));app._request_export()
    assert app._export_pending and app.simple_export_button.cget('state')=='disabled'
    release.set();pump(app,lambda:(tmp_path/'export'/'synthetic.jpg').is_file() and app._busy is None)
    assert not app._export_pending and app.progress.get()==1


def test_drop_paths_with_spaces_selection_and_stars(app,dng_path,tmp_path):
    from types import SimpleNamespace
    folder=tmp_path/'camera photos';folder.mkdir()
    second=folder/'second photo.DNG';second.write_bytes(dng_path.read_bytes())
    third=folder/'third.DNG';third.write_bytes(dng_path.read_bytes())
    assert app._drop_available
    assert app._drop_files(SimpleNamespace(data=f'{{{dng_path}}} {{{folder}}}'))=='copy'
    pump(app,lambda:len(app.files)==3 and app._busy is None and not app._pending_preview)
    app._gallery_click(dng_path,ctrl=False,shift=False)
    app._gallery_click(third,ctrl=False,shift=True)
    assert len(app._selected_photos)==3
    app._include_selected(False);assert not any(p.included for p in app.session.photos.values())
    app._gallery_click(second,ctrl=False,shift=False);app._include_selected(True)
    assert sum(p.included for p in app.session.photos.values())==1
    photo=app.session.photos[str(second)];app._rating_clicked(photo,4);assert photo.rating==4
    app._rating_clicked(photo,4);assert photo.rating==0
    pump(app,lambda:app._busy is None and not app._pending_preview)


def test_autosave_restore_keeps_masks_history_and_original_project(app,dng_path,tmp_path):
    from raw_engine import ProcessingParams
    from dataclasses import replace
    from imaging import LocalAdjustment
    app._add_files([dng_path]);pump(app,lambda:app._preview is not None and app._busy is None)
    app.session.project_path=tmp_path/'my-project.rawstudio'
    app._edit_change(exposure_ev=.6,masks=(LocalAdjustment(kind='brush',points=((.5,.5),)),))
    pump(app,lambda:app._busy is None and not app._pending_preview)
    app._save_recovery();pump(app,lambda:app._saved_snapshot is not None)
    assert app.session.project_path==tmp_path/'my-project.rawstudio'
    app._clear_files();app._open_project_path(app._autosave_path,recovery=True)
    assert app._params().exposure_ev==.6 and len(app._params().masks)==1
    assert app.session.project_path==tmp_path/'my-project.rawstudio'
    app._undo();assert app._params().exposure_ev==0 and not app._params().masks
    pump(app,lambda:app._busy is None and not app._pending_preview)


def test_inline_validation_and_retry_only_failed_files(app,dng_path,tmp_path):
    broken=tmp_path/'broken.DNG';broken.write_bytes(b'not a raw')
    app._add_files([dng_path,broken]);pump(app,lambda:app._preview is not None and app._busy is None)
    app.output_var.set(str(tmp_path/'export'));app.max_edge_var.set('wrong');app._request_export()
    assert app._busy is None and app.simple_export_error.cget('text')
    assert app.edge_entry.cget('border_color')=='#ed7878'
    app.max_edge_var.set('0');app._request_export();pump(app,lambda:app._busy is None)
    assert '1 готови' in app.report_label.cget('text') and len(app._failed_files)==1
    broken.write_bytes(dng_path.read_bytes());app._retry_failed();pump(app,lambda:app._busy is None)
    outputs=list((tmp_path/'export').glob('*.jpg'))
    assert {p.name for p in outputs}=={'synthetic.jpg','broken.jpg'}


def test_numeric_entry_exact_value_and_small_window(app,dng_path):
    app._add_files([dng_path]);pump(app,lambda:app._preview is not None and app._busy is None)
    slider,label=app._simple_sliders['exposure_ev']
    slider.master.pack(fill='x');app.update()
    app.focus_force();app.update()
    app._edit_numeric(slider,label,lambda value:app._simple_changed('exposure_ev',value),'Яркост')
    app.update()
    entry=next(w for w in label.master.winfo_children() if w.winfo_class()=='Frame' and hasattr(w,'get'))
    entry._entry.focus_force();app.update()
    entry.delete(0,'end');entry.insert(0,'0.37');entry._entry.event_generate('<Return>');app.update()
    assert app._params().exposure_ev==.37
    app.geometry('900x640');app.update();app._responsive_layout()
    assert app.compare_control.grid_info()['column']==0
    assert app.after_image_label.winfo_height()>=120
    pump(app,lambda:app._busy is None and not app._pending_preview)


def test_shutdown_waits_for_latest_recovery_snapshot(app,dng_path,monkeypatch):
    import experience
    from studio import EditSession
    from dataclasses import replace
    entered,release=threading.Event(),threading.Event()
    original=experience.write_snapshot
    calls=[]
    def hold_first(snapshot):
        calls.append(snapshot)
        if len(calls)==1:
            entered.set();assert release.wait(5)
        original(snapshot)
    monkeypatch.setattr(experience,'write_snapshot',hold_first)
    app._add_files([dng_path]);pump(app,lambda:app._preview is not None and app._busy is None)
    app._save_recovery();pump(app,entered.is_set)
    app.exposure_slider.set(.8);app._commit_current();app._on_close()
    assert app._autosave_next is not None
    release.set()
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        try:app.update()
        except Exception:pass
        if len(calls)==2 and not app._autosave_worker.is_alive():break
        time.sleep(.01)
    restored=EditSession.load(app._autosave_path)
    assert len(calls)==2 and restored.photos[str(dng_path)].params.exposure_ev==.8
