"""Exercise the actual bundled Tk/DnD/LibRaw/OpenCV dependencies in CI."""
from pathlib import Path
import json
import time
import os
import cv2
import customtkinter as ctk
from app_gui import AppGUI


def smoke(input_path,output_dir):
    directory=Path(output_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    os.environ['RAW_STUDIO_CONFIG_DIR']=str(directory/'settings')
    cv2.setNumThreads(2);ctk.set_appearance_mode('Dark')
    app=AppGUI()
    try:
        if not app._drop_available:raise RuntimeError('TkDnD failed in the Windows bundle')
        app._add_files([Path(input_path).resolve()]);app.output_var.set(str(directory))
        app._request_export()
        deadline=time.monotonic()+90
        while app._busy or app._export_pending or not list(directory.glob('*.jpg')):
            app.update();time.sleep(.01)
            if time.monotonic()>deadline:raise RuntimeError('Bundled import/export timed out')
        if app.progress.get()!=1:raise RuntimeError('Export did not complete')
        (directory/'smoke.json').write_text(json.dumps({'success':True,'tkdnd':app.TkdndVersion,'files':[p.name for p in directory.glob('*.jpg')]}))
    finally:
        app._on_close()
        while app._autosave_worker and app._autosave_worker.is_alive():
            app.update();time.sleep(.01)

