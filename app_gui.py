"""CustomTkinter desktop UI. All Tk operations stay on the main thread.

Workers communicate through Queue; a short after() poll delivers their results.
Settings are captured before spawning a worker, never read from Tk variables there.
"""
from __future__ import annotations

from datetime import datetime
from dataclasses import asdict, replace
import io
import math
import time
from pathlib import Path
from queue import Empty, Queue
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageDraw
import numpy as np
import rawpy

from batch import discover_raws, run_batch, unique_inputs, FolderWatcher
from studio import (EditSession, PhotoState, BUILTIN_PRESETS, EDIT_FIELDS, EXPORT_FIELDS,
                    save_preset, load_preset, save_lens_profile, load_lens_profile, read_json, params_from_dict)
from experience import StudioExperience
from studio_workflow import COPY_GROUPS
from imaging import LocalAdjustment
from raw_engine import (ProcessingCancelled, ProcessingParams, RAW_EXTENSIONS,
                        RawEngine, PreviewResult)

WB_LABELS = {
    "От камерата": "camera",
    "Автоматичен (LibRaw)": "auto",
    "Gray World": "gray_world",
    "Shades of Gray": "shades_of_gray",
}
BG = ("#f3f5f8", "#101419")
CARD = ("#ffffff", "#1a212a")
MUTED = ("#596577", "#9eacbd")
ACCENT = ("#087c6d", "#20b99b")


class PreviewLabel(ctk.CTkLabel):
    """Detach the native Tk image when clearing a CustomTkinter preview.

    CTkLabel 6.0 leaves the native label unchanged for image=None. Once the
    old CTkImage is collected, Tk still refers to its deleted pyimage name
    and rejects the next text/image update. Clear that Tcl option explicitly.
    """

    def _update_image(self) -> None:
        if self._image is None:
            self._label.configure(image="")
        else:
            super()._update_image()


class AppGUI(StudioExperience, ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RAW Studio · Python RAW Processor")
        width = min(1380, max(1100, self.winfo_screenwidth() - 90))
        height = min(900, max(720, self.winfo_screenheight() - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(900, 640)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.engine = RawEngine()
        self.files: list[Path] = []
        self.events: Queue[tuple[int, str, dict]] = Queue()
        self._job_id = 0
        self._busy: str | None = None
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._closing = False
        self._preview_timer: str | None = None
        self._resize_timer: str | None = None
        self._pending_preview = False
        self._preview: PreviewResult | None = None
        self._batch_preview: Image.Image | None = None
        self._images: list[ctk.CTkImage] = []
        self._locked_widgets: list = []
        self._setting_widgets: list = []
        self._file_choices: dict[str, Path] = {}
        self._log_lines = 0
        self.session = EditSession()
        self._active_path = None
        self._base_edit = ProcessingParams()
        self._loading = False
        self._slider_labels = {}
        self._extra_sliders = {}
        self._clipboard = None
        self._pause = threading.Event()
        self._watch_cancel = threading.Event()
        self._watch_worker = None
        self._watch_pending = []
        self._thumb_cancel = threading.Event()
        self._thumb_worker = None
        self._thumb_epoch = 0
        self._thumb_cache = {}
        self._gallery_page = 0
        self._gallery_buttons = {}
        self._gallery_controls = []
        self._zoom = None
        self._center = (.5,.5)
        self._view_rects = {}
        self._stroke = []
        self._drag_origin = None
        self._mouse_point = None
        self._watch_timer = None
        self._gallery_timer = None
        self._watch_exporting = False
        self._watch_generation = 0
        self._metadata = {}
        self._project_dirty = False
        self.tool_var = tk.StringVar(value='Местене')
        self.ratio_var = tk.StringVar(value='Свободно')
        self.preset_var = tk.StringVar(value='Естествено')
        self.favorite_filter_var = tk.BooleanVar(value=False)
        self.clip_warning_var = tk.BooleanVar(value=False)
        self.template_var = tk.StringVar(value='{stem}')
        self.max_edge_var = tk.StringVar(value='0')
        self.preserve_exif_var = tk.BooleanVar(value=False)
        self.exif_camera_var = tk.BooleanVar(value=True)
        self.exif_date_var = tk.BooleanVar(value=True)
        self.exif_exposure_var = tk.BooleanVar(value=True)
        self.monochrome_var = tk.BooleanVar(value=False)
        self.lens_var = tk.BooleanVar(value=False)
        self.auto_lens_var = tk.BooleanVar(value=False)


        self.wb_var = tk.StringVar(value="От камерата")
        self.auto_exposure_var = tk.BooleanVar(value=True)
        self.tone_var = tk.BooleanVar(value=True)
        self.denoise_var = tk.BooleanVar(value=False)
        self.sharpen_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.exact_preview_var = tk.BooleanVar(value=False)
        self.format_var = tk.StringVar(value="JPG")
        self.depth_var = tk.StringVar(value="16 бита")
        self.output_var = tk.StringVar(value="")
        self.selected_var = tk.StringVar(value="Няма избрани файлове")
        self.compare_var = tk.StringVar(value="След")
        self._init_experience()

        self._build_sidebar()
        self._build_main()
        self._finish_experience()
        self._update_format()
        self._bind_editor_keys()
        self._watch_timer = self.after(500,self._drain_watch)
        self._poll_id = self.after(70, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log("Избери RAW снимки, изходна папка и настройки. Оригиналите се запазват.")

    @staticmethod
    def _label(parent, text: str, **kwargs):
        return ctk.CTkLabel(parent, text=text, anchor="w", **kwargs)

    def _section(self, parent, text: str):
        return self._accordion(parent, text)

    def _button(self, parent, text: str, command: Callable, **kwargs):
        button = ctk.CTkButton(parent, text=text, command=lambda:self._ui_action(command), height=35,
                               corner_radius=8, **kwargs)
        button.pack(fill="x", padx=12, pady=4)
        self._locked_widgets.append(button)
        return button

    def _switch(self, parent, text: str, variable: tk.Variable,
                command: Callable | None = None, *, setting: bool = True):
        widget = ctk.CTkSwitch(parent, text=text, variable=variable,
                              command=command or self._schedule_preview,
                              progress_color=ACCENT, font=ctk.CTkFont(size=12))
        widget.pack(fill="x", padx=12, pady=7)
        (self._setting_widgets if setting else self._locked_widgets).append(widget)
        return widget

    def _slider(self, parent, title: str, low: float, high: float, initial: float,
                steps: int, formatter: Callable[[float], str]):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(7, 0))
        title_label = self._label(row, title, font=ctk.CTkFont(size=12))
        title_label.pack(side="left")
        value_label = ctk.CTkLabel(row, text=formatter(initial), text_color=MUTED,
                                  font=ctk.CTkFont(size=12))
        value_label.pack(side="right")

        def changed(value: float) -> None:
            value_label.configure(text=formatter(value))
            self._sync_simple_controls()
            self._schedule_preview()

        slider = ctk.CTkSlider(parent, from_=low, to=high, number_of_steps=steps,
                              command=changed, progress_color=ACCENT, button_color=ACCENT)
        slider.set(initial)
        slider.pack(fill="x", padx=12, pady=(5, 7))
        self._setting_widgets.append(slider)
        self._slider_labels[slider] = (value_label,formatter)
        self._decorate_slider(slider,value_label,title_label,title,initial,changed)
        return slider, value_label

    def _build_sidebar(self) -> None:
        self._build_simple_sidebar()
        self.advanced_sidebar = sidebar = ctk.CTkFrame(self,width=340,corner_radius=0,fg_color=CARD)
        sidebar.grid(row=0,column=0,sticky='nsew')
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0,weight=1)
        sidebar.grid_rowconfigure(1,weight=1)
        brand = ctk.CTkFrame(sidebar,fg_color='transparent')
        brand.grid(row=0,column=0,sticky='ew',padx=20,pady=(16,4))
        self._label(brand,'RAW Studio',font=ctk.CTkFont(size=25,weight='bold')).pack(side='left')
        ctk.CTkButton(brand,text='Лесен режим',width=95,command=lambda:self._set_mode('Лесен')).pack(side='right')
        tabs = ctk.CTkTabview(sidebar,fg_color=CARD,segmented_button_selected_color=ACCENT)
        tabs.grid(row=1,column=0,sticky='nsew',padx=4)
        panels = {}
        for name in ['Редакция','Маски','Експорт','Проект']:
            tab = tabs.add(name)
            tab.grid_rowconfigure(0,weight=1)
            tab.grid_columnconfigure(0,weight=1)
            panel = ctk.CTkScrollableFrame(tab,fg_color=CARD,corner_radius=0)
            panel.grid(row=0,column=0,sticky='nsew')
            panels[name] = panel
        controls = panels['Редакция']
        edit_root = controls
        controls = self._section(edit_root,'СНИМКИ И ГОТОВИ СТИЛОВЕ')
        self._button(controls,'+ Добави RAW файлове',self._choose_files,fg_color=ACCENT)
        self._button(controls,'Избери папка със снимки',self._choose_folder)
        self._switch(controls,'Включи подпапките',self.recursive_var,command=lambda:None,setting=False)
        self.file_count_label = self._label(controls,'0 избрани снимки',text_color=MUTED)
        self.file_count_label.pack(fill='x',padx=12)
        self._button(controls,'Изчисти списъка',self._clear_files)
        preset = ctk.CTkOptionMenu(controls,values=list(BUILTIN_PRESETS),variable=self.preset_var,command=lambda name:self._ui_action(self._apply_builtin,name))
        preset.pack(fill='x',padx=12,pady=8)
        self._setting_widgets.append(preset)
        self._button(controls,'Запази собствен стил…',self._save_preset)
        self._button(controls,'Зареди стил…',self._load_preset)
        self._button(controls,'Копирай настройките',self._copy_settings)
        self.copy_group_menu = ctk.CTkOptionMenu(controls,values=list(COPY_GROUPS),variable=self.copy_group_var)
        self.copy_group_menu.pack(fill='x',padx=12,pady=4)
        self._setting_widgets.append(self.copy_group_menu)
        self._button(controls,'Приложи към включените снимки',self._paste_settings)
        controls = self._section(edit_root,'БАЛАНС И СВЕТЛИНА')
        self.wb_menu = ctk.CTkOptionMenu(controls,values=list(WB_LABELS),variable=self.wb_var,command=lambda _:self._schedule_preview())
        self.wb_menu.pack(fill='x',padx=12,pady=6)
        self._setting_widgets.append(self.wb_menu)
        self._button(controls,'Пипетка за баланс на бялото',lambda:self._set_tool('Пипетка'))
        self._switch(controls,'Автоматична експозиция',self.auto_exposure_var)
        self.exposure_slider,_ = self._slider(controls,'Експозиция',-3,3,0,120,lambda v:f'{v:+.2f} EV')
        for key,title,low,high,initial in [
            ('shadows','Сенки',-1,1,0),('highlights','Светли участъци',-1,1,0),
            ('whites','Бели тонове',-1,1,0),('blacks','Черни тонове',-1,1,0),
            ('temperature','Температура',-1,1,0),('tint','Оттенък',-1,1,0),
            ('saturation','Наситеност',0,2,1),('vibrance','Живост на цветовете',-1,1,0)]:
            self._extra_sliders[key],_ = self._slider(controls,title,low,high,initial,100,lambda v:f'{v:+.2f}')
        self._switch(controls,'Черно-бяло',self.monochrome_var)
        self._switch(controls,'Локален контраст (CLAHE)',self.tone_var)
        self.tone_slider,_ = self._slider(controls,'Сила на CLAHE',0,1,.30,100,lambda v:f'{v:.0%}')
        self.clip_slider,_ = self._slider(controls,'CLAHE clip limit',.1,8,2,158,lambda v:f'{v:.2g}')
        controls = self._section(edit_root,'ДЕТАЙЛ')
        self._switch(controls,'Премахване на шум',self.denoise_var)
        self.noise_slider,_ = self._slider(controls,'Сила на филтъра',0,15,3,30,lambda v:f'{v:.1f}')
        self._switch(controls,'Изостряне',self.sharpen_var)
        self.sharp_slider,_ = self._slider(controls,'Сила на изостряне',0,2,.45,80,lambda v:f'{v:.2f}')
        self.hist_canvas = tk.Canvas(controls,height=90,bg='#131a22',highlightthickness=0)
        self.hist_canvas.pack(fill='x',padx=12,pady=10)
        self.hist_label = self._label(controls,'Хистограмата се появява след преглед.',text_color=MUTED,font=ctk.CTkFont(size=11))
        self.hist_label.pack(fill='x',padx=12)
        self._switch(controls,'Покажи загубените тонове',self.clip_warning_var,command=self._render_preview)

        tools = panels['Маски']
        tools_root = tools
        tools = self._section(tools_root,'ИЗРЯЗВАНЕ И ГЕОМЕТРИЯ')
        self._label(tools,'Инструментът се използва върху снимката.',wraplength=260).pack(fill='x',padx=12)
        tool_menu = ctk.CTkOptionMenu(tools,values=['Местене','Изрязване','Пипетка','Четка','Градиент'],variable=self.tool_var,command=self._set_tool)
        tool_menu.pack(fill='x',padx=12,pady=8)
        self._setting_widgets.append(tool_menu)
        ratio = ctk.CTkOptionMenu(tools,values=['Свободно','1:1','4:5','3:2','16:9'],variable=self.ratio_var)
        ratio.pack(fill='x',padx=12,pady=8)
        self._setting_widgets.append(ratio)
        self._button(tools,'Завърти на 90°',self._rotate)
        self._button(tools,'Огледално хоризонтално',lambda:self._edit_change(flip_horizontal=not self._params().flip_horizontal))
        self._button(tools,'Огледално вертикално',lambda:self._edit_change(flip_vertical=not self._params().flip_vertical))
        self._button(tools,'Възстанови целия кадър',lambda:self._edit_change(crop=(0,0,1,1)))
        self._extra_sliders['straighten'],_ = self._slider(tools,'Изправяне на хоризонта',-15,15,0,300,lambda v:f'{v:+.1f}°')
        tools = self._section(tools_root,'ЛОКАЛНИ КОРЕКЦИИ')
        self.local_radius,_ = self._slider(tools,'Размер на четката',.005,.5,.08,99,lambda v:f'{v:.1%}')
        self.local_ev,_ = self._slider(tools,'Локална експозиция',-3,3,.5,120,lambda v:f'{v:+.2f} EV')
        self.local_shadows,_ = self._slider(tools,'Локални сенки',-1,1,0,100,lambda v:f'{v:+.2f}')
        self.local_saturation,_ = self._slider(tools,'Локална наситеност',0,2,1,100,lambda v:f'{v:.2f}')
        self.mask_label = self._label(tools,'0 локални маски',text_color=MUTED)
        self.mask_label.pack(fill='x',padx=12)
        self._button(tools,'Изтрий последната маска',lambda:self._edit_change(masks=self._params().masks[:-1]))
        self._button(tools,'Изтрий всички маски',lambda:self._edit_change(masks=()))
        self._label(tools,'Четка: рисувай с ляв бутон. Градиент: влачи от зона без корекция към зона с пълна корекция. Маските следват координатите на изрязания кадър.',wraplength=260,justify='left',text_color=MUTED).pack(fill='x',padx=12,pady=8)
        tools = self._section(tools_root,'КОРЕКЦИИ НА ОБЕКТИВА')
        self._switch(tools,'Включи корекциите',self.lens_var)
        self._switch(tools,'Автоматичен собствен профил',self.auto_lens_var)
        for key,title,low,high in [('lens_k1','Изкривяване k1',-.5,.5),('lens_k2','Изкривяване k2',-.5,.5),('vignette','Винетиране',-1,2),('ca_red','Червен цветен кант',-.02,.02),('ca_blue','Син цветен кант',-.02,.02)]:
            self._extra_sliders[key],_ = self._slider(tools,title,low,high,0,200,lambda v:f'{v:+.4f}')
        self._button(tools,'Зареди профил за обектив…',self._load_lens)
        self._button(tools,'Запази профил за този обектив…',self._save_lens)

        export = panels['Експорт']
        export = self._section(export,'ФОРМАТ И РАЗМЕР')
        self.format_control = ctk.CTkSegmentedButton(export,values=['JPG','PNG'],variable=self.format_var,command=self._update_format,selected_color=ACCENT)
        self.format_control.pack(fill='x',padx=12,pady=6)
        self._setting_widgets.append(self.format_control)
        self.quality_slider,self.quality_label = self._slider(export,'Качество на JPG',1,100,95,99,lambda v:f'{round(v)}%')
        self.compression_slider,self.compression_label = self._slider(export,'PNG компресия',0,9,4,9,lambda v:str(round(v)))
        self.depth_menu = ctk.CTkOptionMenu(export,values=['8 бита','16 бита'],variable=self.depth_var)
        self.depth_menu.pack(fill='x',padx=12,pady=7)
        self._setting_widgets.append(self.depth_menu)
        self.format_hint = self._label(export,'',wraplength=260,justify='left',text_color=MUTED,font=ctk.CTkFont(size=11))
        self.format_hint.pack(fill='x',padx=12,pady=4)
        self._extra_sliders['max_edge'],_ = self._slider(export,'Дълга страна (0 = оригинал)',0,16000,0,160,lambda v:f'{round(v)} px')
        self.edge_entry = edge_entry = ctk.CTkEntry(export,textvariable=self.max_edge_var,placeholder_text='Точен размер, например 2048')
        edge_entry.pack(fill='x',padx=12,pady=4)
        self._setting_widgets.append(edge_entry)
        self._extra_sliders['max_edge'].configure(command=lambda value:self._set_edge(value))
        self._label(export,'Шаблон за име',text_color=MUTED).pack(fill='x',padx=12,pady=(12,4))
        self.naming_entry = naming = ctk.CTkEntry(export,textvariable=self.template_var)
        naming.pack(fill='x',padx=12,pady=4)
        self.export_error = self._label(export,'',text_color='#ed7878',wraplength=260)
        self.export_error.pack(fill='x',padx=12)
        self._setting_widgets.append(naming)
        self._label(export,'{stem} · {index:04d} · {date} · {camera}',text_color=MUTED,font=ctk.CTkFont(size=10)).pack(fill='x',padx=12)
        self._switch(export,'Запази избрани EXIF данни',self.preserve_exif_var,command=lambda:None)
        self._switch(export,'Фотоапарат и обектив',self.exif_camera_var,command=lambda:None)
        self._switch(export,'Дата на снимката',self.exif_date_var,command=lambda:None)
        self._switch(export,'Експозиция, ISO и фокусно разстояние',self.exif_exposure_var,command=lambda:None)
        self._label(export,'GPS и серийни номера не се копират.',text_color=MUTED,font=ctk.CTkFont(size=11)).pack(fill='x',padx=12)
        self.output_entry = ctk.CTkEntry(export,textvariable=self.output_var,placeholder_text='Папка за резултатите')
        self.output_entry.pack(fill='x',padx=12,pady=12)
        self._locked_widgets.append(self.output_entry)
        self._button(export,'Избери изходна папка',self._choose_output)
        self._switch(export,'Презаписвай готовите файлове',self.overwrite_var,command=lambda:None)
        self._button(export,'Продължи запазена серия…',self._resume_batch)

        project = panels['Проект']
        project_root = project
        project = self._section(project_root,'ПРОЕКТ И ИСТОРИЯ')
        self._button(project,'Запази проект…',self._save_project)
        self._button(project,'Отвори проект…',self._load_project)
        self._button(project,'Отмени (Ctrl+Z)',self._undo)
        self._button(project,'Повтори (Ctrl+Y)',self._redo)
        self._button(project,'Нулирай редакциите',lambda:self._replace_edit(ProcessingParams()))
        self.project_label = self._label(project,'Няма отворен проект',wraplength=260,text_color=MUTED)
        self.project_label.pack(fill='x',padx=12,pady=10)
        project = self._section(project_root,'НАБЛЮДАВАНА ПАПКА')
        self._button(project,'Избери папка за автоматичен експорт…',self._start_watch)
        self.watch_stop = self._button(project,'Спри наблюдението',self._stop_watch)
        self.watch_label = self._label(project,'Наблюдението е изключено.',wraplength=260,text_color=MUTED)
        self.watch_label.pack(fill='x',padx=12,pady=8)
        self.metadata_box = ctk.CTkTextbox(project,height=180,wrap='word')
        self.metadata_box.pack(fill='x',padx=12,pady=12)
        self.metadata_box.configure(state='disabled')
        footer = ctk.CTkFrame(sidebar,fg_color='transparent')
        footer.grid(row=2,column=0,sticky='ew',padx=18,pady=12)
        self.start_button = ctk.CTkButton(footer,text='Експортирай включените снимки',height=40,fg_color=ACCENT,command=self._request_export)
        self.start_button.pack(fill='x')
        self._export_buttons.append(self.start_button)
        self._locked_widgets.append(self.start_button)
        self.theme_menu = ctk.CTkOptionMenu(footer,values=['Тъмна тема','Светла тема','Системна тема'],command=self._change_theme,height=26)
        self.theme_menu.set('Тъмна тема')
        self.theme_menu.pack(fill='x',pady=(8,0))

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        self.main_panel = main
        main.grid(row=0, column=1, sticky="nsew", padx=24, pady=22)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self._label(header, "ПРЕДВАРИТЕЛЕН ПРЕГЛЕД", text_color=MUTED,
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self._build_workspace_header(header)

        selection = ctk.CTkFrame(main, fg_color="transparent")
        selection.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        selection.grid_columnconfigure(0, weight=1)
        self.file_menu = ctk.CTkOptionMenu(selection, values=["Няма избрани файлове"],
                                          variable=self.selected_var,
                                          command=lambda _: self._ui_action(self._selection_changed))
        self.file_menu.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._locked_widgets.append(self.file_menu)
        self.preview_button = ctk.CTkButton(selection, text="Обнови прегледа", width=150,
                                           command=self._request_preview)
        self.preview_button.grid(row=0, column=1)
        self.selection_panel = selection
        self._locked_widgets.append(self.preview_button)
        self.preview_switch = ctk.CTkSwitch(selection, text="Точен преглед (по-бавен)",
                                            variable=self.exact_preview_var,
                                            command=self._schedule_preview,
                                            progress_color=ACCENT)
        self.preview_switch.grid(row=1, column=0, sticky="w", pady=(12, 0))
        self._setting_widgets.append(self.preview_switch)
        self.compare_control = ctk.CTkSegmentedButton(selection, values=["Преди", "След", "Сравнение", "Плъзгач"],
                                                      variable=self.compare_var,
                                                      command=lambda _: self._render_preview(),
                                                      selected_color=ACCENT)
        self.compare_control.grid(row=1, column=1, pady=(12, 0))

        self.preview_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12)
        self.preview_card.grid(row=2, column=0, sticky="nsew")
        self.preview_card.grid_rowconfigure(0, weight=1)
        self.preview_card.grid_columnconfigure((0, 1), weight=1, uniform="preview")
        self.before_panel = self._make_preview_panel(self.preview_card, "ПРЕДИ КОРЕКЦИИТЕ", 0)
        self.after_panel = self._make_preview_panel(self.preview_card, "СЛЕД КОРЕКЦИИТЕ", 1)
        self.before_image_label.configure(text="Добави RAW снимка\nза предварителен преглед")
        self.after_image_label.configure(text="Автоматични корекции\nс пълен контрол")
        self.preview_card.bind("<Configure>", self._preview_resized)
        self.preview_info = self._label(main, "Размерът за експорт се избира в раздел Експорт.",
                                        text_color=MUTED, font=ctk.CTkFont(size=11),height=18)
        self.preview_info.grid(row=3, column=0, sticky="ew", pady=(8, 14))

        toolbar = ctk.CTkFrame(main,fg_color='transparent')
        toolbar.grid(row=8,column=0,sticky='ew',pady=4)
        for title,command in [('Побери',lambda:self._set_zoom(None)),('100%',lambda:self._set_zoom(1)),('−',lambda:self._zoom_step(.8)),('+',lambda:self._zoom_step(1.25)),('↶',self._undo),('↷',self._redo)]:
            ctk.CTkButton(toolbar,text=title,width=54,height=25,command=lambda fn=command:self._ui_action(fn)).pack(side='left',padx=2)
        self.zoom_label = self._label(toolbar,'Побери',text_color=MUTED)
        self.zoom_label.pack(side='left',padx=8)
        self.split_slider = ctk.CTkSlider(toolbar,from_=0,to=1,number_of_steps=100,command=lambda _:self._render_preview(),width=130)
        self.split_slider.set(.5)
        self.split_slider.pack(side='right',padx=4)
        self.split_hint = self._label(toolbar,'Преди / След',text_color=MUTED)
        self.split_hint.pack(side='right')
        self._build_gallery(main)
        self._build_welcome_and_report(main)

        self.progress_card = progress_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=10)
        progress_card.grid(row=4, column=0, sticky="ew", pady=(0, 14))
        progress_card.grid_columnconfigure(0, weight=1)
        self.status_label = self._label(progress_card, "Готов за работа", font=ctk.CTkFont(size=13))
        self.status_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 2))
        self.stop_button = ctk.CTkButton(progress_card, text="Спри", width=78, height=30,
                                        state="disabled", fg_color=("#ad3a3a", "#873e48"),
                                        hover_color=("#942f2f", "#70333c"), command=self._stop)
        self.stop_button.grid(row=0, column=2, rowspan=2, padx=8, pady=12)
        self.pause_button = ctk.CTkButton(progress_card,text='Пауза',width=80,height=30,state='disabled',command=self._toggle_pause)
        self.pause_button.grid(row=0,column=1,rowspan=2,padx=4)
        self.progress_label = self._label(progress_card, "0 / 0 снимки · 0%", text_color=MUTED,
                                          font=ctk.CTkFont(size=11))
        self.progress_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
        self.progress = ctk.CTkProgressBar(progress_card, progress_color=ACCENT, height=6)
        self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", padx=16, pady=(5, 14))
        self.progress.set(0)

        self.log_header = log_header = ctk.CTkFrame(main, fg_color="transparent")
        log_header.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        self._label(log_header, "ДНЕВНИК", text_color=MUTED,
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.open_output_button = ctk.CTkButton(log_header, text="Отвори резултатите", width=150,
                                               height=27, command=self._open_output)
        self.open_output_button.pack(side="right")
        self._locked_widgets.append(self.open_output_button)
        self.log_box = ctk.CTkTextbox(main, height=65, fg_color=CARD,
                                      font=ctk.CTkFont(family="Consolas", size=11), wrap="word")
        self.log_box.grid(row=6, column=0, sticky="ew")
        self.log_box.tag_config("error", foreground="#ed7878")
        self.log_box.tag_config("success", foreground="#32bca0")
        self.log_box.configure(state="disabled")

    def _make_preview_panel(self, parent, title: str, column: int):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=column, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        self._label(frame, title, text_color=MUTED,
                    font=ctk.CTkFont(size=11, weight="bold"),height=18).grid(row=0, column=0, sticky="ew", pady=4)
        label = PreviewLabel(frame, text="", text_color=MUTED, corner_radius=8,
                             fg_color=("#eef1f5", "#131a22"), font=ctk.CTkFont(size=15))
        label.grid(row=1, column=0, sticky="nsew")
        for sequence, handler in [('<ButtonPress-1>',self._mouse_down),('<B1-Motion>',self._mouse_drag),('<ButtonRelease-1>',self._mouse_up),('<MouseWheel>',self._mouse_wheel),('<Button-4>',self._mouse_wheel),('<Button-5>',self._mouse_wheel)]:
            label._label.bind(sequence,lambda event,box=label,fn=handler:self._ui_action(fn,event,box))
        if column == 0:
            self.before_image_label = label
        else:
            self.after_image_label = label
        return frame

    def _params(self) -> ProcessingParams:
        # Main thread only. Capture a fixed snapshot for the whole batch.
        return replace(self._base_edit,
            **{k:round(v.get(),6) for k,v in self._extra_sliders.items() if k!='max_edge'},
            max_edge=self._export_edge(),
            monochrome=self.monochrome_var.get(), lens_enabled=self.lens_var.get(), auto_lens=self.auto_lens_var.get(),
            filename_template=self.template_var.get(), preserve_exif=self.preserve_exif_var.get(),
            exif_camera=self.exif_camera_var.get(), exif_date=self.exif_date_var.get(), exif_exposure=self.exif_exposure_var.get(),
            white_balance=WB_LABELS[self.wb_var.get()],
            auto_exposure=self.auto_exposure_var.get(),
            exposure_ev=round(self.exposure_slider.get(), 6),
            tone_mapping=self.tone_var.get(), clahe_clip=round(self.clip_slider.get(),6),
            tone_strength=round(self.tone_slider.get(),6), denoise=self.denoise_var.get(),
            denoise_strength=round(self.noise_slider.get(),6), sharpen=self.sharpen_var.get(),
            sharpen_amount=round(self.sharp_slider.get(),6), output_format=self.format_var.get().lower(),
            quality=round(self.quality_slider.get()),
            png_compression=round(self.compression_slider.get()),
            png_bit_depth=16 if self.depth_var.get() == "16 бита" else 8,
        )

    def _update_format(self, _value=None) -> None:
        is_jpg = self.format_var.get() == "JPG"
        editable = self._busy not in {"batch", "scan"} and not self._closing
        self.quality_slider.configure(state="normal" if editable and is_jpg else "disabled")
        self.compression_slider.configure(state="normal" if editable and not is_jpg else "disabled")
        self.depth_menu.configure(state="normal" if editable and not is_jpg else "disabled")
        self.format_hint.configure(text=(
            "JPG: по-високо качество = по-голям файл. Изходът е 8-битов."
            if is_jpg else "PNG: без загуба. Компресия 0–9 променя размера и времето, не качеството."))

    @staticmethod
    def _change_theme(value: str) -> None:
        ctk.set_appearance_mode({"Тъмна тема": "Dark", "Светла тема": "Light",
                                 "Системна тема": "System"}[value])

    def _log(self, message: str, tag: str = "") -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n", tag or ())
        self._log_lines += 1
        if self._log_lines > 600:
            self.log_box.delete("1.0", "101.0")
            self._log_lines -= 100
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _choose_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(RAW_EXTENSIONS))
        patterns += " " + patterns.upper()
        selected = filedialog.askopenfilenames(parent=self, title="Избери RAW снимки",
                                               filetypes=[("RAW снимки", patterns), ("Всички файлове", "*.*")])
        if selected:
            self._add_files([Path(path) for path in selected])

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Папка с RAW снимки")
        if not folder:
            return
        recursive = self.recursive_var.get()

        def scan(emit, cancel):
            try:
                files, errors = discover_raws(folder, recursive, cancel)
                emit("scan_result", {"files": files, "errors": errors})
            except ProcessingCancelled:
                emit("scan_cancelled", {})
            except Exception as error:
                emit("fatal_error", {"error": str(error)})
            finally:
                emit("job_done", {})

        self._launch("scan", scan)
        self.status_label.configure(text="Търсене на RAW файлове…")

    def _add_files(self, files: list[Path]) -> None:
        old_count = len(self.files)
        current_params = self._params()
        self.session.defaults = self._import_defaults(current_params)
        self.session.add(files)
        self.files = unique_inputs([*self.files, *files])
        self._file_choices = {f"{i:03d} · {path.name}": path for i, path in enumerate(self.files, 1)}
        choices = list(self._file_choices) or ["Няма избрани файлове"]
        current = self.selected_var.get()
        self.file_menu.configure(values=choices)
        if current not in self._file_choices:
            self.selected_var.set(choices[0])
        self.file_count_label.configure(text=f"{len(self.files)} избрани снимки")
        self.progress_label.configure(text=f"0 / {len(self.files)} снимки · 0%")
        self.progress.set(0)
        if self.files and not self.output_var.get().strip():
            self.output_var.set(str(self.files[0].parent / "processed"))
        self._log(f"Добавени: {len(self.files) - old_count}. Общо: {len(self.files)} RAW снимки.")
        if self.files and self._active_path is None:
            self._active_path = str(self._file_choices[self.selected_var.get()])
            self._load_params(self.session.photos[self._active_path].params)
        self._refresh_gallery()
        self._update_experience()
        if self.files:
            self._pending_preview = True
            self._schedule_preview()

    def _clear_files(self) -> None:
        self._thumb_cancel.set()
        self.session = EditSession()
        self._active_path = None
        self._metadata = {}
        self.files.clear()
        self._file_choices.clear()
        self.file_menu.configure(values=["Няма избрани файлове"])
        self.selected_var.set("Няма избрани файлове")
        self.file_count_label.configure(text="0 избрани снимки")
        self.progress_label.configure(text="0 / 0 снимки · 0%")
        self.progress.set(0)
        self._preview = None
        self._batch_preview = None
        self._pending_preview = False
        self._images.clear()
        self.before_image_label.configure(image=None, text="Добави RAW снимка\nза предварителен преглед")
        self.after_image_label.configure(image=None, text="Автоматични корекции\nс пълен контрол")
        self.preview_info.configure(text="Размерът за експорт се избира в раздел Експорт.")
        self._refresh_gallery()
        self.hist_canvas.delete('all')
        self._selected_photos.clear()
        self._last_selected_index = None
        self._selection_anchor = None
        self._project_dirty = True
        self._update_experience()
        self._log("Списъкът е изчистен.")

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Избери изходна папка")
        if folder:
            self.output_var.set(folder)

    def _selection_changed(self) -> None:
        try:
            self._commit_current()
        except ValueError:
            original = next((key for key,path in self._file_choices.items() if str(path)==self._active_path),None)
            if original: self.selected_var.set(original)
            raise
        path = self._file_choices.get(self.selected_var.get())
        self._active_path = str(path) if path else None
        if self._active_path in self.session.photos:
            self._load_params(self.session.photos[self._active_path].params)
        self._center = (.5,.5)
        self._metadata = {}
        self._displayed_path = None
        if self._preview is None and self._batch_preview is None:
            self._images.clear()
            self.before_image_label.configure(image=None, text="Зареждане…")
            self.after_image_label.configure(image=None, text="Зареждане…")
        self.preview_info.configure(text="Зареждане на избраната снимка…")
        self._refresh_gallery()
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._ux_ready and not self._loading:self._update_experience()
        if self._loading or self._closing or not self.files or self._busy in {"batch", "scan"}:
            return
        self._pending_preview = True
        self._quick_preview_pending = True
        if self._busy == "preview":
            self._cancel.set()
        if self._preview_timer:
            self.after_cancel(self._preview_timer)
        self._preview_timer = self.after(400, self._request_preview)

    def _refine_preview(self):
        self._quick_preview_pending = False
        self._request_preview()

    def _request_preview(self) -> None:
        self._preview_timer = None
        if self._closing or not self.files:
            return
        if self._busy:
            if self._busy == "preview":
                self._pending_preview = True
            return
        path = self._file_choices.get(self.selected_var.get())
        if path is None:
            return
        try:
            params = self._params()
            self._commit_current()
        except ValueError as error:
            messagebox.showerror("Настройки", str(error), parent=self)
            return
        draft = not self.exact_preview_var.get() or getattr(self,"_quick_preview_pending",False)
        self._quick_preview_pending = False
        self._pending_preview = False

        def generate(emit, cancel):
            try:
                result = self.engine.preview(path, params, draft=draft, cancel=cancel,
                                             on_stage=lambda message: emit("stage", {"message": message}))
                if not cancel.is_set():
                    emit("preview_result", {"result": result})
            except ProcessingCancelled:
                pass
            except Exception as error:
                emit("preview_error", {"name": path.name, "error": str(error) or type(error).__name__})
            finally:
                emit("job_done", {})

        self._launch("preview", generate)
        self.preview_info.configure(text=f"{path.name} · Обновяване на прегледа…")
        self.status_label.configure(text=f"Преглед: {path.name}")

    def _launch(self, kind: str, target: Callable) -> None:
        if self._busy or self._closing:
            return
        self._job_id += 1
        token = self._job_id
        self._cancel = threading.Event()
        cancel = self._cancel
        self._busy = kind
        self._refresh_enabled()

        def emit(event: str, payload: dict) -> None:
            self.events.put((token, event, payload))

        # Non-daemon: closing waits for a safe boundary, never kills an in-flight save.
        self._worker = threading.Thread(target=target, args=(emit, cancel),
                                         name=f"raw-{kind}", daemon=False)
        self._worker.start()

    def _refresh_enabled(self) -> None:
        for widget in self._locked_widgets:
            widget.configure(state="disabled" if self._busy or self._closing else "normal")
        for widget in self._setting_widgets:
            widget.configure(state="disabled" if self._busy in {"batch", "scan"} or self._closing else "normal")
        self.stop_button.configure(state="normal" if self._busy and not self._closing else "disabled")
        for widget in self._gallery_controls:
            widget.configure(state='disabled' if self._busy in {'batch','scan'} or self._closing else 'normal')
        self.pause_button.configure(state='normal' if self._busy=='batch' and not self._closing else 'disabled')
        self._update_format()
        self._refresh_experience_enabled()

    def _start_batch(self, *, files_override=None, resume=False) -> None:
        if self._busy:
            return
        if not self.files:
            messagebox.showinfo("Няма снимки", "Първо избери RAW файлове или папка.", parent=self)
            return
        output = self.output_var.get().strip()
        if not output:
            messagebox.showinfo("Изходна папка", "Избери папка за готовите снимки.", parent=self)
            return
        directory = Path(output).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            messagebox.showerror("Изходна папка", "Избраният път не е директория.", parent=self)
            return
        try:
            params = self._params()
        except ValueError as error:
            messagebox.showerror("Настройки", str(error), parent=self)
            return
        self._commit_current()
        self._failed_files = []
        self.report_card.grid_remove()
        self.session.add(self.files)
        files = files_override if files_override is not None else [p.path for p in self.session.photos.values() if p.included]
        if not files:
            messagebox.showinfo('Няма включени снимки','Включи снимки от галерията.',parent=self)
            return
        shared = {name:getattr(params,name) for name in EXPORT_FIELDS}
        per_file = {str(path):replace(self.session.photos[str(path)].params,**shared) for path in files}
        self._pause.clear()
        self.pause_button.configure(text='Пауза')
        overwrite = self.overwrite_var.get()
        self._pending_preview = False
        if self._preview_timer:
            self.after_cancel(self._preview_timer)
            self._preview_timer = None
        self.progress.set(0)
        self.progress_label.configure(text=f"0 / {len(files)} снимки · 0%")
        self._log(f"Старт: {len(files)} снимки → {directory} ({params.output_format.upper()}).")
        if overwrite:
            self._log("Включено е презаписване на съществуващи изходни снимки.")

        def process(emit, cancel):
            run_batch(self.engine, files, directory, params, overwrite, cancel, emit,per_file=per_file,pause=self._pause,resume=resume)

        self._launch("batch", process)

    def _stop(self) -> None:
        if self._busy:
            self._cancel.set()
            self._pending_preview = False
            if self._preview_timer:
                self.after_cancel(self._preview_timer)
                self._preview_timer = None
            self.status_label.configure(text="Спиране след текущата операция…")
            self.stop_button.configure(state="disabled")
            self._log("Заявено спиране. Текущата LibRaw/OpenCV операция трябва да приключи.")

    def _poll_events(self) -> None:
        for _ in range(100):
            try:
                token, event, data = self.events.get_nowait()
            except Empty:
                break
            if token not in {-1,self._job_id}:
                continue
            self._handle_event(event, data)
        if self._closing and not (self._worker and self._worker.is_alive()) and not (self._autosave_worker and self._autosave_worker.is_alive()):
            self.destroy()
            return
        self._poll_id = self.after(70, self._poll_events)

    def _handle_event(self, event: str, data: dict) -> None:
        if event == "autosaved":
            self._saved_snapshot = data["snapshot"]
            self.autosave_label.configure(text=f"Автоматично запазено · {datetime.now():%H:%M:%S}")
            if self._autosave_next:
                snapshot,self._autosave_next = self._autosave_next,None
                self._write_recovery_snapshot(snapshot)
            return
        if event == "autosave_error":
            self._log("Автоматичното запазване не успя: "+data["error"],"error")
            return
        if event == 'thumbnail':
            if data['epoch']==self._thumb_epoch and data['path'] in self._gallery_buttons:
                image = ctk.CTkImage(light_image=data['image'],dark_image=data['image'],size=data['image'].size)
                button = self._gallery_buttons[data['path']]
                button.configure(image=image)
                button._thumbnail = image
                self._thumb_cache[data['path']] = data['image']
                while len(self._thumb_cache)>200: self._thumb_cache.pop(next(iter(self._thumb_cache)))
            return
        if event in {'watch_files','watch_error'} and (self._watch_cancel.is_set() or data.get('generation')!=self._watch_generation): return
        if event == 'watch_files':
            self._watch_pending = unique_inputs([*self._watch_pending,*data['files']])
            self._log(f"Нови стабилни RAW файлове: {len(data['files'])}.")
            return
        if event == 'watch_error':
            self._log('Наблюдавана папка: '+data['error'],'error')
            return
        if event == "stage":
            if not self._cancel.is_set():
                self.status_label.configure(text=data["message"])
        elif event == "scan_result":
            self._add_files(data["files"])
            for error in data["errors"]:
                self._log(f"Недостъпна папка: {error}", "error")
            if not data["files"]:
                self._log("В папката не са намерени RAW файлове.")
        elif event == "scan_cancelled":
            self._log("Търсенето е спряно.")
        elif event == "preview_result":
            # A changed slider can cancel a result already in the Queue.
            if not self._cancel.is_set():
                self._preview = data["result"]
                self._displayed_path = self._active_path
                self._batch_preview = None
                self._render_preview()
                result = self._preview
                self._metadata = result.metadata or {}
                self._draw_histogram(result.histogram)
                self.metadata_box.configure(state='normal')
                self.metadata_box.delete('1.0','end')
                self.metadata_box.insert('end','\n'.join(f'{k}: {v}' for k,v in self._metadata.items()) or 'Няма достъпни EXIF данни.')
                self.metadata_box.configure(state='disabled')
                if result.draft and self.exact_preview_var.get() and not self._pending_preview:
                    self._preview_timer = self.after(600,self._refine_preview)
                mode = "Бърз, приблизителен преглед" if result.draft else "Обработен в пълна резолюция"
                self.preview_info.configure(text=f"{result.name} · {result.dimensions[0]} × {result.dimensions[1]} · {mode}")
        elif event == "preview_error":
            self._log(f"Преглед {data['name']}: {data['error']}", "error")
        elif event == "file_started":
            self.status_label.configure(text=f"{data['index']} / {data['total']} · {data['name']}")
            self._log(f"Обработка: {data['name']}")
        elif event == "file_saved":
            result = data["result"]
            self._log(f"Готово: {result.output_path.name} · {result.dimensions[0]}×{result.dimensions[1]} · {result.seconds:.1f} s", "success")
            if result.preview is not None:
                # Never display the old photo as the 'before' of a new batch file.
                self._preview = None
                self._batch_preview = result.preview
                self.before_image_label.configure(image=None, text="Прегледът преди корекциите\nсе зарежда от избора на снимка")
                self.compare_var.set("След")
                self._render_preview()
                self.preview_info.configure(text=f"Запазено: {result.output_path.name} · {result.dimensions[0]} × {result.dimensions[1]}")
        elif event == "file_error":
            self._log(f"Грешка {data['name']}: {data['error']}", "error")
            if data.get("source"): self._failed_files.append(Path(data["source"]))
        elif event == "fatal_error":
            self._log(f"Обработката не може да продължи: {data['error']}", "error")
        elif event == "progress":
            ratio = data["completed"] / max(1, data["total"])
            self.progress.set(ratio)
            self.progress_label.configure(text=f"{data['completed']} / {data['total']} снимки · {ratio:.0%} · грешки: {data['failed']}")
        elif event == "batch_done":
            summary = data["summary"]
            stopped = summary.cancelled or summary.completed < summary.total
            title = "Спряно" if stopped else "Завършено"
            status = f"{title} · готови: {summary.succeeded} · грешки: {summary.failed} · {summary.seconds:.1f} s"
            self.status_label.configure(text=status)
            self._log(status, "success" if not stopped and not summary.failed else "")
            self._show_report(summary)
            self._finish_job(preserve_status=True)
        elif event == "job_done":
            self._finish_job()

    def _finish_job(self, preserve_status: bool = False) -> None:
        previous_kind = self._busy
        self._busy = None
        if previous_kind=='batch': self._watch_exporting = False
        self._refresh_enabled()
        if not preserve_status and not self._closing:
            self.status_label.configure(text="Спряно" if self._cancel.is_set() else "Готов за работа")
        self._update_experience()
        if not self._closing and self._queued_drop:
            paths,self._queued_drop = self._queued_drop,[]
            self._import_paths(paths)
            return
        if not self._closing and self._export_pending:
            self._export_pending = False
            override,self._export_override = self._export_override,None
            self._request_export(files_override=override)
            return
        if not self._closing and self.files and self._pending_preview:
            self._schedule_preview()
        elif previous_kind == "scan" and self.files and not self._cancel.is_set():
            self._schedule_preview()

    def _preview_resized(self, _event=None) -> None:
        if self._closing:
            return
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        self._resize_timer = self.after(100, self._render_preview)

    def _display_image(self, label, pil_image: Image.Image) -> None:
        scale = label._get_widget_scaling()
        available_w = max(40,label.winfo_width()/scale-16)
        available_h = max(40,label.winfo_height()/scale-16)
        ratio = min(available_w/pil_image.width,available_h/pil_image.height,1.) if self._zoom is None else self._zoom/scale
        view_w,view_h = min(pil_image.width,available_w/ratio),min(pil_image.height,available_h/ratio)
        left = max(0,min(pil_image.width-view_w,self._center[0]*pil_image.width-view_w/2))
        top = max(0,min(pil_image.height-view_h,self._center[1]*pil_image.height-view_h/2))
        box = (int(left),int(top),max(int(left)+1,round(left+view_w)),max(int(top)+1,round(top+view_h)))
        tile = pil_image.crop(box)
        if self.clip_warning_var.get():
            array = np.array(tile)
            high = array.max(axis=2)==255
            low = array.min(axis=2)==0
            array[low] = [40,100,255]
            array[high] = [255,40,40]
            tile = Image.fromarray(array)
        if self._stroke:
            draw = ImageDraw.Draw(tile)
            points = [(x*pil_image.width-box[0],y*pil_image.height-box[1]) for x,y in self._stroke]
            if self.tool_var.get()=='Изрязване' and len(points)>1:
                draw.rectangle((min(points[0][0],points[-1][0]),min(points[0][1],points[-1][1]),max(points[0][0],points[-1][0]),max(points[0][1],points[-1][1])),outline='#20b99b',width=max(1,round(2/ratio)))
            elif len(points)>1:
                draw.line(points,fill='#20b99b',width=max(1,round(2/ratio)))
        size = (max(1,round(tile.width*ratio)),max(1,round(tile.height*ratio)))
        self._view_rects[label] = (box,size,pil_image.size)
        image = ctk.CTkImage(light_image=tile,dark_image=tile,size=size)
        self._images.append(image)
        label.configure(image=image,text='')

    def _render_preview(self) -> None:
        if self.compare_var.get()=="Плъзгач":
            self.split_slider.pack(side="right",padx=4);self.split_hint.pack(side="right")
        else:
            self.split_slider.pack_forget();self.split_hint.pack_forget()
        self._resize_timer = None
        if self._closing:
            return
        mode = self.compare_var.get()
        if mode == "Преди":
            self.after_panel.grid_remove()
            self.before_panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        elif mode in {"След","Плъзгач"}:
            self.before_panel.grid_remove()
            self.after_panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        else:
            self.before_panel.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=10, pady=10)
            self.after_panel.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=10, pady=10)
        if self._preview:
            self._images.clear()
            # Layout is updated before measuring the image boxes.
            self.update_idletasks()
            if mode != "След":
                self._display_image(self.before_image_label, self._preview.before)
            if mode != "Преди":
                shown = self._preview.after
                if mode=='Плъзгач':
                    shown = shown.copy()
                    split = round(shown.width*self.split_slider.get())
                    shown.paste(self._preview.before.crop((0,0,split,shown.height)),(0,0))
                    ImageDraw.Draw(shown).line((split,0,split,shown.height),fill='#20b99b',width=2)
                self._display_image(self.after_image_label,shown)
        elif self._batch_preview is not None and mode != "Преди":
            self._images.clear()
            self.update_idletasks()
            self._display_image(self.after_image_label, self._batch_preview)

    def _open_output(self) -> None:
        path = Path(self.output_var.get().strip()).expanduser()
        if not self.output_var.get().strip() or not path.is_dir():
            messagebox.showinfo("Резултати", "Изходната папка още не съществува.", parent=self)
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as error:
            messagebox.showerror("Отваряне на папка", str(error), parent=self)

    def _on_close(self) -> None:
        if self._closing:
            return
        self._save_recovery()
        self._save_preferences()
        self._closing = True
        if self._autosave_timer: self.after_cancel(self._autosave_timer)
        self._pending_preview = False
        self._cancel.set()
        self._watch_cancel.set()
        self._thumb_cancel.set()
        for timer in (self._watch_timer,self._gallery_timer):
            if timer: self.after_cancel(timer)
        for timer in (self._preview_timer, self._resize_timer):
            if timer:
                self.after_cancel(timer)
        self._preview_timer = self._resize_timer = None
        if (self._worker and self._worker.is_alive()) or (self._autosave_worker and self._autosave_worker.is_alive()):
            self.status_label.configure(text="Затваряне след текущата операция…")
            self._refresh_enabled()
        else:
            self.after_cancel(self._poll_id)
            self.destroy()

    def _commit_current(self):
        if self._loading:
            return
        params = self._params()
        if self._active_path in self.session.photos:
            photo = self.session.photos[self._active_path]
            changed = photo.params != params
            photo.set_params(params)
            self._project_dirty |= changed
            if changed and self._active_path in self._gallery_buttons:
                self._gallery_buttons[self._active_path].configure(text=photo.path.name[:18]+(" •" if photo.cursor else ""))
        self.session.defaults = params
        self.session.output = self.output_var.get()
        self.session.selected = self._active_path or ''

    def _load_params(self, params):
        self._loading = True
        try:
            self._base_edit = params
            self.wb_var.set(next(k for k,v in WB_LABELS.items() if v==params.white_balance))
            for variable,field in [(self.auto_exposure_var,'auto_exposure'),(self.tone_var,'tone_mapping'),
                    (self.denoise_var,'denoise'),(self.sharpen_var,'sharpen'),(self.monochrome_var,'monochrome'),
                    (self.lens_var,'lens_enabled'),(self.auto_lens_var,'auto_lens'),(self.preserve_exif_var,'preserve_exif'),
                    (self.exif_camera_var,'exif_camera'),(self.exif_date_var,'exif_date'),(self.exif_exposure_var,'exif_exposure')]:
                variable.set(getattr(params,field))
            self.format_var.set(params.output_format.upper())
            self.depth_var.set('16 бита' if params.png_bit_depth==16 else '8 бита')
            self.template_var.set(params.filename_template)
            self.max_edge_var.set(str(params.max_edge))
            sliders = dict(self._extra_sliders)
            sliders.update(exposure_ev=self.exposure_slider,tone_strength=self.tone_slider,clahe_clip=self.clip_slider,
                           denoise_strength=self.noise_slider,sharpen_amount=self.sharp_slider,quality=self.quality_slider,png_compression=self.compression_slider)
            for key,slider in sliders.items():
                value = getattr(params,key)
                steps = slider.cget('number_of_steps')
                slider.configure(number_of_steps=None)
                slider.set(value)
                slider.configure(number_of_steps=steps)
                label,formatter = self._slider_labels[slider]
                label.configure(text=formatter(value))
            self.mask_label.configure(text=f'{len(params.masks)} локални маски')
            self._update_format()
        finally:
            self._loading = False
            self._sync_simple_controls()

    def _replace_edit(self, params):
        if self._busy in {'batch','scan'}:
            return
        self._commit_current()
        if self._active_path in self.session.photos:
            self.session.photos[self._active_path].set_params(params)
        self._load_params(params)
        self._project_dirty = True
        self._schedule_preview()

    def _edit_change(self, **changes):
        try:
            self._replace_edit(replace(self._params(),**changes))
        except (ValueError,TypeError) as error:
            messagebox.showerror('Редакция',str(error),parent=self)

    def _apply_builtin(self, name):
        current = self._params()
        preset = ProcessingParams(**BUILTIN_PRESETS[name])
        self._replace_edit(replace(current,**{k:getattr(preset,k) for k in EDIT_FIELDS}))

    def _save_preset(self):
        path = filedialog.asksaveasfilename(parent=self,title='Запази preset',defaultextension='.json',filetypes=[('RAW Studio preset','*.json')])
        if path:
            try:
                save_preset(path,self._params())
                self._log('Запазен preset: '+path)
            except (OSError,ValueError) as error: messagebox.showerror('Preset',str(error),parent=self)

    def _load_preset(self):
        path = filedialog.askopenfilename(parent=self,title='Зареди preset',filetypes=[('RAW Studio preset','*.json')])
        if path:
            try: self._replace_edit(load_preset(path,self._params()))
            except (OSError,ValueError,TypeError,KeyError) as error: messagebox.showerror('Preset',str(error),parent=self)

    def _copy_settings(self):
        self._commit_current()
        p = self._params()
        self._clipboard = {k:getattr(p,k) for k in COPY_GROUPS[self.copy_group_var.get()]}
        self._log('Копирани настройки: '+self.copy_group_var.get())

    def _paste_settings(self):
        if not self._clipboard:
            self._log('Първо копирай настройки от снимка.')
            return
        self._commit_current()
        count = 0
        for photo in self.session.photos.values():
            if photo.included:
                photo.set_params(replace(photo.params,**self._clipboard))
                count += 1
        if self._active_path in self.session.photos:
            self._load_params(self.session.photos[self._active_path].params)
        self._project_dirty = True
        self._schedule_preview()
        self._log(f'Настройки приложени към {count} снимки. Изрязването и маските са индивидуални.')

    def _undo(self):
        if self._busy in {'batch','scan'}: return
        self._commit_current()
        if self._active_path in self.session.photos:
            self._load_params(self.session.photos[self._active_path].undo())
            self._project_dirty = True
            self._schedule_preview()

    def _redo(self):
        if self._busy in {'batch','scan'}: return
        self._commit_current()
        if self._active_path in self.session.photos:
            self._load_params(self.session.photos[self._active_path].redo())
            self._project_dirty = True
            self._schedule_preview()

    def _save_project(self):
        self._commit_current()
        path = filedialog.asksaveasfilename(parent=self,title='Запази проект',defaultextension='.rawstudio',filetypes=[('RAW Studio project','*.rawstudio')])
        if path:
            try:
                self.session.save(path)
                self._project_dirty = False
                self._remember_project(path)
                self.project_label.configure(text=Path(path).name)
                self._log('Запазен проект: '+path)
            except (OSError,ValueError) as error: messagebox.showerror('Проект',str(error),parent=self)

    def _load_project(self):
        path = filedialog.askopenfilename(parent=self,title='Отвори проект',filetypes=[('RAW Studio project','*.rawstudio')])
        if not path: return
        try:
            session = EditSession.load(path)
        except (OSError,ValueError,TypeError,KeyError) as error:
            messagebox.showerror('Проект',str(error),parent=self)
            return
        self._clear_files()
        self.session = session
        self.output_var.set(session.output)
        self._load_params(session.defaults)
        self._add_files([photo.path for photo in session.photos.values()])
        selected = next((key for key,value in self._file_choices.items() if str(value)==session.selected),None)
        if selected:
            self.selected_var.set(selected)
            self._active_path = None
            self._selection_changed()
        self._project_dirty = False
        self.project_label.configure(text=Path(path).name)
        missing = [p.path.name for p in session.photos.values() if not p.path.is_file()]
        if missing: self._log('Липсващи оригинали: '+', '.join(missing),'error')
        self._remember_project(path)
        self._log('Проектът е възстановен, включително историята и локалните маски.')

    def _save_lens(self):
        if not self._metadata.get('Image Model') or not self._metadata.get('EXIF LensModel'):
            self._log('Липсват модел на камера/обектив. Профилът ще може да се зарежда ръчно.')
        folder = Path.home()/'.raw-studio'/'lenses'
        folder.mkdir(parents=True,exist_ok=True)
        path = filedialog.asksaveasfilename(parent=self,title='Запази собствен профил за обектив',initialdir=folder,defaultextension='.json',filetypes=[('Lens profile','*.json')])
        if path:
            try:
                save_lens_profile(path,self._params(),self._metadata)
                self._log('Запазен профил за обектив: '+path)
            except (OSError,ValueError) as error: messagebox.showerror('Обектив',str(error),parent=self)

    def _load_lens(self):
        path = filedialog.askopenfilename(parent=self,title='Зареди профил за обектив',filetypes=[('Lens profile','*.json')])
        if path:
            try: self._replace_edit(load_lens_profile(path,self._params()))
            except (OSError,ValueError,TypeError,KeyError) as error: messagebox.showerror('Обектив',str(error),parent=self)

    def _rotate(self):
        self._edit_change(rotation=(self._params().rotation+1)%4)
        self._center = (.5,.5)

    def _set_tool(self, value):
        if self._busy=='batch': return
        self.tool_var.set(value)
        self.compare_var.set('След' if value in {'Четка','Градиент','Изрязване'} else 'Преди' if value=='Пипетка' else self.compare_var.get())
        self._stroke = []
        self._render_preview()
        hints = {'Изрязване':'Влачи върху кадъра за изрязване.', 'Пипетка':'Щракни върху неутрално сива област в прегледа ПРЕДИ.',
                 'Четка':'Рисувай с ляв бутон; всяко движение добавя локална маска.',
                 'Градиент':'Влачи от 0% към 100% локална корекция.','Местене':'Колелце: увеличение. Ляв бутон: местене на увеличената снимка.'}
        cursor = 'crosshair' if value!='Местене' else 'hand2'
        for label in [self.before_image_label,self.after_image_label]: label._label.configure(cursor=cursor)
        self.tool_hint.configure(text=hints[value])
        self._log(hints[value])

    def _set_zoom(self, value):
        self._zoom = value
        self.zoom_label.configure(text='Побери' if value is None else f'{value:.0%}')
        if value is not None and value>=1 and self._preview and self._preview.draft:
            self.exact_preview_var.set(True)
            self._schedule_preview()
        self._render_preview()

    def _zoom_step(self, multiplier):
        self._set_zoom(min(4.,max(.125,(self._zoom or .5)*multiplier)))

    def _mouse_wheel(self,event,label):
        if self._busy=='batch': return
        up = getattr(event,'delta',0)>0 or getattr(event,'num',None)==4
        self._zoom_step(1.25 if up else .8)
        return 'break'

    def _point(self,event,label,clamp=False):
        if label not in self._view_rects: return None
        box,size,full = self._view_rects[label]
        scale = label._get_widget_scaling()
        x = event.x_root-label.winfo_rootx()
        y = event.y_root-label.winfo_rooty()
        x -= (label.winfo_width()-size[0]*scale)/2
        y -= (label.winfo_height()-size[1]*scale)/2
        if not clamp and not (0<=x<=size[0]*scale and 0<=y<=size[1]*scale): return None
        u = (box[0]+x/(size[0]*scale)*(box[2]-box[0]))/full[0]
        v = (box[1]+y/(size[1]*scale)*(box[3]-box[1]))/full[1]
        return max(0,min(1,u)),max(0,min(1,v))

    def _mouse_down(self,event,label):
        if not self._preview or self._busy in {'batch','scan'} or self._displayed_path != self._active_path: return
        point = self._point(event,label)
        if point is None: return
        self._drag_origin = point
        self._mouse_point = point
        self._last_drag_root = (event.x_root,event.y_root)
        tool = self.tool_var.get()
        if tool=='Пипетка':
            image = np.asarray(self._preview.before).astype(np.float32)/255
            x,y = round(point[0]*(image.shape[1]-1)),round(point[1]*(image.shape[0]-1))
            patch = image[max(0,y-3):y+4,max(0,x-3):x+4]
            linear = np.where(patch<=.04045,patch/12.92,((patch+.055)/1.055)**2.4).mean(axis=(0,1))
            if linear.min()<.005 or patch.max()>=.99:
                self._log('Избери сива област без прекалено тъмни или изгорели пиксели.')
                return
            gains = np.clip(linear.mean()/linear,.25,4)
            self._edit_change(wb_gains=tuple(float(v) for v in gains),temperature=0,tint=0,white_balance='camera' if self._params().white_balance in {'gray_world','shades_of_gray'} else self._params().white_balance)
            self._log('Балансът е зададен от избраната област.')
        elif tool!='Местене':
            self._stroke = [point]

    def _mouse_drag(self,event,label):
        if self._drag_origin is None or not self._preview or self._busy in {'batch','scan'}: return
        point = self._point(event,label,True)
        if point is None: return
        tool = self.tool_var.get()
        if tool=='Местене' and self._zoom is not None:
            # Root coordinates remain stable even after moving the displayed image.
            if not hasattr(self,'_last_drag_root') or self._last_drag_root is None:
                self._last_drag_root = (event.x_root,event.y_root)
            dx,dy = event.x_root-self._last_drag_root[0],event.y_root-self._last_drag_root[1]
            full = self._preview.after.size
            self._center = (max(0,min(1,self._center[0]-dx/(full[0]*self._zoom))),max(0,min(1,self._center[1]-dy/(full[1]*self._zoom))))
            self._last_drag_root = (event.x_root,event.y_root)
        elif tool in {'Изрязване','Градиент'}:
            self._stroke = [self._drag_origin,point]
        elif tool=='Четка' and len(self._stroke)<5000:
            if math.dist(point,self._stroke[-1])>.002: self._stroke.append(point)
        self._render_preview()

    def _mouse_up(self,event,label):
        if not self._preview or self._busy in {'batch','scan'} or self._displayed_path != self._active_path: return
        points = tuple(self._stroke)
        self._stroke = []
        self._drag_origin = None
        self._last_drag_root = None
        tool = self.tool_var.get()
        if tool=='Изрязване' and len(points)==2:
            x0,x1 = sorted((points[0][0],points[1][0]))
            y0,y1 = sorted((points[0][1],points[1][1]))
            if x1-x0>.005 and y1-y0>.005:
                if self.ratio_var.get()!='Свободно':
                    a,b = map(float,self.ratio_var.get().split(':'))
                    ratio = a/b*self._preview.after.height/self._preview.after.width
                    if (x1-x0)/(y1-y0)>ratio: x1=x0+(y1-y0)*ratio
                    else: y1=y0+(x1-x0)/ratio
                old = self._params().crop
                self._edit_change(crop=(old[0]+x0*(old[2]-old[0]),old[1]+y0*(old[3]-old[1]),old[0]+x1*(old[2]-old[0]),old[1]+y1*(old[3]-old[1])))
        elif tool in {'Четка','Градиент'} and points:
            if tool=='Градиент' and (len(points)!=2 or math.dist(*points)<.005): return
            mask = LocalAdjustment(kind='brush' if tool=='Четка' else 'gradient',points=points,radius=self.local_radius.get(),exposure=self.local_ev.get(),shadows=self.local_shadows.get(),saturation=self.local_saturation.get())
            self._edit_change(masks=(*self._params().masks,mask))
        self._render_preview()

    def _draw_histogram(self,data):
        self.hist_canvas.delete('all')
        if not data: return
        w = max(250,self.hist_canvas.winfo_width())
        h = 90
        for counts,color in [(data['luma'],'#a3adbd'),*zip(data['rgb'],['#ee7171','#5fc98b','#71a6ee'])]:
            maximum = max(max(counts),1)
            points = [coordinate for index,value in enumerate(counts) for coordinate in (index/255*w,h-4-(h-10)*math.log1p(value)/math.log1p(maximum))]
            self.hist_canvas.create_line(*points,fill=color,width=1)
        self.hist_label.configure(text=f"Черно: {data['shadows']:.1%} · Светлини: {data['highlights']:.1%}")

    def _build_gallery(self,parent):
        self._build_experience_gallery(parent)

    def _gallery_step(self, delta):
        self._gallery_page = max(0,self._gallery_page+delta)
        self._refresh_gallery()

    def _refresh_gallery(self,reset=False):
        page = self._populate_gallery(reset)
        epoch = self._thumb_epoch
        paths = [p.path for p in page if str(p.path) not in self._thumb_cache]
        if self._gallery_timer: self.after_cancel(self._gallery_timer)
        self._gallery_timer = self.after(800,lambda:self._start_thumbnails(paths,epoch))

    def _start_thumbnails(self,paths,epoch):
        self._gallery_timer = None
        if self._closing or epoch!=self._thumb_epoch: return
        if self._busy or (self._thumb_worker and self._thumb_worker.is_alive()):
            self._gallery_timer = self.after(500,lambda:self._start_thumbnails(paths,epoch))
            return
        self._thumb_cancel = threading.Event()
        cancel = self._thumb_cancel
        def generate():
            for path in paths:
                if cancel.is_set(): break
                try:
                    with rawpy.imread(str(path)) as raw:
                        try:
                            thumb = raw.extract_thumb()
                            if thumb.format==rawpy.ThumbFormat.JPEG:
                                with Image.open(io.BytesIO(thumb.data)) as source: image=source.convert('RGB')
                            else: image=Image.fromarray(thumb.data)
                        except rawpy.LibRawNoThumbnailError:
                            image=Image.fromarray(raw.postprocess(half_size=True,use_camera_wb=True))
                    image.thumbnail((84,32),Image.Resampling.LANCZOS)
                    if not cancel.is_set(): self.events.put((-1,'thumbnail',{'epoch':epoch,'path':str(path),'image':image}))
                except Exception:
                    continue
        self._thumb_worker = threading.Thread(target=generate,name='raw-thumbnails',daemon=True)
        self._thumb_worker.start()

    def _select_photo(self,path):
        if self._busy in {'batch','scan'}: return
        key = next((k for k,p in self._file_choices.items() if p==path),None)
        if key and str(path)!=self._active_path:
            self.selected_var.set(key)
            self._selection_changed()

    def _set_included(self, photo, value):
        if self._busy=='batch': return
        photo.included = value
        count = sum(p.included for p in self.session.photos.values())
        self._update_experience()
        self._project_dirty = True
        self._refresh_gallery()

    def _set_rating(self, photo, rating):
        photo.rating = rating
        self._project_dirty = True
        if self.favorite_filter_var.get(): self._refresh_gallery()

    def _bind_editor_keys(self):
        def safe(action):
            def run(event):
                focused = self.focus_get()
                if focused and focused.winfo_class() in {'Entry','Text'}: return
                self._ui_action(action)
                return 'break'
            return run
        for sequence,action in [('<Control-z>',self._undo),('<Control-y>',self._redo),('<Left>',lambda:self._navigate(-1)),('<Right>',lambda:self._navigate(1))]:
            self.bind(sequence,safe(action))
        self.bind('<Control-s>',lambda event:self._ui_action(self._save_project) if not self._busy else None)

    def _navigate(self,direction):
        if self._busy in {'batch','scan'} or not self.files: return
        current = self._file_choices.get(self.selected_var.get())
        index = self.files.index(current) if current in self.files else 0
        self._select_photo(self.files[(index+direction)%len(self.files)])

    def _toggle_pause(self):
        if self._busy!='batch': return
        if self._pause.is_set():
            self._pause.clear()
            self.pause_button.configure(text='Пауза')
            self._log('Серията продължава.')
        else:
            self._pause.set()
            self.pause_button.configure(text='Продължи')
            self.status_label.configure(text='Пауза след текущата операция…')
            self._log('Заявена пауза на серията.')

    def _resume_batch(self):
        path = filedialog.askopenfilename(parent=self,title='Продължи серия',initialfile='raw-studio-batch.json',filetypes=[('Batch journal','*.json')])
        if not path: return
        try:
            data = read_json(path)
            if data.get('version')!=1 or not isinstance(data.get('jobs'),list) or not data['jobs']:
                raise ValueError('Невалиден дневник на серия.')
            files = []
            photos = {}
            for row in data['jobs']:
                source = Path(row['source']).resolve()
                if source.suffix.lower() not in RAW_EXTENSIONS: raise ValueError('Невалиден RAW път.')
                p = params_from_dict(row['params'])
                photos[str(source)] = PhotoState(source,p)
                files.append(source)
            self._clear_files()
            self.session.photos = photos
            self._load_params(photos[str(files[0])].params)
            self.output_var.set(str(Path(path).resolve().parent))
            self.overwrite_var.set(bool(data.get('overwrite',False)))
            self._add_files(files)
            # Start before the deferred automatic preview begins.
            self._start_batch(resume=True)
        except (OSError,ValueError,KeyError,TypeError) as error:
            messagebox.showerror('Продължаване на серия',str(error),parent=self)

    def _start_watch(self):
        self._params()  # Validate before starting an unattended watcher.
        if not self.output_var.get().strip():
            messagebox.showinfo('Автоматичен експорт','Първо избери изходна папка в раздел Експорт.',parent=self)
            return
        folder = filedialog.askdirectory(parent=self,title='Наблюдавай папка за нови RAW снимки')
        if not folder: return
        self._stop_watch()
        watcher = FolderWatcher(folder,self.recursive_var.get())
        self._watch_cancel = threading.Event()
        cancel = self._watch_cancel
        generation = self._watch_generation
        def emit(kind,data): self.events.put((-1,kind,dict(data,generation=generation)))
        self._watch_worker = threading.Thread(target=watcher.run,args=(cancel,emit),name='raw-watch',daemon=True)
        self._watch_worker.start()
        self.watch_label.configure(text='Наблюдавана папка: '+folder)
        self._log('Автоматичен експорт на нови файлове. Съществуващите файлове са пропуснати.')

    def _stop_watch(self):
        self._watch_generation += 1
        self._watch_cancel.set()
        self._watch_pending.clear()
        self.watch_label.configure(text='Наблюдението е изключено.')

    def _drain_watch(self):
        self._watch_timer = None
        if self._closing: return
        if self._watch_pending and not self._busy and not self._watch_cancel.is_set():
            paths,self._watch_pending = self._watch_pending,[]
            try:
                self._add_files(paths)
                self._watch_exporting = True
                self._start_batch(files_override=paths)
            except ValueError as error:
                self._watch_exporting = False
                self._stop_watch()
                self._log('Автоматичният експорт е спрян: '+str(error),'error')
        self._watch_timer = self.after(500,self._drain_watch)


    def _export_edge(self):
        try:
            return int(self.max_edge_var.get().strip() or '0')
        except ValueError as error:
            raise ValueError('Дългата страна трябва да е цяло число (0–16000).') from error

    def _set_edge(self,value):
        self.max_edge_var.set(str(round(value)))
        label,formatter = self._slider_labels[self._extra_sliders['max_edge']]
        label.configure(text=formatter(value))


    def _set_all_included(self,value):
        if self._busy in {'batch','scan'}: return
        for photo in self.session.photos.values(): photo.included = value
        self._project_dirty = True
        self._refresh_gallery()

    def _include_current_only(self):
        if self._busy in {'batch','scan'}: return
        for key,photo in self.session.photos.items(): photo.included = key==self._active_path
        self._project_dirty = True
        self._refresh_gallery()


    def _ui_action(self, action, *args):
        """Keep invalid user edits out of Tk callback tracebacks."""
        try:
            return action(*args)
        except ValueError as error:
            self._log(str(error),'error')
            messagebox.showerror('Настройки',str(error),parent=self)
            return None
