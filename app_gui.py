"""CustomTkinter desktop UI. All Tk operations stay on the main thread.

Workers communicate through Queue; a short after() poll delivers their results.
Settings are captured before spawning a worker, never read from Tk variables there.
"""
from __future__ import annotations

from datetime import datetime
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
from PIL import Image

from batch import discover_raws, run_batch, unique_inputs
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


class AppGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RAW Studio · Python RAW Processor")
        width = min(1380, max(1100, self.winfo_screenwidth() - 90))
        height = min(900, max(720, self.winfo_screenheight() - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(1100, 720)
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
        self.compare_var = tk.StringVar(value="Сравнение")

        self._build_sidebar()
        self._build_main()
        self._update_format()
        self._poll_id = self.after(70, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log("Избери RAW снимки, изходна папка и настройки. Оригиналите се запазват.")

    @staticmethod
    def _label(parent, text: str, **kwargs):
        return ctk.CTkLabel(parent, text=text, anchor="w", **kwargs)

    def _section(self, parent, text: str):
        self._label(parent, text, font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=MUTED).pack(fill="x", padx=12, pady=(20, 8))

    def _button(self, parent, text: str, command: Callable, **kwargs):
        button = ctk.CTkButton(parent, text=text, command=command, height=35,
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
        self._label(row, title, font=ctk.CTkFont(size=12)).pack(side="left")
        value_label = ctk.CTkLabel(row, text=formatter(initial), text_color=MUTED,
                                  font=ctk.CTkFont(size=12))
        value_label.pack(side="right")

        def changed(value: float) -> None:
            value_label.configure(text=formatter(value))
            self._schedule_preview()

        slider = ctk.CTkSlider(parent, from_=low, to=high, number_of_steps=steps,
                              command=changed, progress_color=ACCENT, button_color=ACCENT)
        slider.set(initial)
        slider.pack(fill="x", padx=12, pady=(5, 7))
        self._setting_widgets.append(slider)
        return slider, value_label

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=CARD)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(1, weight=1)
        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 10))
        self._label(brand, "RAW Studio", font=ctk.CTkFont(size=27, weight="bold")).pack(fill="x")
        self._label(brand, "Твоите снимки. В най-добрата им светлина.",
                    font=ctk.CTkFont(size=11), text_color=MUTED).pack(fill="x", pady=(2, 0))

        controls = ctk.CTkScrollableFrame(sidebar, fg_color="transparent", corner_radius=0)
        controls.grid(row=1, column=0, sticky="nsew", padx=8)
        self._section(controls, "01  /  СНИМКИ")
        self._button(controls, "+  Добави RAW файлове", self._choose_files,
                     fg_color=ACCENT, hover_color=("#096759", "#15977e"), text_color="white")
        self._button(controls, "Избери папка със снимки", self._choose_folder)
        self._switch(controls, "Включи подпапките", self.recursive_var,
                     command=lambda: None, setting=False)
        self.file_count_label = self._label(controls, "0 избрани снимки", text_color=MUTED)
        self.file_count_label.pack(fill="x", padx=12, pady=4)
        self._button(controls, "Изчисти списъка", self._clear_files,
                     fg_color=("#e4e9ef", "#2b3643"), text_color=("#26313f", "#d7dfe8"))

        self._section(controls, "02  /  БАЛАНС И СВЕТЛИНА")
        self.wb_menu = ctk.CTkOptionMenu(controls, values=list(WB_LABELS), variable=self.wb_var,
                                        command=lambda _: self._schedule_preview())
        self.wb_menu.pack(fill="x", padx=12, pady=(0, 8))
        self._setting_widgets.append(self.wb_menu)
        self._switch(controls, "Автоматична експозиция", self.auto_exposure_var)
        self.exposure_slider, _ = self._slider(controls, "Експозиция", -3, 3, 0, 60,
                                              lambda v: f"{v:+.1f} EV")
        self._switch(controls, "Локален контраст (CLAHE)", self.tone_var)
        self.tone_slider, _ = self._slider(controls, "Сила на корекцията", 0, 1, .30, 20,
                                           lambda v: f"{v:.0%}")
        self.clip_slider, _ = self._slider(controls, "CLAHE clip limit", .5, 5, 2, 18,
                                           lambda v: f"{v:.2g}")

        self._section(controls, "03  /  ДЕТАЙЛ")
        self._switch(controls, "Премахване на шум", self.denoise_var)
        self.noise_slider, _ = self._slider(controls, "Сила на филтъра", 1, 12, 3, 22,
                                            lambda v: f"{v:.1f}")
        self._switch(controls, "Изостряне", self.sharpen_var)
        self.sharp_slider, _ = self._slider(controls, "Сила на изостряне", 0, 1.5, .45, 30,
                                            lambda v: f"{v:.2f}")

        self._section(controls, "04  /  ЕКСПОРТ")
        self.format_control = ctk.CTkSegmentedButton(controls, values=["JPG", "PNG"],
                                                    variable=self.format_var,
                                                    command=self._update_format,
                                                    selected_color=ACCENT)
        self.format_control.pack(fill="x", padx=12, pady=(0, 5))
        self._setting_widgets.append(self.format_control)
        self.quality_slider, self.quality_label = self._slider(
            controls, "Качество на JPG", 1, 100, 92, 99, lambda v: f"{round(v)}%")
        self.compression_slider, self.compression_label = self._slider(
            controls, "PNG компресия", 0, 9, 4, 9, lambda v: str(round(v)))
        self.depth_menu = ctk.CTkOptionMenu(controls, values=["8 бита", "16 бита"],
                                           variable=self.depth_var)
        self.depth_menu.pack(fill="x", padx=12, pady=7)
        self._setting_widgets.append(self.depth_menu)
        self.format_hint = self._label(controls, "", text_color=MUTED, wraplength=260,
                                       justify="left", font=ctk.CTkFont(size=11))
        self.format_hint.pack(fill="x", padx=12, pady=4)

        self.output_entry = ctk.CTkEntry(controls, textvariable=self.output_var,
                                        placeholder_text="Папка за готовите снимки")
        self.output_entry.pack(fill="x", padx=12, pady=(10, 4))
        self._locked_widgets.append(self.output_entry)
        self._button(controls, "Избери изходна папка", self._choose_output)
        self._switch(controls, "Презаписвай готовите файлове", self.overwrite_var,
                     command=lambda: None)
        self._label(controls, "По подразбиране: ново име при съвпадение.", text_color=MUTED,
                    font=ctk.CTkFont(size=11)).pack(fill="x", padx=12, pady=(2, 16))

        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=20, pady=16)
        self.start_button = ctk.CTkButton(footer, text="Обработи всички снимки", height=43,
                                         fg_color=ACCENT, hover_color=("#096759", "#15977e"),
                                         text_color="white", font=ctk.CTkFont(size=14, weight="bold"),
                                         command=self._start_batch)
        self.start_button.pack(fill="x")
        self._locked_widgets.append(self.start_button)
        self.theme_menu = ctk.CTkOptionMenu(footer, values=["Тъмна тема", "Светла тема", "Системна тема"],
                                           command=self._change_theme, height=28)
        self.theme_menu.set("Тъмна тема")
        self.theme_menu.pack(fill="x", pady=(10, 0))

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=24, pady=22)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self._label(header, "ПРЕДВАРИТЕЛЕН ПРЕГЛЕД", text_color=MUTED,
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self._label(header, "RAW → JPG / PNG", text_color=ACCENT,
                    font=ctk.CTkFont(size=12)).pack(side="right")

        selection = ctk.CTkFrame(main, fg_color="transparent")
        selection.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        selection.grid_columnconfigure(0, weight=1)
        self.file_menu = ctk.CTkOptionMenu(selection, values=["Няма избрани файлове"],
                                          variable=self.selected_var,
                                          command=lambda _: self._selection_changed())
        self.file_menu.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._locked_widgets.append(self.file_menu)
        self.preview_button = ctk.CTkButton(selection, text="Обнови прегледа", width=150,
                                           command=self._request_preview)
        self.preview_button.grid(row=0, column=1)
        self._locked_widgets.append(self.preview_button)
        self.preview_switch = ctk.CTkSwitch(selection, text="Точен преглед (по-бавен)",
                                            variable=self.exact_preview_var,
                                            command=self._schedule_preview,
                                            progress_color=ACCENT)
        self.preview_switch.grid(row=1, column=0, sticky="w", pady=(12, 0))
        self._setting_widgets.append(self.preview_switch)
        self.compare_control = ctk.CTkSegmentedButton(selection, values=["Преди", "След", "Сравнение"],
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
        self.preview_info = self._label(main, "Прегледът е умален; експортът винаги е с пълна резолюция.",
                                        text_color=MUTED, font=ctk.CTkFont(size=11))
        self.preview_info.grid(row=3, column=0, sticky="ew", pady=(8, 14))

        progress_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=10)
        progress_card.grid(row=4, column=0, sticky="ew", pady=(0, 14))
        progress_card.grid_columnconfigure(0, weight=1)
        self.status_label = self._label(progress_card, "Готов за работа", font=ctk.CTkFont(size=13))
        self.status_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 2))
        self.stop_button = ctk.CTkButton(progress_card, text="Спри", width=78, height=30,
                                        state="disabled", fg_color=("#ad3a3a", "#873e48"),
                                        hover_color=("#942f2f", "#70333c"), command=self._stop)
        self.stop_button.grid(row=0, column=1, rowspan=2, padx=16, pady=12)
        self.progress_label = self._label(progress_card, "0 / 0 снимки · 0%", text_color=MUTED,
                                          font=ctk.CTkFont(size=11))
        self.progress_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
        self.progress = ctk.CTkProgressBar(progress_card, progress_color=ACCENT, height=6)
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(5, 14))
        self.progress.set(0)

        log_header = ctk.CTkFrame(main, fg_color="transparent")
        log_header.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        self._label(log_header, "ДНЕВНИК", text_color=MUTED,
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.open_output_button = ctk.CTkButton(log_header, text="Отвори резултатите", width=150,
                                               height=27, command=self._open_output)
        self.open_output_button.pack(side="right")
        self._locked_widgets.append(self.open_output_button)
        self.log_box = ctk.CTkTextbox(main, height=142, fg_color=CARD,
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
                    font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="ew", pady=8)
        label = ctk.CTkLabel(frame, text="", text_color=MUTED, corner_radius=8,
                             fg_color=("#eef1f5", "#131a22"), font=ctk.CTkFont(size=15))
        label.grid(row=1, column=0, sticky="nsew")
        if column == 0:
            self.before_image_label = label
        else:
            self.after_image_label = label
        return frame

    def _params(self) -> ProcessingParams:
        # Main thread only. Capture a fixed snapshot for the whole batch.
        return ProcessingParams(
            white_balance=WB_LABELS[self.wb_var.get()],
            auto_exposure=self.auto_exposure_var.get(),
            exposure_ev=round(self.exposure_slider.get(), 2),
            tone_mapping=self.tone_var.get(), clahe_clip=self.clip_slider.get(),
            tone_strength=self.tone_slider.get(), denoise=self.denoise_var.get(),
            denoise_strength=self.noise_slider.get(), sharpen=self.sharpen_var.get(),
            sharpen_amount=self.sharp_slider.get(), output_format=self.format_var.get().lower(),
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
        if self.files:
            self._pending_preview = True
            self._schedule_preview()

    def _clear_files(self) -> None:
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
        self.preview_info.configure(text="Прегледът е умален; експортът винаги е с пълна резолюция.")
        self._log("Списъкът е изчистен.")

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Избери изходна папка")
        if folder:
            self.output_var.set(folder)

    def _selection_changed(self) -> None:
        self._preview = None
        self._batch_preview = None
        self._images.clear()
        self.before_image_label.configure(image=None, text="Зареждане…")
        self.after_image_label.configure(image=None, text="Зареждане…")
        self.preview_info.configure(text="Зареждане на избраната снимка…")
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._closing or not self.files or self._busy in {"batch", "scan"}:
            return
        self._pending_preview = True
        if self._busy == "preview":
            self._cancel.set()
        if self._preview_timer:
            self.after_cancel(self._preview_timer)
        self._preview_timer = self.after(400, self._request_preview)

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
        except ValueError as error:
            messagebox.showerror("Настройки", str(error), parent=self)
            return
        draft = not self.exact_preview_var.get()
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
        self._update_format()

    def _start_batch(self) -> None:
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
        files = self.files.copy()
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
            run_batch(self.engine, files, directory, params, overwrite, cancel, emit)

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
            if token != self._job_id:
                continue
            self._handle_event(event, data)
        if self._closing and not (self._worker and self._worker.is_alive()):
            self.destroy()
            return
        self._poll_id = self.after(70, self._poll_events)

    def _handle_event(self, event: str, data: dict) -> None:
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
                self._batch_preview = None
                self._render_preview()
                result = self._preview
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
            self._finish_job(preserve_status=True)
        elif event == "job_done":
            self._finish_job()

    def _finish_job(self, preserve_status: bool = False) -> None:
        previous_kind = self._busy
        self._busy = None
        self._refresh_enabled()
        if not preserve_status and not self._closing:
            self.status_label.configure(text="Спряно" if self._cancel.is_set() else "Готов за работа")
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
        available_w = max(40, label.winfo_width() - 16)
        available_h = max(40, label.winfo_height() - 16)
        ratio = min(available_w / pil_image.width, available_h / pil_image.height, 1.0)
        size = (max(1, round(pil_image.width * ratio)), max(1, round(pil_image.height * ratio)))
        image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size)
        self._images.append(image)
        label.configure(image=image, text="")

    def _render_preview(self) -> None:
        self._resize_timer = None
        if self._closing:
            return
        mode = self.compare_var.get()
        if mode == "Преди":
            self.after_panel.grid_remove()
            self.before_panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        elif mode == "След":
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
                self._display_image(self.after_image_label, self._preview.after)
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
        self._closing = True
        self._pending_preview = False
        self._cancel.set()
        for timer in (self._preview_timer, self._resize_timer):
            if timer:
                self.after_cancel(timer)
        self._preview_timer = self._resize_timer = None
        if self._worker and self._worker.is_alive():
            self.status_label.configure(text="Затваряне след текущата операция…")
            self._refresh_enabled()
        else:
            self.after_cancel(self._poll_id)
            self.destroy()
