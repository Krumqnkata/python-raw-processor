"""RAW Studio: complete standalone application. Generated from the modular project."""
from __future__ import annotations

# ---------- raw_engine.py ----------
"""RAW decoding, high precision corrections and atomic JPG/PNG export.

This module does not know about Tkinter. Arrays are RGB, never BGR, until
the explicit OpenCV export/denoising boundary.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import rawpy
from PIL import Image

RAW_EXTENSIONS = frozenset({
    ".3fr", ".arw", ".cr2", ".cr3", ".crw", ".dng", ".erf", ".fff",
    ".iiq", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf",
    ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
})
StageCallback = Callable[[str], None]


class ProcessingCancelled(Exception):
    """Cooperative cancellation at a safe processing boundary."""


@dataclass(frozen=True)
class ProcessingParams:
    white_balance: str = "camera"
    auto_exposure: bool = True
    exposure_ev: float = 0.0
    tone_mapping: bool = True
    clahe_clip: float = 2.0
    tone_strength: float = 0.30
    denoise: bool = False
    denoise_strength: float = 3.0
    sharpen: bool = True
    sharpen_amount: float = 0.45
    output_format: str = "jpg"
    quality: int = 92
    png_compression: int = 4
    png_bit_depth: int = 16

    def __post_init__(self) -> None:
        if self.white_balance not in {"camera", "auto", "gray_world", "shades_of_gray"}:
            raise ValueError("Непознат метод за баланс на бялото.")
        if self.output_format not in {"jpg", "png"}:
            raise ValueError("Изходният формат трябва да е jpg или png.")
        if self.png_bit_depth not in {8, 16}:
            raise ValueError("PNG трябва да е 8 или 16 бита.")
        for name, low, high in [
            ("exposure_ev", -3.0, 3.0), ("clahe_clip", 0.1, 8.0),
            ("tone_strength", 0.0, 1.0), ("denoise_strength", 0.0, 15.0),
            ("sharpen_amount", 0.0, 2.0), ("quality", 1, 100),
            ("png_compression", 0, 9),
        ]:
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Невалидна стойност за {name}: {value}")
        if type(self.quality) is not int or type(self.png_compression) is not int:
            raise ValueError("Качеството и PNG компресията трябва да са цели числа.")


@dataclass
class PreviewResult:
    before: Image.Image
    after: Image.Image
    name: str
    dimensions: tuple[int, int]
    draft: bool = True


@dataclass
class ProcessResult:
    output_path: Path
    dimensions: tuple[int, int]
    seconds: float
    preview: Image.Image | None = None


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ProcessingCancelled()


class RawEngine:
    """Decode linear 16-bit sRGB; correct in float32; quantize only on export.

    CLAHE uses a 16-bit L channel. fastNlMeansDenoisingColored only supports
    8-bit input, so its denoised residual is added back to the float image.
    A PNG16 therefore retains the original sub-8-bit signal elsewhere.
    """

    @staticmethod
    def _stage(message: str, callback: StageCallback | None,
               cancel: threading.Event | None) -> None:
        check_cancel(cancel)
        if callback:
            callback(message)

    def decode(self, input_path: str | Path, params: ProcessingParams,
               *, draft: bool = False, cancel: threading.Event | None = None,
               on_stage: StageCallback | None = None) -> np.ndarray:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Файлът не съществува: {path}")
        if path.suffix.lower() not in RAW_EXTENSIONS:
            raise ValueError(f"Неподдържано RAW разширение: {path.suffix}")
        self._stage("Декодиране на RAW…", on_stage, cancel)
        # Let LibRaw read EXIF orientation and apply the camera colour matrix.
        # Camera WB is also a sensible starting point for the custom algorithms.
        with rawpy.imread(str(path)) as raw:
            camera_wb = params.white_balance != "auto"
            if camera_wb:
                wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)
                if len(wb) < 3 or not np.all(np.isfinite(wb[:3])) or np.any(wb[:3] <= 0):
                    camera_wb = False
                    self._stage("Липсва валиден баланс от камерата; използвам автоматичен.",
                                on_stage, cancel)
            rgb16 = raw.postprocess(
                output_bps=16, output_color=rawpy.ColorSpace.sRGB,
                gamma=(1.0, 1.0), no_auto_bright=True,
                use_camera_wb=camera_wb, use_auto_wb=not camera_wb,
                half_size=draft, highlight_mode=rawpy.HighlightMode.Blend,
            )
        check_cancel(cancel)
        if rgb16.dtype != np.uint16 or rgb16.ndim != 3 or rgb16.shape[2] != 3:
            raise ValueError("RAW декодерът не върна 16-битово RGB изображение.")
        if draft:
            # Preview is intentionally approximate; never reuse it for export.
            h, w = rgb16.shape[:2]
            scale = min(1.0, 1400.0 / max(h, w))
            if scale < 1:
                rgb16 = cv2.resize(rgb16, (max(1, round(w * scale)), max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
        return rgb16

    @staticmethod
    def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
        rgb = np.clip(rgb, 0, 1)
        return np.where(rgb <= 0.0031308, rgb * 12.92,
                        1.055 * np.power(rgb, 1 / 2.4) - 0.055).astype(np.float32)

    @staticmethod
    def to_uint8(rgb: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def to_uint16(rgb: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(rgb, 0, 1) * 65535).astype(np.uint16)

    @staticmethod
    def _sample(rgb: np.ndarray) -> np.ndarray:
        # Bound statistical work and temporary arrays on very large photos.
        step = max(1, math.ceil(math.sqrt(rgb.shape[0] * rgb.shape[1] / 250_000)))
        return rgb[::step, ::step]

    def _white_balance(self, rgb: np.ndarray, method: str) -> np.ndarray:
        samples = self._sample(rgb).reshape(-1, 3)
        usable = (samples.min(axis=1) > 0.01) & (samples.max(axis=1) < 0.98)
        samples = samples[usable]
        if len(samples) < 16:
            return rgb
        power = 6 if method == "shades_of_gray" else 1
        estimates = np.mean(samples ** power, axis=0) ** (1 / power)
        target = estimates.mean()
        gains = np.clip(target / np.maximum(estimates, 1e-6), 0.25, 4.0)
        return np.clip(rgb * gains.astype(np.float32), 0, 1)

    def _exposure(self, rgb: np.ndarray, ev: float, auto: bool) -> np.ndarray:
        gain = 1.0
        if auto:
            sample = self._sample(rgb)
            luminance = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            positive = luminance[luminance > 1e-5]
            if positive.size >= 16:
                # A conservative highlight-based adjustment, limited to +/- 2 EV.
                gain = float(np.clip(0.80 / max(float(np.percentile(positive, 98)), 1e-5),
                                     0.25, 4.0))
        gain *= 2.0 ** ev
        # Above 90% luminance, use a smooth shoulder, not hard highlight clipping.
        scaled = rgb * gain
        peak = scaled.max(axis=2, keepdims=True)
        shoulder = np.where(peak > 0.9,
                            0.9 + 0.1 * (1.0 - np.exp(-np.maximum(peak - 0.9, 0) / 0.1)),
                            peak)
        return np.clip(scaled * shoulder / np.maximum(peak, 1e-7), 0, 1)

    @staticmethod
    def _tone_map(srgb: np.ndarray, clip: float, strength: float) -> np.ndarray:
        # OpenCV float RGB -> Lab supports high precision (uint16 RGB -> Lab does not).
        lab = cv2.cvtColor(np.ascontiguousarray(srgb), cv2.COLOR_RGB2LAB)
        lightness = np.rint(np.clip(lab[:, :, 0] / 100, 0, 1) * 65535).astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(lightness).astype(np.float32) * (100 / 65535)
        lab[:, :, 0] = lab[:, :, 0] * (1 - strength) + enhanced * strength
        return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)

    def apply_corrections(self, rgb16: np.ndarray, params: ProcessingParams,
                          *, cancel: threading.Event | None = None,
                          on_stage: StageCallback | None = None) -> np.ndarray:
        """Return display-ready float32 sRGB in [0, 1], with no 8-bit truncation."""
        if rgb16.dtype != np.uint16 or rgb16.ndim != 3 or rgb16.shape[2] != 3:
            raise ValueError("Очаква се uint16 RGB масив с форма (H, W, 3).")
        if not rgb16.size:
            raise ValueError("Празно изображение.")
        rgb = rgb16.astype(np.float32) / 65535
        if params.white_balance in {"gray_world", "shades_of_gray"}:
            self._stage("Баланс на бялото…", on_stage, cancel)
            rgb = self._white_balance(rgb, params.white_balance)
        if params.auto_exposure or params.exposure_ev:
            self._stage("Експозиция и светлини…", on_stage, cancel)
            rgb = self._exposure(rgb, params.exposure_ev, params.auto_exposure)
        self._stage("sRGB тонална крива…", on_stage, cancel)
        srgb = self.linear_to_srgb(rgb)
        if params.tone_mapping and params.tone_strength > 0:
            self._stage("CLAHE върху L канала…", on_stage, cancel)
            srgb = self._tone_map(srgb, params.clahe_clip, params.tone_strength)
        if params.denoise and params.denoise_strength > 0:
            self._stage("Премахване на шум…", on_stage, cancel)
            # This specific OpenCV function requires 8-bit BGR input.
            bgr8 = cv2.cvtColor(self.to_uint8(srgb), cv2.COLOR_RGB2BGR)
            clean = cv2.fastNlMeansDenoisingColored(
                bgr8, None, params.denoise_strength, params.denoise_strength, 7, 21)
            residual = (cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) -
                        cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB).astype(np.float32)) / 255
            srgb = np.clip(srgb + residual, 0, 1)
        if params.sharpen and params.sharpen_amount > 0:
            self._stage("Изостряне…", on_stage, cancel)
            # Sharpen luminance only to reduce coloured halos.
            luma = srgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            blurred = cv2.GaussianBlur(luma, (0, 0), 1.0)
            detail = (luma - blurred) * params.sharpen_amount
            srgb = np.clip(srgb + detail[:, :, None], 0, 1)
        check_cancel(cancel)
        return np.ascontiguousarray(srgb, dtype=np.float32)

    @staticmethod
    def _preview_image(srgb: np.ndarray, max_size: tuple[int, int] = (1400, 1000)) -> Image.Image:
        h, w = srgb.shape[:2]
        scale = min(1.0, max_size[0] / w, max_size[1] / h)
        if scale < 1:
            srgb = cv2.resize(srgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                              interpolation=cv2.INTER_AREA)
        return Image.fromarray(RawEngine.to_uint8(srgb))

    def preview(self, input_path: str | Path, params: ProcessingParams, *, draft: bool = True,
                cancel: threading.Event | None = None,
                on_stage: StageCallback | None = None) -> PreviewResult:
        rgb16 = self.decode(input_path, params, draft=draft, cancel=cancel, on_stage=on_stage)
        before = self._preview_image(self.linear_to_srgb(rgb16.astype(np.float32) / 65535))
        corrected = self.apply_corrections(rgb16, params, cancel=cancel, on_stage=on_stage)
        return PreviewResult(before, self._preview_image(corrected), Path(input_path).name,
                             (rgb16.shape[1], rgb16.shape[0]), draft)

    def save_image(self, srgb: np.ndarray, output_path: str | Path, params: ProcessingParams,
                   *, overwrite: bool = False, cancel: threading.Event | None = None) -> Path:
        """Encode first, then atomically publish; never leave a partial final file.

        Exclusive creation of a placeholder reserves new filenames against another
        batch/process. It is removed on error. Existing exports need explicit overwrite.
        """
        check_cancel(cancel)
        destination = Path(output_path).expanduser().resolve()
        expected = {".jpg", ".jpeg"} if params.output_format == "jpg" else {".png"}
        if destination.suffix.lower() not in expected:
            raise ValueError("Разширението на изходния файл не съответства на формата.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if params.output_format == "jpg":
            pixels = self.to_uint8(srgb)
            extension = ".jpg"
            options = [cv2.IMWRITE_JPEG_QUALITY, params.quality]
        else:
            pixels = self.to_uint16(srgb) if params.png_bit_depth == 16 else self.to_uint8(srgb)
            extension = ".png"
            options = [cv2.IMWRITE_PNG_COMPRESSION, params.png_compression]
        success, encoded = cv2.imencode(extension, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), options)
        if not success:
            raise OSError("Неуспешно кодиране на изображението.")
        reserved = False
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".raw-export-", suffix=".tmp",
                                             dir=destination.parent, delete=False) as temporary:
                temp_name = temporary.name
                temporary.write(encoded.tobytes())
                temporary.flush()
                os.fsync(temporary.fileno())
            check_cancel(cancel)
            if not overwrite:
                # Fail safely if a destination appeared after batch planning.
                fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                os.close(fd)
                reserved = True
            os.replace(temp_name, destination)
            temp_name = None
            reserved = False
            return destination
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
            if reserved:
                destination.unlink(missing_ok=True)

    def process_file(self, input_path: str | Path, output_path: str | Path,
                     params: ProcessingParams, *, overwrite: bool = False,
                     cancel: threading.Event | None = None,
                     on_stage: StageCallback | None = None,
                     include_preview: bool = False) -> ProcessResult:
        started = time.perf_counter()
        if Path(input_path).expanduser().resolve() == Path(output_path).expanduser().resolve():
            raise ValueError("Входният RAW файл не може да бъде изходен файл.")
        rgb16 = self.decode(input_path, params, cancel=cancel, on_stage=on_stage)
        corrected = self.apply_corrections(rgb16, params, cancel=cancel, on_stage=on_stage)
        self._stage("Запазване…", on_stage, cancel)
        destination = self.save_image(corrected, output_path, params,
                                      overwrite=overwrite, cancel=cancel)
        preview = self._preview_image(corrected) if include_preview else None
        return ProcessResult(destination, (corrected.shape[1], corrected.shape[0]),
                             time.perf_counter() - started, preview)

# ---------- batch.py ----------
"""Filesystem discovery and sequential background jobs with UI-neutral events."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import os
import threading
import time


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

# ---------- app_gui.py ----------
"""CustomTkinter desktop UI. All Tk operations stay on the main thread.

Workers communicate through Queue; a short after() poll delivers their results.
Settings are captured before spawning a worker, never read from Tk variables there.
"""

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

# ---------- raw_processor.py ----------
"""Launch the RAW Studio desktop application: python raw_processor.py."""
import sys


def main() -> None:
    try:
        import cv2
        import customtkinter as ctk
    except ImportError as error:
        print(f"Липсва зависимост: {error}\nИзпълни: python -m pip install -r requirements.txt",
              file=sys.stderr)
        raise SystemExit(1) from error
    # LibRaw/OpenCV may use their own native threads; batches are intentionally serial.
    cv2.setNumThreads(2)
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = AppGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
