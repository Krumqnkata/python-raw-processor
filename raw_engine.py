"""RAW decoding, high precision corrections and atomic JPG/PNG export.

This module does not know about Tkinter. Arrays are RGB, never BGR, until
the explicit OpenCV export/denoising boundary.
"""
from __future__ import annotations

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
