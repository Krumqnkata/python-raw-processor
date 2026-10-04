"""RAW Studio: complete standalone application. Generated from the modular project."""
from __future__ import annotations

# ---------- imaging.py ----------
"""Resolution-independent, non-destructive editing operations (float RGB)."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class LocalAdjustment:
    kind: str = 'brush'
    points: tuple[tuple[float, float], ...] = ()
    radius: float = .08
    exposure: float = .5
    shadows: float = 0.0
    saturation: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'points', tuple(tuple(p) for p in self.points))
        if self.kind not in {'brush', 'gradient'} or not self.points:
            raise ValueError('Маската трябва да е четка или градиент с точки.')
        if self.kind == 'gradient' and len(self.points) != 2:
            raise ValueError('Градиентът изисква две точки.')
        if len(self.points) > 5000 or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in self.points):
            raise ValueError('Невалидни координати на маската.')
        for name, low, high in [('radius', .005, .5), ('exposure', -3, 3), ('shadows', -1, 1), ('saturation', 0, 2)]:
            v = getattr(self, name)
            if not math.isfinite(v) or not low <= v <= high:
                raise ValueError('Невалидна локална корекция.')


def tonal(rgb, shadows=0., highlights=0., whites=0., blacks=0.):
    luma = rgb @ np.array([.2126, .7152, .0722], np.float32)
    delta = (shadows * .35 * (1-luma)**3 * (1-np.exp(-luma*12)) +
             highlights * .3 * luma**3 + whites * .2 * luma**6 +
             blacks * .15 * (1-luma)**6)
    return np.clip(rgb + delta[..., None], 0, 1).astype(np.float32)


def colour(rgb, saturation=1., vibrance=0., monochrome=False):
    luma = (rgb @ np.array([.2126, .7152, .0722], np.float32))[..., None]
    chroma = rgb.max(axis=2, keepdims=True) - rgb.min(axis=2, keepdims=True)
    gain = saturation * (1 + vibrance * (1-chroma))
    return np.repeat(luma,3,axis=2).astype(np.float32) if monochrome else np.clip(luma + (rgb-luma)*gain,0,1).astype(np.float32)


def mask_weights(shape, mask):
    h, w = shape[:2]
    if mask.kind == 'gradient':
        (x0,y0),(x1,y1) = mask.points
        yy,xx = np.ogrid[:h,:w]
        dx,dy = (x1-x0)*w,(y1-y0)*h
        distance = dx*dx+dy*dy
        if distance < 1e-6:
            return np.zeros((h,w), np.float32)
        return np.clip(((xx-x0*w)*dx+(yy-y0*h)*dy)/distance,0,1).astype(np.float32)
    weights = np.zeros((h,w), np.float32)
    radius = max(1., mask.radius * min(w,h))
    # Work only inside the brush bounding boxes, not H*W*number_of_points.
    points = list(mask.points)
    dense = [points[0]]
    for a,b in zip(points, points[1:]):
        steps = max(1, math.ceil(math.hypot((b[0]-a[0])*w,(b[1]-a[1])*h)/(radius*.4)))
        dense.extend((a[0]+(b[0]-a[0])*t/steps,a[1]+(b[1]-a[1])*t/steps) for t in range(1,steps+1))
    for x,y in dense:
        cx,cy = x*(w-1),y*(h-1)
        x0,x1 = max(0,int(cx-radius)),min(w,int(cx+radius)+2)
        y0,y1 = max(0,int(cy-radius)),min(h,int(cy+radius)+2)
        yy,xx = np.ogrid[y0:y1,x0:x1]
        stamp = np.clip(1-((xx-cx)**2+(yy-cy)**2)/radius**2,0,1)**2
        np.maximum(weights[y0:y1,x0:x1],stamp,out=weights[y0:y1,x0:x1])
    return weights


def local_adjustments(rgb, masks, check=lambda: None):
    for mask in masks:
        check()
        alpha = mask_weights(rgb.shape, mask)[...,None]
        linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        linear = np.clip(linear*2**mask.exposure,0,1)
        exposed = np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055).astype(np.float32)
        adjusted = tonal(exposed,shadows=mask.shadows)
        adjusted = colour(adjusted, mask.saturation)
        rgb = rgb*(1-alpha)+adjusted*alpha
    return np.ascontiguousarray(rgb,np.float32)


def lens_correct(rgb, p, check=lambda: None):
    if not p.lens_enabled:
        return rgb
    check()
    if not any((p.lens_k1,p.lens_k2,p.vignette,p.ca_red,p.ca_blue)):
        return rgb
    h,w = rgb.shape[:2]
    # Row blocks bound temporary coordinate memory for large RAW images.
    result = np.empty_like(rgb)
    for start in range(0,h,256):
        check()
        end = min(h,start+256)
        yy,xx = np.mgrid[start:end,:w].astype(np.float32)
        nx,ny = (xx-(w-1)/2)/max(w,h)*2,(yy-(h-1)/2)/max(w,h)*2
        r2 = nx*nx+ny*ny
        scale = 1+p.lens_k1*r2+p.lens_k2*r2*r2
        for channel,ca in enumerate((p.ca_red,0,p.ca_blue)):
            mx = ((nx*scale*(1+ca))*max(w,h)/2+(w-1)/2).astype(np.float32)
            my = ((ny*scale*(1+ca))*max(w,h)/2+(h-1)/2).astype(np.float32)
            result[start:end,:,channel] = cv2.remap(rgb[:,:,channel],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        result[start:end] *= np.exp(p.vignette*r2)[...,None]
    return np.clip(result,0,1)


def geometry(rgb, p):
    if p.rotation:
        rgb = np.rot90(rgb, -p.rotation)
    if p.flip_horizontal:
        rgb = rgb[:,::-1]
    if p.flip_vertical:
        rgb = rgb[::-1]
    if p.straighten:
        h,w = rgb.shape[:2]
        matrix = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),p.straighten,1)
        rgb = cv2.warpAffine(np.ascontiguousarray(rgb),matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    x0,y0,x1,y1 = p.crop
    h,w = rgb.shape[:2]
    left,top = min(w-1,round(x0*w)),min(h-1,round(y0*h))
    right,bottom = max(left+1,round(x1*w)),max(top+1,round(y1*h))
    return np.ascontiguousarray(rgb[top:bottom,left:right])


def resize_export(rgb, max_edge):
    h,w = rgb.shape[:2]
    if max_edge and max(h,w) > max_edge:
        scale = max_edge/max(h,w)
        rgb = cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    return rgb


def histogram(image):
    pixels = np.asarray(image.convert('RGB'))
    h,w = pixels.shape[:2]
    step = max(1,int(math.ceil(math.sqrt(h*w/500000))))
    sample = pixels[::step,::step]
    channels = [np.bincount(sample[:,:,i].ravel(),minlength=256).tolist() for i in range(3)]
    luma = np.rint(sample @ np.array([.2126,.7152,.0722])).astype(np.uint8)
    return {'rgb':channels,'luma':np.bincount(luma.ravel(),minlength=256).tolist(),
            'shadows':float((sample.min(axis=2)==0).mean()),'highlights':float((sample.max(axis=2)==255).mean())}

# ---------- metadata.py ----------
"""Read selected EXIF fields and embed safe metadata without rewriting pixels."""
from pathlib import Path
import struct
import zlib
import exifread
from PIL import Image, TiffImagePlugin

FIELDS = {'Image Make':271,'Image Model':272,'EXIF LensModel':42036,
          'Image DateTime':306,'EXIF DateTimeOriginal':36867,'EXIF DateTimeDigitized':36868,
          'EXIF ExposureTime':33434,'EXIF FNumber':33437,'EXIF ISOSpeedRatings':34855,
          'EXIF FocalLength':37386,'EXIF ExposureBiasValue':37380}


def read_metadata(path):
    with Path(path).open('rb') as handle:
        tags = exifread.process_file(handle,details=False,strict=False)
    result = {}
    for key in FIELDS:
        if key not in tags:
            continue
        tag = tags[key]
        if tag.field_type in {2,7}:
            value = str(tag).strip('\x00')[:512]
        else:
            vals = tag.values
            value = vals[0] if isinstance(vals,list) and vals else vals
            if hasattr(value,'num'):
                value = [int(value.num),int(value.den)]
            if not isinstance(value,(str,int,float,list)):
                value = str(value)
        result[key] = value
    return result


def exif_bytes(data, params, size):
    exif = Image.Exif()
    sub = {40961:1,40962:size[0],40963:size[1]}
    exif[274] = 1  # Pixels already reflect RAW orientation, rotation and crop.
    exif[305] = 'RAW Studio'
    for key,value in data.items():
        if key not in FIELDS:
            continue
        is_camera = key in {'Image Make','Image Model','EXIF LensModel'}
        is_date = 'DateTime' in key
        if (is_camera and not params.exif_camera) or (is_date and not params.exif_date) or (not is_camera and not is_date and not params.exif_exposure):
            continue
        if isinstance(value,list) and len(value)==2:
            if value[1]==0: continue
            value = TiffImagePlugin.IFDRational(*value)
        (sub if key.startswith('EXIF ') else exif)[FIELDS[key]] = value
    exif[34665] = sub
    return exif.tobytes()


def embed_exif(encoded, extension, exif):
    """Insert JPEG APP1 or PNG eXIf chunks, preserving PNG16 sample precision."""
    data = encoded.tobytes() if hasattr(encoded,'tobytes') else bytes(encoded)
    if extension == '.jpg':
        if len(exif)+2 > 65535:
            raise ValueError('EXIF блокът е твърде голям.')
        return data[:2]+b'\xff\xe1'+struct.pack('>H',len(exif)+2)+exif+data[2:]
    payload = exif[6:] if exif.startswith(b'Exif\x00\x00') else exif
    chunk = b'eXIf'+payload
    block = struct.pack('>I',len(payload))+chunk+struct.pack('>I',zlib.crc32(chunk)&0xffffffff)
    # Insert after IHDR, before image data.
    offset = 8+12+struct.unpack('>I',data[8:12])[0]
    return data[:offset]+block+data[offset:]

# ---------- raw_engine.py ----------
"""RAW decoding, high precision corrections and atomic JPG/PNG export.

This module does not know about Tkinter. Arrays are RGB, never BGR, until
the explicit OpenCV export/denoising boundary.
"""

from dataclasses import dataclass, replace
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

    shadows: float = 0.0
    highlights: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    saturation: float = 1.0
    vibrance: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    wb_gains: tuple[float, float, float] = (1., 1., 1.)
    monochrome: bool = False
    rotation: int = 0
    straighten: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    crop: tuple[float, float, float, float] = (0., 0., 1., 1.)
    masks: tuple[LocalAdjustment, ...] = ()
    lens_enabled: bool = False
    auto_lens: bool = False
    lens_k1: float = 0.0
    lens_k2: float = 0.0
    vignette: float = 0.0
    ca_red: float = 0.0
    ca_blue: float = 0.0
    max_edge: int = 0
    filename_template: str = '{stem}'
    preserve_exif: bool = False
    exif_camera: bool = True
    exif_date: bool = True
    exif_exposure: bool = True

    def __post_init__(self) -> None:
        for name in ['auto_exposure','tone_mapping','denoise','sharpen','monochrome','flip_horizontal','flip_vertical','lens_enabled','auto_lens','preserve_exif','exif_camera','exif_date','exif_exposure']:
            if type(getattr(self,name)) is not bool:
                raise ValueError('Невалиден превключвател: '+name)
        object.__setattr__(self, 'crop', tuple(self.crop))
        object.__setattr__(self, 'wb_gains', tuple(self.wb_gains))
        object.__setattr__(self, 'masks', tuple(self.masks))
        if len(self.crop) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.crop) or not (self.crop[0] < self.crop[2] and self.crop[1] < self.crop[3]):
            raise ValueError('Невалидно изрязване.')
        if len(self.wb_gains) != 3 or any(not math.isfinite(v) or not .25 <= v <= 4 for v in self.wb_gains):
            raise ValueError('Невалидни коефициенти за бялото.')
        if len(self.masks) > 100 or any(not isinstance(m, LocalAdjustment) for m in self.masks):
            raise ValueError('Невалидни локални маски (максимум 100).')
        if type(self.rotation) is not int or not 0 <= self.rotation <= 3:
            raise ValueError('Невалидно завъртане.')
        if type(self.max_edge) is not int or not 0 <= self.max_edge <= 16000:
            raise ValueError('Дългата страна трябва да е 0–16000 px.')
        validate_template(self.filename_template)
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
            ('shadows', -1, 1), ('highlights', -1, 1), ('whites', -1, 1), ('blacks', -1, 1),
            ('saturation', 0, 2), ('vibrance', -1, 1), ('temperature', -1, 1), ('tint', -1, 1),
            ('straighten', -15, 15), ('lens_k1', -.5, .5), ('lens_k2', -.5, .5),
            ('vignette', -1, 2), ('ca_red', -.02, .02), ('ca_blue', -.02, .02),
        ]:
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Невалидна стойност за {name}: {value}")
            if isinstance(value,float): object.__setattr__(self,name,round(value,6))
        if type(self.quality) is not int or type(self.png_compression) is not int:
            raise ValueError("Качеството и PNG компресията трябва да са цели числа.")


@dataclass
class PreviewResult:
    before: Image.Image
    after: Image.Image
    name: str
    dimensions: tuple[int, int]
    draft: bool = True
    metadata: dict | None = None
    histogram: dict | None = None


@dataclass
class ProcessResult:
    output_path: Path
    dimensions: tuple[int, int]
    seconds: float
    preview: Image.Image | None = None


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ProcessingCancelled()


def validate_template(template):
    import string
    if not isinstance(template,str) or not template or len(template)>120:
        raise ValueError('Невалиден шаблон за име.')
    try:
        for _,field,spec,conversion in string.Formatter().parse(template):
            if field is not None and (field not in {'stem','index','date','camera'} or conversion or (spec and (field != 'index' or spec not in {'02d','03d','04d','05d','06d'}))):
                raise ValueError('Използвай {stem}, {index:04d}, {date}, {camera}.')
        rendered = template.format(stem='photo',index=1,date='20261004',camera='camera')
        if any(c in rendered for c in '/\\<>:"|?*') or any(ord(c)<32 for c in rendered) or rendered.strip(' .') != rendered or not rendered:
            raise ValueError('Името съдържа забранени символи.')
    except ValueError as error:
        raise ValueError('Невалиден шаблон за име: '+str(error)) from error


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
        gains = np.asarray(params.wb_gains,np.float32) * np.exp(np.array([params.temperature*.5,-params.tint*.35,-params.temperature*.5],np.float32))
        rgb = np.clip(rgb*gains,0,1)
        if params.auto_exposure or params.exposure_ev:
            self._stage("Експозиция и светлини…", on_stage, cancel)
            rgb = self._exposure(rgb, params.exposure_ev, params.auto_exposure)
        self._stage("sRGB тонална крива…", on_stage, cancel)
        srgb = self.linear_to_srgb(rgb)
        if params.tone_mapping and params.tone_strength > 0:
            self._stage("CLAHE върху L канала…", on_stage, cancel)
            srgb = self._tone_map(srgb, params.clahe_clip, params.tone_strength)
        srgb = tonal(srgb, params.shadows, params.highlights, params.whites, params.blacks)
        srgb = colour(srgb, params.saturation, params.vibrance, params.monochrome)
        srgb = lens_correct(srgb, params, lambda: check_cancel(cancel))
        srgb = geometry(srgb, params)
        srgb = local_adjustments(srgb, params.masks, lambda: check_cancel(cancel))
        srgb = resize_export(srgb, params.max_edge)
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
        params, metadata = self._file_params(input_path, params, on_stage, cancel)
        rgb16 = self.decode(input_path, params, draft=draft, cancel=cancel, on_stage=on_stage)
        # Preview never applies export resizing. Exact mode retains pixels for 100% zoom.
        preview_params = replace(params, max_edge=0)
        before_rgb = geometry(lens_correct(self.linear_to_srgb(rgb16.astype(np.float32)/65535), params, lambda: check_cancel(cancel)), params)
        corrected = self.apply_corrections(rgb16, preview_params, cancel=cancel, on_stage=on_stage)
        size = (corrected.shape[1], corrected.shape[0])
        bound = (1400,1000) if draft else size
        before = self._preview_image(before_rgb, bound)
        after = self._preview_image(corrected, bound)
        return PreviewResult(before, after, Path(input_path).name, size, draft, metadata, histogram(after))

    @staticmethod
    def _file_params(path, params, on_stage=None, cancel=None):
        try:
            metadata = read_metadata(path)
        except Exception as error:
            metadata = {}
            if on_stage: on_stage('EXIF данните не могат да се прочетат: '+str(error))
        if params.auto_lens:
            params, name = match_lens_profile(Path.home()/'.raw-studio'/'lenses', metadata, params)
            if on_stage: on_stage('Профил за обектив: '+name if name else 'Няма съвпадащ профил; използвам ръчните настройки за обектива.')
        check_cancel(cancel)
        return params, metadata

    def save_image(self, srgb: np.ndarray, output_path: str | Path, params: ProcessingParams,
                   *, overwrite: bool = False, cancel: threading.Event | None = None,
                   metadata: dict | None = None) -> Path:
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
        payload = encoded.tobytes()
        if params.preserve_exif:
            payload = embed_exif(payload, extension, exif_bytes(metadata or {}, params, (srgb.shape[1],srgb.shape[0])))
        reserved = False
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".raw-export-", suffix=".tmp",
                                             dir=destination.parent, delete=False) as temporary:
                temp_name = temporary.name
                temporary.write(payload)
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
        params, metadata = self._file_params(input_path, params, on_stage, cancel)
        rgb16 = self.decode(input_path, params, cancel=cancel, on_stage=on_stage)
        corrected = self.apply_corrections(rgb16, params, cancel=cancel, on_stage=on_stage)
        self._stage("Запазване…", on_stage, cancel)
        destination = self.save_image(corrected, output_path, params,
                                      overwrite=overwrite, cancel=cancel, metadata=metadata)
        preview = self._preview_image(corrected) if include_preview else None
        return ProcessResult(destination, (corrected.shape[1], corrected.shape[0]),
                             time.perf_counter() - started, preview)

# ---------- studio.py ----------
"""Versioned projects, presets, per-photo edit history and lens profiles."""
from dataclasses import asdict, fields, replace
from pathlib import Path
import json
import os
import tempfile


def atomic_json(path, data):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as handle:
            name = handle.name
            json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name,path)
        name = None
    finally:
        if name: Path(name).unlink(missing_ok=True)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 25_000_000:
        raise ValueError('Файлът с настройки е твърде голям.')
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def params_from_dict(data):
    if not isinstance(data,dict) or set(data)-{f.name for f in fields(ProcessingParams)}:
        raise ValueError('Непознати настройки в проекта/профила.')
    data = dict(data)
    data['masks'] = tuple(LocalAdjustment(**m) for m in data.get('masks',()))
    return ProcessingParams(**data)


BUILTIN_PRESETS = {
    'Естествено':{},
    'Портрет':{'tone_strength':.18,'shadows':.18,'vibrance':.12,'sharpen_amount':.25},
    'Пейзаж':{'tone_strength':.38,'highlights':-.2,'vibrance':.25,'sharpen_amount':.6},
    'Черно-бяло':{'monochrome':True,'tone_strength':.4},
}
# Geometry and masks are photo-specific. Export options belong to the job.
EDIT_FIELDS = {f.name for f in fields(ProcessingParams)}-{
    'crop','rotation','straighten','flip_horizontal','flip_vertical','masks',
    'max_edge','filename_template','preserve_exif','exif_camera','exif_date','exif_exposure',
    'output_format','quality','png_compression','png_bit_depth',
}
EXPORT_FIELDS = {f.name for f in fields(ProcessingParams)}-EDIT_FIELDS-{
    'crop','rotation','straighten','flip_horizontal','flip_vertical','masks',
}


class PhotoState:
    def __init__(self, path, params=None, included=True, rating=0):
        self.path = Path(path).expanduser().resolve()
        self.params = params or ProcessingParams()
        self.included = bool(included)
        if type(rating) is not int or not 0 <= rating <= 5:
            raise ValueError('Оценката трябва да е от 0 до 5.')
        self.rating = rating
        self.history = [self.params]
        self.cursor = 0

    def set_params(self, params):
        if self.params == params:
            return
        self.history = self.history[:self.cursor+1]
        self.history.append(params)
        self.history = self.history[-100:]
        self.cursor = len(self.history)-1
        self.params = params

    def undo(self):
        self.cursor = max(0,self.cursor-1)
        self.params = self.history[self.cursor]
        return self.params

    def redo(self):
        self.cursor = min(len(self.history)-1,self.cursor+1)
        self.params = self.history[self.cursor]
        return self.params

    def serialize(self, base):
        try: path = os.path.relpath(self.path,base)
        except ValueError: path = str(self.path)
        return {'path':path,'params':asdict(self.params),'included':self.included,'rating':self.rating,
                'history':[asdict(p) for p in self.history],'cursor':self.cursor}


class EditSession:
    def __init__(self):
        self.photos = {}
        self.defaults = ProcessingParams()
        self.output = ''
        self.selected = ''
        self.project_path = None

    def add(self, paths):
        known = {os.path.normcase(key) for key in self.photos}
        for path in paths:
            key = str(Path(path).expanduser().resolve())
            if os.path.normcase(key) not in known:
                self.photos[key] = PhotoState(path,self.defaults)
                known.add(os.path.normcase(key))

    def save(self, path):
        path = Path(path).expanduser().resolve()
        selected = self.selected
        if selected:
            try: selected = os.path.relpath(selected,path.parent)
            except ValueError: pass
        atomic_json(path,{'version':1,'defaults':asdict(self.defaults),'output':self.output,'selected':selected,
                          'photos':[p.serialize(path.parent) for p in self.photos.values()]})
        self.project_path = path

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        data = read_json(path)
        if not isinstance(data,dict) or data.get('version') != 1 or not isinstance(data.get('photos'),list):
            raise ValueError('Неподдържан проект.')
        session = cls()
        session.defaults = params_from_dict(data.get('defaults',{}))
        session.output = str(data.get('output',''))
        selected = str(data.get('selected',''))
        session.selected = str((path.parent/selected).resolve()) if selected and not Path(selected).is_absolute() else selected
        for row in data['photos']:
            source = Path(row['path'])
            if not source.is_absolute(): source = path.parent/source
            photo = PhotoState(source,params_from_dict(row['params']),row.get('included',True),row.get('rating',0))
            history = [params_from_dict(p) for p in row.get('history',[])][-100:]
            cursor = row.get('cursor',len(history)-1)
            if history and type(cursor) is int and 0 <= cursor < len(history):
                photo.history,photo.cursor = history,cursor
                photo.params = history[cursor]
            session.photos[str(photo.path)] = photo
        session.project_path = path
        return session


def save_preset(path, params):
    atomic_json(path,{'version':1,'settings':{k:v for k,v in asdict(params).items() if k in EDIT_FIELDS}})


def load_preset(path, current):
    data = read_json(path)
    settings = data.get('settings',{})
    if data.get('version') != 1 or set(settings)-EDIT_FIELDS:
        raise ValueError('Неподдържан preset.')
    return replace(current,**settings)


LENS_FIELDS = {'lens_k1','lens_k2','vignette','ca_red','ca_blue'}


def save_lens_profile(path, params, metadata):
    atomic_json(path,{'version':1,'camera':str(metadata.get('Image Model','')),
                     'lens':str(metadata.get('EXIF LensModel','')),
                     'settings':{k:getattr(params,k) for k in LENS_FIELDS}})


def load_lens_profile(path, current):
    data = read_json(path)
    if data.get('version') != 1 or set(data.get('settings',{}))-LENS_FIELDS:
        raise ValueError('Неподдържан профил за обектив.')
    return replace(current,lens_enabled=True,**data['settings'])


def match_lens_profile(folder, metadata, current):
    camera,lens = str(metadata.get('Image Model','')),str(metadata.get('EXIF LensModel',''))
    if not camera or not lens:
        return current,None
    for path in sorted(Path(folder).glob('*.json')):
        try:
            data = read_json(path)
            if data.get('camera')==camera and data.get('lens')==lens:
                return load_lens_profile(path,current),path.name
        except (ValueError,KeyError,TypeError,OSError):
            continue
    return current,None

# ---------- batch.py ----------
"""Filesystem discovery and sequential background jobs with UI-neutral events."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import os
import hashlib
import json
import re
from datetime import datetime
from dataclasses import asdict
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

# ---------- app_gui.py ----------
"""CustomTkinter desktop UI. All Tk operations stay on the main thread.

Workers communicate through Queue; a short after() poll delivers their results.
Settings are captured before spawning a worker, never read from Tk variables there.
"""

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
        self.compare_var = tk.StringVar(value="Сравнение")

        self._build_sidebar()
        self._build_main()
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
        self._label(parent, text, font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=MUTED).pack(fill="x", padx=12, pady=(20, 8))

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
        self._slider_labels[slider] = (value_label,formatter)
        return slider, value_label

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self,width=340,corner_radius=0,fg_color=CARD)
        sidebar.grid(row=0,column=0,sticky='nsew')
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0,weight=1)
        sidebar.grid_rowconfigure(1,weight=1)
        brand = ctk.CTkFrame(sidebar,fg_color='transparent')
        brand.grid(row=0,column=0,sticky='ew',padx=20,pady=(16,4))
        self._label(brand,'RAW Studio',font=ctk.CTkFont(size=25,weight='bold')).pack(side='left')
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
        self._section(controls,'СНИМКИ И PRESETS')
        self._button(controls,'+ Добави RAW файлове',self._choose_files,fg_color=ACCENT)
        self._button(controls,'Избери папка със снимки',self._choose_folder)
        self._switch(controls,'Включи подпапките',self.recursive_var,command=lambda:None,setting=False)
        self.file_count_label = self._label(controls,'0 избрани снимки',text_color=MUTED)
        self.file_count_label.pack(fill='x',padx=12)
        self._button(controls,'Изчисти списъка',self._clear_files)
        preset = ctk.CTkOptionMenu(controls,values=list(BUILTIN_PRESETS),variable=self.preset_var,command=lambda name:self._ui_action(self._apply_builtin,name))
        preset.pack(fill='x',padx=12,pady=8)
        self._setting_widgets.append(preset)
        self._button(controls,'Запази собствен preset…',self._save_preset)
        self._button(controls,'Зареди preset…',self._load_preset)
        self._button(controls,'Копирай настройките',self._copy_settings)
        self._button(controls,'Приложи към включените снимки',self._paste_settings)
        self._section(controls,'БАЛАНС И СВЕТЛИНА')
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
        self._section(controls,'ДЕТАЙЛ')
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
        self._section(tools,'ИЗРЯЗВАНЕ И ГЕОМЕТРИЯ')
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
        self._section(tools,'ЛОКАЛНИ КОРЕКЦИИ')
        self.local_radius,_ = self._slider(tools,'Размер на четката',.005,.5,.08,99,lambda v:f'{v:.1%}')
        self.local_ev,_ = self._slider(tools,'Локална експозиция',-3,3,.5,120,lambda v:f'{v:+.2f} EV')
        self.local_shadows,_ = self._slider(tools,'Локални сенки',-1,1,0,100,lambda v:f'{v:+.2f}')
        self.local_saturation,_ = self._slider(tools,'Локална наситеност',0,2,1,100,lambda v:f'{v:.2f}')
        self.mask_label = self._label(tools,'0 локални маски',text_color=MUTED)
        self.mask_label.pack(fill='x',padx=12)
        self._button(tools,'Изтрий последната маска',lambda:self._edit_change(masks=self._params().masks[:-1]))
        self._button(tools,'Изтрий всички маски',lambda:self._edit_change(masks=()))
        self._label(tools,'Четка: рисувай с ляв бутон. Градиент: влачи от зона без корекция към зона с пълна корекция. Маските следват координатите на изрязания кадър.',wraplength=260,justify='left',text_color=MUTED).pack(fill='x',padx=12,pady=8)
        self._section(tools,'КОРЕКЦИИ НА ОБЕКТИВА')
        self._switch(tools,'Включи корекциите',self.lens_var)
        self._switch(tools,'Автоматичен собствен профил',self.auto_lens_var)
        for key,title,low,high in [('lens_k1','Изкривяване k1',-.5,.5),('lens_k2','Изкривяване k2',-.5,.5),('vignette','Винетиране',-1,2),('ca_red','Червен цветен кант',-.02,.02),('ca_blue','Син цветен кант',-.02,.02)]:
            self._extra_sliders[key],_ = self._slider(tools,title,low,high,0,200,lambda v:f'{v:+.4f}')
        self._button(tools,'Зареди профил за обектив…',self._load_lens)
        self._button(tools,'Запази профил за този обектив…',self._save_lens)

        export = panels['Експорт']
        self._section(export,'ФОРМАТ И РАЗМЕР')
        self.format_control = ctk.CTkSegmentedButton(export,values=['JPG','PNG'],variable=self.format_var,command=self._update_format,selected_color=ACCENT)
        self.format_control.pack(fill='x',padx=12,pady=6)
        self._setting_widgets.append(self.format_control)
        self.quality_slider,self.quality_label = self._slider(export,'Качество на JPG',1,100,92,99,lambda v:f'{round(v)}%')
        self.compression_slider,self.compression_label = self._slider(export,'PNG компресия',0,9,4,9,lambda v:str(round(v)))
        self.depth_menu = ctk.CTkOptionMenu(export,values=['8 бита','16 бита'],variable=self.depth_var)
        self.depth_menu.pack(fill='x',padx=12,pady=7)
        self._setting_widgets.append(self.depth_menu)
        self.format_hint = self._label(export,'',wraplength=260,justify='left',text_color=MUTED,font=ctk.CTkFont(size=11))
        self.format_hint.pack(fill='x',padx=12,pady=4)
        self._extra_sliders['max_edge'],_ = self._slider(export,'Дълга страна (0 = оригинал)',0,16000,0,160,lambda v:f'{round(v)} px')
        edge_entry = ctk.CTkEntry(export,textvariable=self.max_edge_var,placeholder_text='Точен размер, например 2048')
        edge_entry.pack(fill='x',padx=12,pady=4)
        self._setting_widgets.append(edge_entry)
        self._extra_sliders['max_edge'].configure(command=lambda value:self._set_edge(value))
        self._label(export,'Шаблон за име',text_color=MUTED).pack(fill='x',padx=12,pady=(12,4))
        naming = ctk.CTkEntry(export,textvariable=self.template_var)
        naming.pack(fill='x',padx=12,pady=4)
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
        self._section(project,'ПРОЕКТ И ИСТОРИЯ')
        self._button(project,'Запази проект…',self._save_project)
        self._button(project,'Отвори проект…',self._load_project)
        self._button(project,'Отмени (Ctrl+Z)',self._undo)
        self._button(project,'Повтори (Ctrl+Y)',self._redo)
        self._button(project,'Нулирай редакциите',lambda:self._replace_edit(ProcessingParams()))
        self.project_label = self._label(project,'Няма отворен проект',wraplength=260,text_color=MUTED)
        self.project_label.pack(fill='x',padx=12,pady=10)
        self._section(project,'НАБЛЮДАВАНА ПАПКА')
        self._button(project,'Избери папка за автоматичен експорт…',self._start_watch)
        self.watch_stop = self._button(project,'Спри наблюдението',self._stop_watch)
        self.watch_label = self._label(project,'Наблюдението е изключено.',wraplength=260,text_color=MUTED)
        self.watch_label.pack(fill='x',padx=12,pady=8)
        self.metadata_box = ctk.CTkTextbox(project,height=180,wrap='word')
        self.metadata_box.pack(fill='x',padx=12,pady=12)
        self.metadata_box.configure(state='disabled')
        footer = ctk.CTkFrame(sidebar,fg_color='transparent')
        footer.grid(row=2,column=0,sticky='ew',padx=18,pady=12)
        self.start_button = ctk.CTkButton(footer,text='Експортирай включените снимки',height=40,fg_color=ACCENT,command=self._start_batch)
        self.start_button.pack(fill='x')
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
        self._label(header, "RAW → JPG / PNG", text_color=ACCENT,
                    font=ctk.CTkFont(size=12)).pack(side="right")

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
                                        text_color=MUTED, font=ctk.CTkFont(size=11))
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
        self._label(toolbar,'Преди / След',text_color=MUTED).pack(side='right')
        self._build_gallery(main)

        progress_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=10)
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

        log_header = ctk.CTkFrame(main, fg_color="transparent")
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
                    font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="ew", pady=8)
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
        self.session.defaults = replace(ProcessingParams(),**{k:getattr(current_params,k) for k in EDIT_FIELDS | EXPORT_FIELDS})
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
        self._preview = None
        self._batch_preview = None
        self._images.clear()
        self.before_image_label.configure(image=None, text="Зареждане…")
        self.after_image_label.configure(image=None, text="Зареждане…")
        self.preview_info.configure(text="Зареждане на избраната снимка…")
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._loading or self._closing or not self.files or self._busy in {"batch", "scan"}:
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
            self._commit_current()
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
        for widget in self._gallery_controls:
            widget.configure(state='disabled' if self._busy in {'batch','scan'} or self._closing else 'normal')
        self.pause_button.configure(state='normal' if self._busy=='batch' and not self._closing else 'disabled')
        self._update_format()

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
        if self._closing and not (self._worker and self._worker.is_alive()):
            self.destroy()
            return
        self._poll_id = self.after(70, self._poll_events)

    def _handle_event(self, event: str, data: dict) -> None:
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
                self._batch_preview = None
                self._render_preview()
                result = self._preview
                self._metadata = result.metadata or {}
                self._draw_histogram(result.histogram)
                self.metadata_box.configure(state='normal')
                self.metadata_box.delete('1.0','end')
                self.metadata_box.insert('end','\n'.join(f'{k}: {v}' for k,v in self._metadata.items()) or 'Няма достъпни EXIF данни.')
                self.metadata_box.configure(state='disabled')
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
        if previous_kind=='batch': self._watch_exporting = False
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
        self._closing = True
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
        if self._worker and self._worker.is_alive():
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
        self._clipboard = {k:getattr(p,k) for k in EDIT_FIELDS}
        self._log('Копирани настройки за светлина, цвят, детайл и обектив.')

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
        if not self._preview or self._busy in {'batch','scan'}: return
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
        if not self._preview or self._busy in {'batch','scan'}: return
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
        outer = ctk.CTkFrame(parent,fg_color=CARD,corner_radius=8)
        outer.grid(row=7,column=0,sticky='ew',pady=(8,0))
        top = ctk.CTkFrame(outer,fg_color='transparent')
        top.pack(fill='x',padx=8,pady=2)
        ctk.CTkButton(top,text='◀',width=32,height=22,command=lambda:self._gallery_step(-1)).pack(side='left')
        ctk.CTkButton(top,text='▶',width=32,height=22,command=lambda:self._gallery_step(1)).pack(side='left',padx=4)
        self.gallery_label = self._label(top,'Галерия',text_color=MUTED,font=ctk.CTkFont(size=11))
        self.gallery_label.pack(side='left',padx=6)
        for title,action in [('Всички',lambda:self._set_all_included(True)),('Нито една',lambda:self._set_all_included(False)),('Текущата',self._include_current_only)]:
            ctk.CTkButton(top,text=title,width=65,height=22,command=action).pack(side='left',padx=2)
        ctk.CTkCheckBox(top,text='Само любими (★ ≥ 1)',variable=self.favorite_filter_var,command=lambda:self._refresh_gallery(reset=True),height=20,font=ctk.CTkFont(size=11)).pack(side='right')
        self.gallery = ctk.CTkScrollableFrame(outer,orientation='horizontal',height=95,fg_color='transparent')
        self.gallery.pack(fill='x',padx=4,pady=2)

    def _gallery_step(self, delta):
        self._gallery_page = max(0,self._gallery_page+delta)
        self._refresh_gallery()

    def _refresh_gallery(self,reset=False):
        if reset: self._gallery_page = 0
        self._thumb_cancel.set()
        self._thumb_epoch += 1
        self._gallery_buttons.clear()
        self._gallery_controls.clear()
        for widget in self.gallery.winfo_children(): widget.destroy()
        photos = [p for p in self.session.photos.values() if not self.favorite_filter_var.get() or p.rating>0]
        page_count = max(1,math.ceil(len(photos)/40))
        self._gallery_page = min(self._gallery_page,page_count-1)
        page = photos[self._gallery_page*40:(self._gallery_page+1)*40]
        included_count = sum(p.included for p in self.session.photos.values())
        self.start_button.configure(text=f'Експортирай {included_count} снимки')
        self.gallery_label.configure(text=f'{len(photos)} снимки · стр. {self._gallery_page+1}/{page_count}')
        for photo in page:
            cell = ctk.CTkFrame(self.gallery,width=105,fg_color='transparent')
            cell.pack(side='left',padx=3)
            key = str(photo.path)
            thumb = self._thumb_cache.get(key)
            image = ctk.CTkImage(light_image=thumb,dark_image=thumb,size=thumb.size) if thumb else None
            button = ctk.CTkButton(cell,text=photo.path.name[:17],width=100,height=52,image=image,compound='top',font=ctk.CTkFont(size=10),command=lambda path=photo.path:self._ui_action(self._select_photo,path))
            button._thumbnail = image
            button.pack(fill='x')
            self._gallery_buttons[key] = button
            included = tk.BooleanVar(value=photo.included)
            check = ctk.CTkCheckBox(cell,text='Експорт',width=62,height=18,checkbox_width=14,checkbox_height=14,variable=included,font=ctk.CTkFont(size=10),command=lambda ph=photo,v=included:self._set_included(ph,v.get()))
            check.pack(side='left',pady=3)
            rating = ctk.CTkOptionMenu(cell,values=['0','1','2','3','4','5'],width=40,height=20,font=ctk.CTkFont(size=10),command=lambda value,ph=photo:self._set_rating(ph,int(value)))
            rating.set(str(photo.rating))
            rating.pack(side='right',pady=3)
            self._gallery_controls.extend([button,check,rating])
            if self._busy in {'batch','scan'}:
                for widget in [button,check,rating]: widget.configure(state='disabled')
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
                    image.thumbnail((84,42),Image.Resampling.LANCZOS)
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
        self.start_button.configure(text=f'Експортирай {count} снимки')
        self._project_dirty = True

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
