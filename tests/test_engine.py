from dataclasses import replace
import threading

import cv2
import numpy as np
import pytest

from raw_engine import RawEngine, ProcessingParams, ProcessingCancelled


@pytest.fixture
def engine():
    cv2.setNumThreads(1)
    return RawEngine()


def test_real_libraw_dng_decode_and_preview(engine, dng_path):
    params = ProcessingParams()
    decoded = engine.decode(dng_path, params)
    assert decoded.dtype == np.uint16
    assert decoded.shape[2] == 3
    assert decoded.max() > 0
    full = engine.preview(dng_path, params, draft=False)
    draft = engine.preview(dng_path, params, draft=True)
    assert full.before.size == full.after.size
    assert full.dimensions[0] > draft.dimensions[0]
    assert not full.draft and draft.draft


@pytest.mark.parametrize('white_balance', ['camera', 'auto', 'gray_world', 'shades_of_gray'])
def test_white_balance_modes_work_through_libraw(engine, dng_path, white_balance):
    rgb = engine.decode(dng_path, ProcessingParams(white_balance=white_balance))
    out = engine.apply_corrections(rgb, ProcessingParams(white_balance=white_balance))
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert 0 <= out.min() <= out.max() <= 1


def test_png16_export_keeps_more_than_eight_bit_precision(engine, dng_path, tmp_path):
    params = ProcessingParams(output_format='png', png_bit_depth=16)
    result = engine.process_file(dng_path, tmp_path / 'готово.png', params)
    saved = cv2.imdecode(np.fromfile(result.output_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert saved.dtype == np.uint16
    assert saved.shape[:2] == (result.dimensions[1], result.dimensions[0])
    assert np.any(saved % 257 != 0)  # Not an 8-bit image merely scaled to 16 bits.


@pytest.mark.parametrize('fmt,depth', [('jpg', 8), ('png', 8)])
def test_eight_bit_export_and_colour_order(engine, tmp_path, fmt, depth):
    srgb = np.zeros((40, 40, 3), np.float32)
    srgb[:, :, 0] = .9  # Red; a mistaken RGB/BGR swap would become blue.
    params = ProcessingParams(output_format=fmt, png_bit_depth=depth, quality=100)
    path = engine.save_image(srgb, tmp_path / f'red.{fmt}', params)
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert bgr.dtype == np.uint8
    assert bgr[:, :, 2].mean() > 220 and bgr[:, :, 0].mean() < 5


def test_denoise_and_sharpen_work_and_preserve_sub_eight_bit_values(engine, dng_path):
    params = ProcessingParams(denoise=True, sharpen=True)
    image = engine.apply_corrections(engine.decode(dng_path, params), params)
    assert np.isfinite(image).all()
    assert np.any(engine.to_uint16(image) % 257 != 0)


def test_neutral_and_black_images_do_not_generate_nan(engine):
    params = ProcessingParams(white_balance='gray_world', denoise=True)
    for value in [0, 12000, 65535]:
        out = engine.apply_corrections(np.full((48, 48, 3), value, np.uint16), params)
        assert np.isfinite(out).all() and 0 <= out.min() <= out.max() <= 1


def test_white_balance_reduces_neutral_cast(engine):
    rgb = np.full((64, 64, 3), [.12, .24, .36], np.float32)
    for mode in ['gray_world', 'shades_of_gray']:
        balanced = engine._white_balance(rgb, mode)
        means = balanced.mean(axis=(0, 1))
        assert means.max() - means.min() < 1e-4


def test_cancelled_export_has_no_file(engine, tmp_path):
    event = threading.Event()
    event.set()
    path = tmp_path / 'cancelled.png'
    with pytest.raises(ProcessingCancelled):
        engine.save_image(np.ones((16, 16, 3), np.float32), path,
                          ProcessingParams(output_format='png'), cancel=event)
    assert not path.exists()


def test_existing_output_requires_explicit_overwrite(engine, tmp_path):
    path = tmp_path / 'keep.jpg'
    path.write_bytes(b'original bytes')
    with pytest.raises(FileExistsError):
        engine.save_image(np.ones((16, 16, 3), np.float32), path, ProcessingParams())
    assert path.read_bytes() == b'original bytes'
    assert not list(tmp_path.glob('.raw-export-*'))


def test_failed_atomic_publish_cleans_up(engine, tmp_path, monkeypatch):
    import raw_engine
    path = tmp_path / 'new.jpg'
    def fail(*_):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(raw_engine.os, 'replace', fail)
    with pytest.raises(OSError):
        engine.save_image(np.ones((16, 16, 3), np.float32), path, ProcessingParams())
    assert not path.exists()
    assert not list(tmp_path.glob('.raw-export-*'))


@pytest.mark.parametrize('update', [
    {'quality': 0}, {'quality': 101}, {'quality': 50.5}, {'png_compression': 10},
    {'exposure_ev': float('nan')}, {'output_format': 'tiff'}, {'png_bit_depth': 12},
    {'white_balance': 'invalid'},
])
def test_reject_invalid_parameters(update):
    with pytest.raises(ValueError):
        replace(ProcessingParams(), **update)
