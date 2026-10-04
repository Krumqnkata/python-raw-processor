"""Generate a small, deterministic Bayer DNG; no downloaded photos required."""
from pathlib import Path

import numpy as np
import pytest
import tifffile


def make_dng(path: Path, width: int = 512, height: int = 384) -> Path:
    y, x = np.mgrid[:height, :width]
    xx = x / (width - 1)
    yy = y / (height - 1)
    # Smooth gradient with colour patches and edges, useful for colour/layout checks.
    scene = np.stack((0.08 + .65 * xx, 0.10 + .60 * yy,
                      0.12 + .45 * (1 - xx)), axis=2)
    scene[height // 4:height // 2, width // 4:width // 2] = [.8, .1, .08]
    bayer = np.empty((height, width), dtype=np.uint16)
    for rows, cols, channel in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]:
        bayer[rows::2, cols::2] = np.rint(scene[rows::2, cols::2, channel] * 4000 + 64)
    # TIFF/DNG tags describe an uncompressed, RGGB, 12-bit sensor in a uint16 container.
    tags = [
        (50706, 'B', 4, (1, 4, 0, 0), False),
        (50707, 'B', 4, (1, 1, 0, 0), False),
        (50708, 's', 0, 'RAW Studio Synthetic Test Camera', False),
        (33421, 'H', 2, (2, 2), False),
        (33422, 'B', 4, (0, 1, 1, 2), False),
        (50713, 'H', 2, (1, 1), False),
        (50714, 'H', 1, 64, False),
        (50717, 'I', 1, 4095, False),
        (50721, '2i', 9, (10000, 10000, 0, 10000, 0, 10000,
                          0, 10000, 10000, 10000, 0, 10000,
                          0, 10000, 0, 10000, 10000, 10000), False),
        (50728, '2I', 3, (1, 1, 1, 1, 1, 1), False),
        (50778, 'H', 1, 21, False),
    ]
    tifffile.imwrite(path, bayer, photometric=32803, metadata=None,
                     extratags=tags, compression=None)
    return path


@pytest.fixture
def dng_path(tmp_path):
    return make_dng(tmp_path / 'synthetic.DNG')
