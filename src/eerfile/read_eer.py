import os

import numpy as np
from tifffile import TiffFile


def read(filename: os.PathLike) -> np.ndarray:
    with TiffFile(filename) as tiff:
        eer_frames = tiff.asarray()
    return eer_frames
