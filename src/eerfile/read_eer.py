"""Read EER files into a numpy array."""

import os
from typing import Optional

import numpy as np
from tifffile import TiffFile


def read(filename: os.PathLike) -> np.ndarray:
    """
    Read all frames from an EER file.

    Parameters
    ----------
    filename: os.PathLike
        Path to the EER file

    Returns
    -------
    eer_frames: np.ndarray
        Array of all EER frames
    """
    with TiffFile(filename) as tiff:
        eer_frames = tiff.asarray()
    return eer_frames


def read_frames(
    filename: os.PathLike, start_frame: int, end_frame: Optional[int] = None
) -> np.ndarray:
    """
    Read a specific range of frames from an EER file.

    Parameters
    ----------
    filename: os.PathLike
        Path to the EER file
    start_frame: int
        Starting frame index (0-based)
    end_frame: Optional[int]
        Ending frame index (exclusive, 0-based). If None, reads to the end.

    Returns
    -------
    eer_frames: np.ndarray
        Array of EER frames for the specified range
    """
    with TiffFile(filename) as tiff:
        if end_frame is None:
            eer_frames = tiff.asarray(key=slice(start_frame, None))
        else:
            eer_frames = tiff.asarray(key=slice(start_frame, end_frame))
    return eer_frames
