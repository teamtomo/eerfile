"""Render EER files into a numpy array."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
from typing import Optional

import numpy as np

from eerfile.data_models import EERHeader
from eerfile.read_eer import read as read_eer


def render(
    file: os.PathLike,
    dose_per_output_frame: float,
    total_fluence: Optional[float] = None,
) -> np.ndarray:
    """
    Render EER files into a numpy array.

    Parameters
    ----------
    file: os.PathLike
        File containing image data in EER format.
    dose_per_output_frame: float
        Dose per output frame in rendered image in electrons per square angstrom.
    total_fluence: Optional[float]
        Total fluence in electrons per square angstrom.

    Returns
    -------
    image: np.ndarray
        `(b, h, w)` rendered image.

    """
    # grab header data and raw frames as numpy array
    eer_header = EERHeader.from_file(file)
    eer_frames = read_eer(file)

    # calculate number of output frames from target dose per output frame
    if total_fluence is None:
        dose_per_eer_frame = eer_header.dose_per_frame_electrons_per_square_angstrom
    else:
        dose_per_eer_frame = total_fluence / eer_header.n_frames
    print(f"dose_per_eer_frame: {dose_per_eer_frame}")
    eer_frames_per_output_frame = round(dose_per_output_frame / dose_per_eer_frame)
    n_output_frames = floor(eer_header.n_frames / eer_frames_per_output_frame)

    # allocate output array
    h, w = (
        eer_header.image_height_pixels,
        eer_header.image_width_pixels,
    )
    image = np.empty(shape=(n_output_frames, h, w), dtype=np.uint16)

    # define a closure to render a single frame from raw eer frames
    def _render_frame(i: int) -> np.ndarray:
        first = i * eer_frames_per_output_frame
        last = first + eer_frames_per_output_frame
        last = min(last, eer_header.n_frames - 1)
        return eer_frames[first:last].sum(axis=0, dtype=image.dtype)

    # submits tasks for each frame to be rendered
    with ThreadPoolExecutor() as executor:
        # keep track of frame index for each task
        future_to_idx = {
            executor.submit(_render_frame, i): i for i in range(n_output_frames)
        }

    # insert results into output array as they complete
    futures = list(future_to_idx.keys())
    for future in as_completed(futures):
        image[future_to_idx[future]] = future.result()

    return image
