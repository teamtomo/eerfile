"""Render EER files into a numpy array."""

import gc
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
from typing import Optional

import numpy as np

from eerfile.data_models import EERHeader
from eerfile.read_eer import read_frames


def render(
    file: os.PathLike,
    dose_per_output_frame: float,
    total_fluence: Optional[float] = None,
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
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
        If not specified, the fluence from the EER file header will be used.
    chunk_size: Optional[int]
        Number of output frames to process in each chunk. If None, processes
        all frames at once (original behavior). Specify to reduce memory usage.
    max_workers: Optional[int]
        Maximum number of worker threads.
        If None, uses default ThreadPoolExecutor behavior.

    Returns
    -------
    image: np.ndarray
        `(b, h, w)` rendered image.

    """
    # grab header data
    eer_header = EERHeader.from_file(file)

    # calculate number of output frames from target dose per output frame
    if total_fluence is None:
        dose_per_eer_frame = eer_header.dose_per_frame_electrons_per_square_angstrom
    else:
        dose_per_eer_frame = total_fluence / eer_header.n_frames

    eer_frames_per_output_frame = round(dose_per_output_frame / dose_per_eer_frame)
    n_output_frames = floor(eer_header.n_frames / eer_frames_per_output_frame)

    # allocate output array
    h, w = (
        eer_header.image_height_pixels,
        eer_header.image_width_pixels,
    )
    image = np.empty(shape=(n_output_frames, h, w), dtype=np.uint16)

    # process frames - either all at once (default) or in chunks if specified
    if chunk_size is None:
        chunk_size = n_output_frames  # Process all frames at once (original behavior)

    for chunk_start in range(0, n_output_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_output_frames)

        try:
            # load only the EER frames needed for this chunk
            first_eer_frame = chunk_start * eer_frames_per_output_frame
            last_eer_frame = min(
                (chunk_end - 1) * eer_frames_per_output_frame
                + eer_frames_per_output_frame,
                eer_header.n_frames,
            )

            # Load only the slice of frames we need for this chunk
            eer_chunk = read_frames(file, first_eer_frame, last_eer_frame)

            # process frames in this chunk with threading
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # keep track of frame index for each task
                future_to_idx = {
                    executor.submit(
                        _render_frame,
                        i,
                        chunk_start,
                        eer_frames_per_output_frame,
                        eer_chunk,
                        image.dtype,
                    ): i
                    for i in range(chunk_start, chunk_end)
                }

                # insert results into output array as they complete
                for future in as_completed(future_to_idx):
                    try:
                        image[future_to_idx[future]] = future.result()
                    except Exception as e:
                        logging.error(
                            f"Error processing frame {future_to_idx[future]}: {e}"
                        )
                        raise

            # explicitly delete chunk and force garbage collection
            del eer_chunk
            gc.collect()

        except MemoryError as e:
            logging.error(
                f"Memory error processing chunk {chunk_start//chunk_size + 1}. "
                f"Try reducing chunk_size."
            )
            raise MemoryError(
                f"Insufficient memory to process chunk. "
                f"Try calling render() with a smaller chunk_size parameter. "
                f"Current chunk_size: {chunk_size}, suggested: {chunk_size // 2}"
            ) from e
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_start//chunk_size + 1}: {e}")
            raise

    return image


def _render_frame(
    i: int,
    chunk_start: int,
    eer_frames_per_output_frame: int,
    eer_chunk: np.ndarray,
    output_dtype: np.dtype,
) -> np.ndarray:
    """
    Render a single frame from a chunk of EER frames.

    Parameters
    ----------
    i: int
        Frame index in the overall output
    chunk_start: int
        Starting frame index of the current chunk
    eer_frames_per_output_frame: int
        Number of EER frames to combine into one output frame
    eer_chunk: np.ndarray
        Chunk of EER frames to process
    output_dtype: np.dtype
        Data type for the output frame

    Returns
    -------
    frame: np.ndarray
        Rendered frame
    """
    chunk_relative_i = i - chunk_start
    first = chunk_relative_i * eer_frames_per_output_frame
    last = first + eer_frames_per_output_frame
    last = min(last, eer_chunk.shape[0])
    return eer_chunk[first:last].sum(axis=0, dtype=output_dtype)
