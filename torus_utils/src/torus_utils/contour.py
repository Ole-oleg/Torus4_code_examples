from typing import NamedTuple
from math import floor
import numpy as np


class Segment(NamedTuple):
    start: complex
    end: complex
    seg_length: int


def make_contour(points: tuple[complex, ...], dx: float = 1e-05) -> np.ndarray:
    """
    Builds polygonal chain contours.
    Input:
    - points: end points of line segments for polygonal chain
    - dx: discterization step (default value = 1e-05)
    """
    contour_lenght = 0
    segment_params = []

    for n in range(len(points) - 1):
        start, end = points[n], points[n + 1]
        segment_length = floor(abs(end - start) / dx) + 2
        segment_params.append(Segment(start, end, segment_length))

        contour_lenght += segment_length

    contour = np.zeros(contour_lenght, dtype=np.complex_)
    start_idx = 0
    for segment in segment_params:
        contour[start_idx : start_idx + segment.seg_length] = np.linspace(
            segment.start, segment.end, segment.seg_length, dtype=np.complex_
        )

        start_idx += segment.seg_length

    return contour
