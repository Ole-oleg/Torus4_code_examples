import numpy as np
from typing import Any
from ._rust_ext import refine

def refine_on_k5(x: np.ndarray[Any, np.dtype[np.complex_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
    """
    Refines a function on K5
    """
    return refine(x)