import numpy as np
from src.alg_functions import Y


def test_Y() -> None:

    x= np.array([1])
    Yx= 1j
    K= 0.5 + 1e-3j
    
    true_value = -0.0032 + 1.6910j

    value = Y(x, Yx, K)
    assert abs(value - true_value) < 1e-04
