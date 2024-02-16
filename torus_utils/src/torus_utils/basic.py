from typing import Tuple
import numpy as np


def get_branch_points(K: complex) -> Tuple[complex, complex, complex, complex]:
    """
    Function for calculating branch points of Riemann surface R_2
    using parameter K
    """
    const_1 = -(3 / 2) * K**2 + 8 - 6 * (np.sqrt(1 - K**2 / 6))
    const_2 = -(3 / 2) * K**2 + 8 + 6 * (np.sqrt(1 - K**2 / 6))

    eta_11 = 1 / 2 * (const_1 - np.sqrt(const_1**2 - 4))
    eta_12 = 1 / 2 * (const_2 + np.sqrt(const_2**2 - 4))
    eta_21 = 1 / 2 * (const_1 + np.sqrt(const_1**2 - 4))
    eta_22 = 1 / 2 * (const_2 - np.sqrt(const_2**2 - 4))

    if eta_21.imag < 0:
        eta_11 = 1 / eta_11
        eta_21 = 1 / eta_21

    if abs(eta_12) < 1:
        eta_12 = 1 / eta_12
        eta_22 = 1 / eta_22

    return (eta_11, eta_12, eta_21, eta_22)
