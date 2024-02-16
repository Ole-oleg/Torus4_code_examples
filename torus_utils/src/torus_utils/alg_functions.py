import numpy as np
from .basic import get_branch_points


def Xi_raw(x: np.ndarray, K: complex):
    """
    Function Xi up to the sign of the square root (+ is chosen)
    """
    a = 3 / 2 * K**2 - 6 + x + 1 / x
    y = (-a - np.sqrt(a**2 - 4 * (2 + x + 1.0 / x))) / (2 * (1 + 1 / x))

    return y


def Yps_raw(x: np.ndarray, K: complex) -> np.ndarray:
    eta_11, eta_12, eta_21, eta_22 = get_branch_points(K)
    yps = np.sqrt((x - eta_11) * (x - eta_12) * (x - eta_21) * (x - eta_22))

    return yps


def refine_Yps(yps_raw: np.ndarray) -> np.ndarray:
    yps_ref = yps_raw.copy()

    for n in range(1, len(yps_raw)):
        if abs(yps_ref[n - 1] + yps_ref[n]) < abs(yps_ref[n - 1] - yps_ref[n]):
            yps_ref[n] = -yps_ref[n]

    return yps_ref


def Y(x: np.ndarray, Yx: complex, K: complex) -> np.ndarray:
    eta_11, eta_12, eta_21, eta_22 = get_branch_points(K)

    value: np.ndarray = np.sqrt(
        (x - eta_11) * (x - eta_12) * (x - eta_21) * (x - eta_22)
    )

    if abs(value[0] + Yx) < abs(value[0] - Yx):
        value[0] = -value[0]

    for n in range(1, len(value)):
        if abs(value[n] + value[n - 1]) < abs(value[n] - value[n - 1]):
            value[n] = -value[n]

    return value


def Yd(x: np.ndarray, Yx: complex, K: complex):
    """
    Derivative of Upsilon
    """
    eta_11, eta_12, eta_21, eta_22 = get_branch_points(K)
    value = (
        0.5
        * Yx
        * (1 / (x - eta_11) + 1 / (x - eta_12) + 1 / (x - eta_21) + 1 / (x - eta_22))
    )

    return value


def Ydd(x: np.ndarray, Yx: complex, K: complex):
    """
    Second derivative of Upsilon
    """
    eta_11, eta_12, eta_21, eta_22 = get_branch_points(K)

    value = 0.25 * Yx * (
        1 / (x - eta_11) + 1 / (x - eta_12) + 1 / (x - eta_21) + 1 / (x - eta_22)
    ) ** 2 - 0.5 * Yx * (
        1 / (x - eta_11) ** 2
        + 1 / (x - eta_12) ** 2
        + 1 / (x - eta_21) ** 2
        + 1 / (x - eta_22) ** 2
    )

    return value


def get_inidence_wave(phi: float, K: complex) -> tuple[complex, complex]:
    phi = phi - np.pi
    if phi > 0:
        Y_start = 1j
    else:
        Y_start = -1j

    dx = 0.000000000001
    ksi_1_0 = np.cos(phi)
    x = np.array([np.exp(1j * ksi_1_0 * K)])
    Yx = Y(x, Y_start, K)[0]

    for _ in range(1_000):
        nev_1 = nev(phi, x, Yx, K)
        nev_2 = nev(phi, x + dx, Yx, K)

        d_nev = (nev_2 - nev_1) / dx

        x = x - nev_1 / d_nev
        Yx = Y(x, Yx, K)[0]

    x_in = x[0]
    y_in = -(3 / 2 * K**2 * x_in - 6 * x_in + x_in**2 + 1) / 2 / (1 + x_in) + Y(
        x, Yx, K
    )[0] / 2 / (1 + x_in)

    return x_in, y_in


def Ksi(x: np.ndarray, Yx: complex, K: complex) -> np.ndarray:
    return -(3 / 2 * K**2 * x - 6 * x + x**2 + 1) / 2 / (1 + x) + Y(x, Yx, K) / 2 / (
        1 + x
    )


def Ksi_d(x: np.ndarray, Yx: complex, K: complex) -> np.ndarray:
    return (
        ((3 * K**2 * x) / 4 + x**2 / 2 - 3 * x + 1 / 2) / (x + 1) ** 2
        - ((3 * K**2) / 4 + x - 3) / (x + 1)
        + ((1 + x) * Yd(x, Yx, K) - Y(x, Yx, K)) / (2 * (1 + x) ** 2)
    )


def nev(phi: float, x: np.ndarray, Yx: complex, K: complex) -> np.ndarray:
    return (
        Ksi_d(x, Yx, K)
        + (np.sqrt(3) - np.tan(phi)) / 2 / np.tan(phi) * Ksi(x, Yx, K) / x
    )

def sheet_number(phi: float) -> int:
    number  = np.ceil(phi / np.pi) % 10

    if number == 0:
        number = 10

    return int(number)