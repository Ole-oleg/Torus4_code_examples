import numpy as np
from .alg_functions import Y, Yd, Ydd


def count_period_transactions(cont):
    c1 = cont[:-1]
    c2 = cont[1:]

    an = np.angle(c2 / c1)

    return sum(an) / (2 * np.pi)


class MFunc:
    def __init__(
        self,
        K: complex,
        eta_21: complex,
        b0: complex,
        Yb0: complex,
        c0: complex,
        Yc0: complex,
    ) -> None:
        self.K = K
        self.eta_21 = eta_21
        self.b0 = b0
        self.Yb0 = Yb0
        self.c0 = c0
        self.Yc0 = Yc0

    def M1(self, x: np.ndarray) -> np.ndarray:
        return (x - self.b0) / (x - self.eta_21)

    def M2(self, x: np.ndarray) -> np.ndarray:
        return (x - self.c0) / (x - self.eta_21)

    def M3(
        self,
        x: np.ndarray,
        Y_x: complex,
        b0: complex,
        Yb0: complex,
    ) -> np.ndarray:
        b = np.array([b0]).flatten()

        result = (
            Y_x / (x - b) ** 2
            + Y(b, Yb0, self.K) / (x - b) ** 2
            + Yd(b, Yb0, self.K) / (x - b)
            - Y(b, Yb0, self.K) / (self.eta_21 - b) ** 2
            - Yd(b, Yb0, self.K) / (self.eta_21 - b)
        )

        return result

    def M4(
        self,
        x: np.ndarray,
        Y_x: complex,
        b0: complex,
        Yb0: complex,
    ) -> np.ndarray:
        b = np.array([b0]).flatten()

        result = (
            Y_x / (x - b) ** 2
            + Y(b, Yb0, self.K) / (x - b) ** 2
            + Yd(b, Yb0, self.K) / (x - b)
            + Ydd(b, Yb0, self.K) / 2
        )

        return result

    def G1(self, x: np.ndarray, Yx: complex) -> np.ndarray:
        return (
            self.M1(x)
            * self.M2(x)
            / (self.M3(x, Yx, self.b0, self.Yb0) * self.M4(x, Yx, self.b0, self.Yb0))
        )

    def G2(self, x: np.ndarray, Yx: complex) -> np.ndarray:
        return (
            self.M1(x)
            * self.M2(x)
            / (self.M3(x, Yx, self.c0, self.Yc0) * self.M4(x, Yx, self.c0, self.Yc0))
        )

    def G3(self, x: np.ndarray, Yx: complex) -> np.ndarray:
        return (
            self.M1(x)
            * self.M2(x)
            / (self.M3(x, -Yx, self.c0, self.Yc0) * self.M4(x, -Yx, self.c0, self.Yc0))
        )

    def G4(self, x: np.ndarray, Yx: complex) -> np.ndarray:
        return (
            self.M1(x)
            * self.M2(x)
            / (self.M3(x, -Yx, self.b0, self.Yb0) * self.M4(x, -Yx, self.b0, self.Yb0))
        )
