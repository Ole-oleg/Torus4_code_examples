import numpy as np
from .basic import get_branch_points
from .alg_functions import Yps_raw, refine_Yps, Y, Yd, Ydd


def abelian_integral(x, K, yps_0) -> complex:
    """
    computes the value of Abelian integral along the contour x
    the contour should not pass the branch points
    the abelian differential is dx / Yps(x)
    """

    dx = x[1:] - x[:-1]

    yps_raw = Yps_raw(x, K)
    yps_raw[0] = yps_0

    yps = refine_Yps(yps_raw)

    yp_med = 0.5 * (yps[1:] + yps[:-1])

    res = np.sum(dx / yp_med)

    return res


def solve_abelian_ode(
    T, K, num_points
) -> tuple[np.ndarray, np.ndarray, complex, np.ndarray]:
    """
    Solves equation dx = Yps(x) d chi
    starting from x = eta_21.
    chi is the Abelian variable.
    -----------
    Inputs:
    T is the the change of chi
    K is the parameter of the problem
    num_points is the number of points into which the segment [0, chi] is
    split
    (x , Yx) is the pair (x, Yps(x)) at the end point
    Yps_cont is the array of Yps corresponding to cont
    """
    eta_11, eta_12, eta_21, eta_22 = get_branch_points(K)

    chi = np.linspace(0, T, num_points)
    s0 = 0.5 * np.sqrt((eta_21 - eta_11) * (eta_21 - eta_12) * (eta_21 - eta_22))

    tau = 0 * chi
    s = 0 * chi
    s[0] = s0

    for n in range(1, len(chi)):
        t2 = tau[n - 1] ** 2
        s[n] = 0.5 * np.sqrt(
            (t2 + eta_21 - eta_11) * (t2 + eta_21 - eta_12) * (t2 + eta_21 - eta_22)
        )

        if abs(s[n] + s[n - 1]) < abs(s[n] - s[n - 1]):
            s[n] = -s[n]

        tau[n] = tau[n - 1] + s[n] * (chi[n] - chi[n - 1])

    Yps = 2 * tau * s
    cont = tau**2 + eta_21

    x = cont[-1]
    Yx = Yps[-1]

    Yps_cont = Yps

    return cont, x, Yx, Yps_cont


def refine(
    b0: complex, c0: complex, Yb0: complex, Yc0: complex, eta21: complex, K: complex
):
    """
    We refine the position of the points of division of the circle in 5 parts
    using the algebraic equations.
    """

    b_prev, Yb_prev = b0, Yb0
    c_prev, Yc_prev = c0, Yc0
    db = dc = 0.001

    for _ in range(20):
        nev_1, nev_2 = _find_nev(b_prev, c_prev, Yb_prev, Yc_prev, eta21, K)

        nev_1_b, nev_2_b = _find_nev(b_prev + db, c_prev, Yb_prev, Yc_prev, eta21, K)
        nev_1_c, nev_2_c = _find_nev(b_prev, c_prev + dc, Yb_prev, Yc_prev, eta21, K)

        d_nev_1_b = (nev_1_b - nev_1) / db
        d_nev_2_b = (nev_2_b - nev_2) / db

        d_nev_1_c = (nev_1_c - nev_1) / dc
        d_nev_2_c = (nev_2_c - nev_2) / dc

        mat = np.row_stack(
            (
                np.column_stack((d_nev_1_b, d_nev_1_c)),
                np.column_stack((d_nev_2_b, d_nev_2_c)),
            )
        )

        vec = np.row_stack((nev_1, nev_2))

        popravki = np.linalg.solve(mat, vec)

        b = b_prev - popravki[0]
        c = c_prev - popravki[1]

        b_prev = b
        c_prev = c

        Yb_prev = Y(b, Yb_prev, K)[0]
        Yc_prev = Y(c, Yc_prev, K)[0]

    return b, c


def _find_nev(
    b: complex,
    c: complex,
    Yb_prev: complex,
    Yc_prev: complex,
    eta21: complex,
    K: complex,
):
    b_prev = np.array([b], dtype=np.complex_)
    c_prev = np.array([c], dtype=np.complex_)

    nev1 = (
        -Y(c_prev, Yc_prev, K) / (c_prev - b_prev) ** 2
        - Y(b_prev, Yb_prev, K) / (c_prev - b_prev) ** 2
        - Yd(b_prev, Yb_prev, K) / (c_prev - b_prev)
        + Y(b_prev, Yb_prev, K) / (eta21 - b_prev) ** 2
        + Yd(b_prev, Yb_prev, K) / (eta21 - b_prev)
    )

    nev2 = (
        Y(b_prev, Yb_prev, K) / (b_prev - c_prev) ** 2
        + Y(c_prev, Yc_prev, K) / (b_prev - c_prev) ** 2
        + Yd(c_prev, Yc_prev, K) / (b_prev - c_prev)
        + Ydd(c_prev, Yc_prev, K) / 2
    )

    return nev1[0], nev2[0]


def refine_5th(arr: np.ndarray) -> np.ndarray:

    xi = np.exp(2j * np.pi / 5)

    outvar = arr

    for n in range(1, len(arr)):
        d0 = abs(outvar[n-1] - arr[n] )
        d1 = abs(outvar[n-1] - arr[n] * xi)
        d2 = abs(outvar[n-1] - arr[n] * xi**2)
        d3 = abs(outvar[n-1] - arr[n] * xi**3)
        d4 = abs(outvar[n-1] - arr[n] * xi**4)

        vals = [d0, d1, d2, d3, d4]
        idx = vals.index(min(vals))
        
        outvar[n] = arr[n] * xi**idx
        
    return outvar