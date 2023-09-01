import numpy as np
import numba
from numba.experimental import jitclass
from numba import float64


spec = [
    ('x0', float64[:]),
    ('a', float64[:]),
    ('b', float64[:]),
    ('c', float64[:]),
    ('d', float64[:]),
]

@jitclass(spec)
class CubicSpline(object):
    def __init__(self, x0, y0):
        self.x0 = x0
        self.a, self.b, self.c, self.d = calc_spline_params(x0, y0)

    def eval(self, x):
        return piece_wise_spline(x, self.x0, self.a, self.b, self.c, self.d)


"""
Calculate the parameters a, b, c, d of a natural cubic spline
"""
@numba.njit(
    # "UniTuple(f8[:], 4)(f8[:], f8[:])",
    cache = True,
    fastmath = True,
)
def calc_spline_params(
    x,
    y,
):
    n = x.size - 1
    a = y.copy()
    h = x[1:] - x[:-1]
    alpha = 3 * ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1])
    c = np.zeros(n+1)
    ell, mu, z = np.ones(n+1), np.zeros(n), np.zeros(n+1)
    for i in range(1, n):
        ell[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / ell[i]
        z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / ell[i]
    for i in range(n-1, -1, -1):
        c[i] = z[i] - mu[i] * c[i+1]
    b = (a[1:] - a[:-1]) / h + (c[:-1] + 2 * c[1:]) * h / 3
    d = np.diff(c) / (3 * h)
    return a[1:], b, c[1:], d
    
"""    
Calculate the value of a cubic spline value
"""
@numba.njit(
    # "f8[:](f8[:], i8[:], f8[:], f8[:], f8[:], f8[:], f8[:])",
    cache = True,
    fastmath = True,
)
def func_spline(
    x,
    ix,
    x0,
    a,
    b,
    c,
    d,
):
    dx = x - x0[1:][ix]
    return a[ix] + (b[ix] + (c[ix] + d[ix] * dx) * dx) * dx

@numba.njit(
    # 'i8[:](f8[:], f8[:], b1)',
    cache = True,
    fastmath = True,
)
def searchsorted_merge(
    a,
    b,
    sort_b,
):
    idx = np.zeros((len(b),), dtype = np.int64)
    if sort_b:
        ib = np.argsort(b)
    pa, pb = 0, 0
    while pb < len(b):
        if pa < len(a) and a[pa] < (b[ib[pb]] if sort_b else b[pb]):
            pa += 1
        else:
            idx[pb] = pa
            pb += 1
    return idx
    
"""
Compute piece-wise spline function for "x" out of sorted "x0" points
"""
@numba.njit(
    # 'f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])',
    cache = True,
    fastmath = True,
)
def piece_wise_spline(
    x,
    x0,
    a,
    b,
    c,
    d,
):
    x = np.asarray(x)
    xsh = x.shape
    x = x.ravel()
    ix = searchsorted_merge(x0[1 : -1], x, True)
    y = func_spline(x, ix, x0, a, b, c, d)
    y = np.ascontiguousarray(y).reshape(xsh)
    return y

"""
Generate a cubic spline interpolator
"""
def cubic_spline(
    x0,
    y0,
):
    a, b, c, d = calc_spline_params(x0, y0)

    @numba.njit()
    def f(x):
        return piece_wise_spline(x, x0, a, b, c, d)

    return f


if __name__ == '__main__':
    pass