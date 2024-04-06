import cython
import numpy as np
cimport numpy as np


cdef class CubicSpline:
    def __cinit__(self, x0, y0):
        self.x0 = x0
        cdef int n = x0.size - 1
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.c = np.zeros(n)
        self.d = np.zeros(n)
        calc_spline_params(x0, y0, self.a, self.b, self.c, self.d)
    def __call__(self, x):
        return piece_wise_spline(x, self.x0, self.a, self.b, self.c, self.d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void calc_spline_params(
    double[:] x,
    double[:] y,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
):
    cdef:
        int n = x.size - 1
        double[:] h = np.empty(n)
        double[:] alpha = np.empty(n-1)
        double[:] ell = np.ones(n+1)
        double[:] mu = np.empty(n)
        double[:] z = np.empty(n+1)
        int i

    for i in range(n+1):
        if i < n: a[i] = y[i+1]
        h[i] = x[i+1] - x[i]
        if i != 0:
            alpha[i-1] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        if i > 0 and i < n:
            ell[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / ell[i]
            z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / ell[i]

    for i in range(n-1, -1, -1):
        c[i-1] = z[i] - mu[i] * c[i]

    for i in range(n):
        if i == 0:
            b[0] = (y[i+1] - y[i]) / h[i] + 2 * c[i] * h[i] / 3
            d[0] = c[i] / (3 * h[i])
        else:
            b[i] = (y[i+1] - y[i]) / h[i] + (c[i-1] + 2 * c[i]) * h[i] / 3
            d[i] = (c[i] - c[i-1]) / (3 * h[i])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void func_spline(
    double[:] x,
    long[:] ix,
    double[:] x0,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
    double[:] y,
):
    cdef:
        int i
        double dx

    for i in range(x.size):
        dx = x[i] - x0[ix[i] + 1]
        y[i] = a[ix[i]] + (b[ix[i]] + (c[ix[i]] + d[ix[i]] * dx) * dx) * dx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void searchsorted_merge(
    double[:] a,
    double[:] b,
    bint sort_b,
    long[:] idx,
):
    cdef:
        long[:] ib
        long pa = 0
        long pb = 0
        long len_a = a.size
    if sort_b:
        ib = np.argsort(b)

    while pb < b.size:
        if pa < len_a and a[pa] < (b[ib[pb]] if sort_b else b[pb]):
            pa += 1
        else:
            if sort_b:
                idx[ib[pb]] = pa
            else:
                idx[pb] = pa
            pb += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef double[:] piece_wise_spline(
    double[:] x,
    double[:] x0,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
):
    cdef:
        int n = x.size
        long[:] ix = np.empty(n, dtype=np.int64)
        double[:] y = np.empty(n)

    searchsorted_merge(x0[1 : -1], x, True, ix)
    func_spline(x, ix, x0, a, b, c, d, y)
    return y
