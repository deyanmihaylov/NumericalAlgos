import cython
import numpy as np
cimport numpy as np

from libc.stdio cimport printf


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
@cython.wraparound(True)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void calc_spline_params(
    double[:] x,
    double[:] y,
    double[:] a_final,
    double[:] b_final,
    double[:] c_final,
    double[:] d_final,
):
    cdef:
        int n = x.size - 1
        double[:] a = np.zeros(n+1)
        double[:] b = np.zeros(n)
        double[:] c = np.zeros(n+1)
        double[:] d = np.zeros(n)
        double[:] h = np.zeros(n)
        double[:] h1 = np.zeros(n-1)
        double[:] h2 = np.zeros(n-1)
        double[:] h3 = np.zeros(n-1)
        double[:] f = np.zeros(n-1)
        int i

    for i in range(n+1):
        a[i] = y[i]

    for i in range(n):
        h[i] = x[i+1] - x[i]

    for i in range(n-1):
        h1[i] = h[i]
        h2[i] = 2 * (h[i+1] + h[i])
        h3[i] = h[i+1]
        f[i] = ((a[i+2] - a[i+1]) / h[i+1] - (a[i+1] - a[i]) / h[i]) * 3

    cdef double[:] z = tri_diag_solve(h1, h2, h3, f)

    c[0] = 0.
    c[n] = 0.
    
    for i in range(1, n):
        c[i] = z[i-1]

    for i in range(n):
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    for i in range(n):
        b[i] = (a[i+1] - a[i]) / h[i] + (2 * c[i+1] + c[i]) / 3 * h[i]

    for i in range(n):
        a_final[i] = a[i+1]
        b_final[i] = b[i]
        c_final[i] = c[i+1]
        d_final[i] = d[i]

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef double[:] tri_diag_solve(
    double[:] A,
    double[:] B,
    double[:] C,
    double[:] F,
):
    cdef:
        int n = B.size
        int i
        double[:] Bs = np.zeros(n)
        double[:] Fs = np.zeros(n)
        double[:] x = np.zeros(n)

    Bs[0] = B[0]
    Fs[0] = F[0]

    for i in range(1, n):
        Bs[i] = B[i] - A[i] / Bs[i - 1] * C[i - 1]
        Fs[i] = F[i] - A[i] / Bs[i - 1] * Fs[i - 1]

    x[n-1] = Fs[n-1] / Bs[n-1]

    for i in range(n - 2, -1, -1):
        x[i] = (Fs[i] - C[i] * x[i + 1]) / Bs[i]
    
    return x

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
