import cython
import numpy as np
cimport numpy as np


cdef class CubicSpline:
    def __cinit__(
        self,
        double[:] x0,
        double[:] y0,
        bc_type="not-a-knot",
    ):
        self.x0 = x0
        self.n = x0.size
        self.a = np.empty(self.n-1)
        self.b = np.empty(self.n-1)
        self.c = np.empty(self.n-1)
        self.d = np.empty(self.n-1)

        if bc_type == "not-a-knot":
            compute_spline_params_not_a_knot(
                x0, y0, self.n,
                self.a, self.b, self.c, self.d,
            )
        elif bc_type == "periodic":
            compute_spline_params_periodic(
                x0, y0, self.n,
                self.a, self.b, self.c, self.d,
            )
        else:
            if bc_type == "clamped":
                self.type_start = 1
                self.type_end = 1
                self.val_start = 0.
                self.val_end = 0.
            elif bc_type == "natural":
                self.type_start = 2
                self.type_end = 2
                self.val_start = 0.
                self.val_end = 0.
            else:
                self.type_start = bc_type[0][0]
                self.type_end = bc_type[1][0]
                self.val_start = bc_type[0][1]
                self.val_end = bc_type[1][1]
            
            compute_spline_params(
                x0, y0, self.n,
                self.type_start, self.val_start,
                self.type_end, self.val_end,
                self.a, self.b, self.c, self.d,
            )

    def __call__(self, x):
        scalar_flag = False
        if np.isscalar(x): scalar_flag = True
        y = np.array(piece_wise_spline(
            np.atleast_1d(x), self.x0, self.a, self.b, self.c, self.d,
        ))
        if scalar_flag:
            y = y[0]
        return y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void solve_tridiag_reduced(
    double[:] a,
    double[:] c,
    double[:] x,
    int n,
):
    cdef int i

    c[0] = c[0] / 2
    x[0] = x[0] / 2

    for i in range(1, n):
        if i < n-1:
            c[i] = c[i] / (2 - a[i-1] * c[i-1])
        x[i] = (x[i] - a[i-1] * x[i-1]) / (2 - a[i-1] * c[i-1])
    
    for i in range(n-2, -1, -1):
        x[i] -= c[i] * x[i + 1]

    for i in range(n):
        x[i] = x[i] / 2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void solve_tridiag_reduced_not_a_knot(
    double[:] a,
    double[:] c,
    double[:] x,
    int n,
    double diag_start,
    double diag_end,
):
    cdef int i

    c[0] = c[0] / diag_start
    x[1] = x[1] / diag_start

    for i in range(1, n-1):
        c[i] = c[i] / (2 - a[i-1] * c[i-1])
        x[i+1] = (x[i+1] - a[i-1] * x[i]) / (2 - a[i-1] * c[i-1])
    
    x[n] = (x[n] - a[n-2] * x[n-1]) / (diag_end - a[n-2] * c[n-2])
    
    for i in range(n-2, -1, -1):
        x[i+1] -= c[i] * x[i+2]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void solve_nearly_tridiagonal_reduced(
    double[:] a,
    double[:] c,
    double[:] x,
    int n,
):
    cdef:
        int i
        double expr1

    a[0] = a[0] / 2
    c[0] = c[0] / 2
    x[0] = x[0] / 2
    
    for i in range(1, n-2):
        expr1 = 2 - a[i] * c[i-1]
        c[i] = c[i] / expr1
        x[i] = (x[i] - a[i] * x[i-1]) / expr1
        a[i] = - a[i] * a[i-1] / expr1

    x[n-2] = (x[n-2] - a[n-2] * x[n-3]) / (2 - a[n-2] * c[n-3])
    a[n-2] = (c[n-2] - a[n-2] * a[n-3]) / (2 - a[n-2] * c[n-3])
    
    for i in range(n-3, -1, -1):
        x[i] -= c[i] * x[i+1]
        a[i] -= c[i] * a[i+1]

    x[n-1] = (
        x[n-1] - a[n-1] * x[n-2] - c[n-1] * x[0]
    ) / (2 - a[n-1] * a[n-2] - c[n-1] * a[0])

    for i in range(n-2, -1, -1):
        x[i] -= a[i] * x[n-1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void compute_spline_params(
    double[:] x,
    double[:] y,
    int n,
    int type_start,
    double value_start,
    int type_end,
    double value_end,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
):
    cdef:
        int i
        double[:] hx = np.empty(n-1)
        double hh
        double[:] μ = np.empty(n-1)
        double[:] λ = np.empty(n-1)

    for i in range(n-1):
        hx[i] = x[i+1] - x[i]

    for i in range(0, n-2):
        hh = x[i+2] - x[i]
        μ[i] = hx[i] / hh
        λ[i+1] = hx[i+1] / hh
        b[i+1] = 6 * (
            (y[i+2] - y[i+1]) / hx[i+1] - (y[i+1] - y[i]) / hx[i]
        ) / hh

    if type_start == 1:
        λ[0] = 1.
        b[0] = 6 * ((y[1] - y[0]) / hx[0] - value_start) / hx[0]
    elif type_start == 2:
        λ[0] = 0.
        b[0] = 2 * value_start

    if type_end == 1:
        μ[n-2] = 1.
        b[n-1] = 6 * (value_end - (y[n-1] - y[n-2]) / hx[n-2]) / hx[n-2]
    elif type_end == 2:
        μ[n-2] = 0.
        b[n-1] = 2 * value_end

    solve_tridiag_reduced(μ, λ, b, n)

    for i in range(n-1):
        a[i] = (b[i+1] - b[i]) / (3 * hx[i])
        c[i] = (y[i+1] - y[i]) / hx[i] - b[i+1] * hx[i] / 3 - b[i] * hx[i] / 1.5
        d[i] = y[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void compute_spline_params_not_a_knot(
    double[:] x,
    double[:] y,
    int n,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
):
    cdef:
        int i
        double[:] hx = np.empty(n-1)
        double hh, diag_start, diag_end
        double[:] μ = np.empty(n-3)
        double[:] λ = np.empty(n-3)

    for i in range(n-1):
        hx[i] = x[i+1] - x[i]

    for i in range(1, n-3):
        hh = x[i+2] - x[i]
        μ[i-1] = hx[i] / hh
        λ[i] = hx[i+1] / hh

    for i in range(0, n-2):
        b[i+1] = 6 * (
            (y[i+2] - y[i+1]) / hx[i+1] - (y[i+1] - y[i]) / hx[i]
        ) / (x[i+2] - x[i])

    λ[0] = 1 - hx[0] / hx[1]
    μ[n-4] = 1 - hx[n-2] / hx[n-3]
    diag_start = 2 + hx[0] / hx[1]
    diag_end = 2 + hx[n-2] / hx[n-3]

    solve_tridiag_reduced_not_a_knot(μ, λ, b, n-2, diag_start, diag_end)

    for i in range(1, n-1):
        b[i] = b[i] / 2
    
    b[0] = ((hx[0] + hx[1]) * b[1] - hx[0] * b[2]) / hx[1]
    b[n-1] = ((hx[n-3] + hx[n-2]) * b[n-2] - hx[n-2] * b[n-3]) / hx[n-3]

    for i in range(n-1):
        a[i] = (b[i+1] - b[i]) / (3 * hx[i])
        c[i] = (y[i+1] - y[i]) / hx[i] - b[i+1] * hx[i] / 3 - b[i] * hx[i] / 1.5
        d[i] = y[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void compute_spline_params_periodic(
    double[:] x,
    double[:] y,
    int n,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
):
    cdef:
        int i
        double[:] hx = np.empty(n-1)
        double hh
        double[:] μ = np.empty(n-1)
        double[:] λ = np.empty(n-1)

    for i in range(n-1):
        hx[i] = x[i+1] - x[i]

    for i in range(0, n-2):
        hh = x[i+2] - x[i]
        μ[i+1] = hx[i] / hh
        λ[i+1] = hx[i+1] / hh
        b[i+1] = 6 * (
            (y[i+2] - y[i+1]) / hx[i+1] - (y[i+1] - y[i]) / hx[i]
        ) / hh

    λ[0] = hx[0] / (hx[0] + hx[n-2])
    λ[n-2] = hx[n-2] / (hx[n-3] + hx[n-2])
    μ[0] = hx[n-2] / (hx[0] + hx[n-2])
    b[0] = 6 * (
        (y[1] - y[0]) / hx[0] - (y[n-1] - y[n-2]) / hx[n-2]
    ) / (hx[0] + hx[n-2])

    solve_nearly_tridiagonal_reduced(μ, λ, b, n-1)

    for i in range(n-1):
        b[i] = b[i] / 2

    for i in range(n-2):
        a[i] = (b[i+1] - b[i]) / (3 * hx[i])
        c[i] = (y[i+1] - y[i]) / hx[i] - b[i+1] * hx[i] / 3 - b[i] * hx[i] / 1.5
        d[i] = y[i]

    a[n-2] = (b[0] - b[n-2]) / (3 * hx[n-2])
    c[n-2] = (y[n-1] - y[n-2]) / hx[n-2] - b[0] * hx[n-2] / 3 - b[n-2] * hx[n-2] / 1.5
    d[n-2] = y[n-2]

@cython.boundscheck(False)
@cython.wraparound(True)
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
        double[:] h = np.zeros(n)
        double[:] alpha = np.empty(n-1)
        double[:] ell = np.ones(n+1)
        double[:] mu = np.zeros(n)
        double[:] z = np.zeros(n+1)
        int i

    for i in range(n+1):
        if i < n:
            a[i] = y[i+1]
        h[i] = x[i+1] - x[i]
        if i != 0:
            alpha[i-1] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        if i > 0 and i < n:
            ell[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / ell[i]
            z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / ell[i]

    c[n-1] = 0.
    for i in range(n-2, -1, -1):
        c[i] = z[i+1] - mu[i+1] * c[i+1]

    b[0] = (y[1] - y[0]) / h[0] + 2 * c[0] * h[0] / 3
    d[0] = (c[0] - z[0] + mu[0] * c[0]) / (3 * h[0])
    for i in range(1, n):
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
        dx = x[i] - x0[ix[i]]
        y[i] = d[ix[i]] + (c[ix[i]] + (b[ix[i]] + a[ix[i]] * dx) * dx) * dx

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

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void solve_tridiag(
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
    double[:] x,
):
    cdef:
        int n = d.size
        int i
        double[:] w = np.empty(n-1)
        double[:] g = np.empty(n)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n-1):
        w[i] = c[i] / (b[i] - a[i-1] * w[i-1])
    
    for i in range(1, n):
        g[i] = (d[i] - a[i-1] * g[i-1]) / (b[i] - a[i-1] * w[i-1])
    
    x[n-1] = g[n-1]

    for i in range(n-1, 0, -1):
        x[i-1] = g[i-1] - w[i-1] * x[i]

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cpdef void method1(
    double[:] x,
    double[:] y,
    double[:] a,
    double[:] b,
    double[:] c,
    double[:] d,
):
    cdef:
        int n = x.size - 1
        int n2 = 2 * n
        int n3 = 3 * n
        int n4 = 4 * n
        int i, j, k, m, p, q
        double[:, :] matrix = np.zeros((n4, n4), dtype=np.float64)
        double[:] rhs = np.zeros(n4, dtype=np.float64)
        double prev_xcb, prev_xsq, prev_x

    for i in range(n):
        j = 2 * i
        k = 4 * i
        m = j + 1

        if i == 0:
            matrix[0, 0] = x[0] * x[0] * x[0]
            matrix[0, 1] = x[0] * x[0]
            matrix[0, 2] = x[0]
        else:
            matrix[j, k] = prev_xcb
            matrix[j, k + 1] = prev_xsq
            matrix[j, k + 2] = prev_x
        
        matrix[j, k + 3] = 1.
        rhs[j] = y[i]

        prev_xcb = x[i+1] * x[i+1] * x[i+1]
        prev_xsq = x[i+1] * x[i+1]
        prev_x = x[i+1]

        matrix[m, k] = prev_xcb
        matrix[m, k + 1] = prev_xsq
        matrix[m, k + 2] = prev_x
        matrix[m, k + 3] = 1.
        rhs[m] = y[i+1]

        if i != n-1:
            p = n2 + i
            q = n3 - 1 + i

            matrix[p, k] = 3 * prev_xsq
            matrix[p, k + 1] = 2 * prev_x
            matrix[p, k + 2] = 1

            matrix[p, k + 4] = -3 * prev_xsq
            matrix[p, k + 5] = -2 * prev_x
            matrix[p, k + 6] = -1

            matrix[q, k] = 6 * prev_x
            matrix[q, k + 1] = 2

            matrix[q, k + 4] = -6 * prev_x
            matrix[q, k + 5] = -2

    matrix[n4 - 2, 0] = 6 * x[0]
    matrix[n4 - 2, 1] = 2

    matrix[n4 - 1, n4 - 4] = 6 * x[n]
    matrix[n4 - 1, n4 - 3] = 2

    coeffs = np.linalg.solve(matrix, rhs)

    for i in range(n):
        k = 4 * i
        a[i] = coeffs[k]
        b[i] = coeffs[k + 1]
        c[i] = coeffs[k + 2]
        d[i] = coeffs[k + 3]