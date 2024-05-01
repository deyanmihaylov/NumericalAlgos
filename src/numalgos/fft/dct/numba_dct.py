import numba as nb
import numpy as np

@nb.jit(
    nb.float64[:](nb.float64[:]),
    nopython=True,
    parallel=True,
    fastmath=True,
)
def dct1d(x):
    n = x.size

    y = np.zeros((n), dtype=np.float64)
    ks = np.arange(0, n)

    y = x[0] + (-1)**ks * x[n-1]
    for k in range(n):
        # y[k] = x[0] + (-1)**k * x[n-1]
        for i in range(1, n-1):
            y[k] += 2 * x[i] * np.cos(np.pi * k * i / (n - 1))
        # y[k] = x[0] + (-1)**k * x[n-1] + 2 * np.sum(np.array([x[i] * np.cos(np.pi * k * i / (n - 1)) for i in range(1, n-1)]))
        y[k] /= np.sqrt(2 * (n - 1))
    return y

@nb.jit(
    nb.float64[:](nb.float64[:]),
    nopython=True,
    # parallel=True,
    fastmath=True,
)
def dct1d_new(x):
    n = x.size
    yy = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            yy[i, j] = np.cos(np.pi * i * j / (n-1))
    y = np.sum(yy, axis=0)
    return y