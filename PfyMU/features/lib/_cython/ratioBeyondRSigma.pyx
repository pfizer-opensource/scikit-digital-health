# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as npy_double
from libc.math cimport sqrt, pow, abs
from PfyMU.features.lib._cython.common cimport mean_sd_1d


def cy_RatioBeyondRSigma(const double[:, :, :] x, double r):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, j, k
    xcount = zeros((m, p), dtype=npy_double, order='C')
    cdef double[:, ::1] count = xcount
    cdef double mu = 0., sigma = 0.

    for i in range(m):
        for k in range(p):
            mu = 0.
            sigma = 0.
            mean_sd_1d(x[i, :, k], &mu, &sigma)
            sigma *= r
            for j in range(n):
                if (abs(x[i, j, k] - mu) > sigma):
                    count[i, k] += 1
    for i in range(m):
        for j in range(p):
            count[i, j] /= n
    return xcount