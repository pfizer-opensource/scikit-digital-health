# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as np_double


def RangeCount(const double[:, :, :] x, double xmin, double xmax):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, j, k
    xcount = zeros((m, p), dtype=np_double)
    cdef double[:, :] count = xcount

    for i in range(m):
        for j in range(n):
            for k in range(p):
                if (xmin <= x[i, j, k] < xmax):
                    count[i, k] += 1.
    for i in range(m):
        for k in range(p):
            count[i, k] /= n
    return xcount