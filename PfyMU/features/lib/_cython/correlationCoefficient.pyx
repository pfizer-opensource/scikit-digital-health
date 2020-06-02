# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as npy_double
from libc.math cimport sqrt
from .common cimport mean_1d


def CC3(const double[:, :, :] x):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], i, j, k
    r_ = zeros((m, 3), dtype=npy_double)

    cdef double[:, :] r = r_
    cdef double[3] mns = [0.0, 0.0, 0.0], top = [0.0, 0.0, 0.0]
    cdef double[3] dfs = [0.0, 0.0, 0.0], d2s = [0.0, 0.0, 0.0]

    for i in range(m):
        # initialize/reset each loop
        mns = [0.0, 0.0, 0.0]
        top = [0.0, 0.0, 0.0]
        dfs = [0.0, 0.0, 0.0]
        d2s = [0.0, 0.0, 0.0]

        for k in range(3):
            mean_1d(x[i, :, k], &mns[k])
        
        for j in range(n):
            for k in range(3):
                dfs[k] = x[i, j, k] - mns[k]
                d2s[k] += dfs[k]**2
            
            top[0] += (dfs[0] * dfs[1])
            top[1] += (dfs[0] * dfs[2])
            top[2] += (dfs[1] * dfs[2])
        
        r[i, 0] = top[0] / (sqrt(d2s[0] * d2s[1]))
        r[i, 1] = top[1] / (sqrt(d2s[0] * d2s[2]))
        r[i, 2] = top[2] / (sqrt(d2s[1] * d2s[2]))

    return r_
        
