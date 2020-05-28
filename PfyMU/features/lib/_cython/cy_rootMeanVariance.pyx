# cython: infer_types = True
# cython: wraparound = False
# cython: boundscheck = False
cimport cython
from numpy import zeros, double as npy_double, array
from libc.math cimport sqrt

from PfyMU.features.lib._cython.common cimport mean_1d, variance_1d


cpdef cy_RootMeanVariance(const double[:, :, :] x):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, k
    cdef double[:] var = zeros(p, dtype=npy_double)

    activity = zeros(m, dtype=npy_double)
    cdef double[:] act_idx = activity

    for i in range(m):
        for k in range(p):
            variance_1d(x[i, :, k], &var[k], 1)
        mean_1d(var, &act_idx[i])
        act_idx[i] = sqrt(act_idx[i])
      
    return activity