# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as npy_double
from libc.math cimport sqrt
from PfyMU.features.lib._cython.common cimport mean_sd_1d


cpdef cy_CID(const double[:, :, :] x, bint normalize):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, j, k

    dist = zeros((m, p), dtype=npy_double)
    cdef double[:, ::1] cid = dist
    cdef double mu = 0., sigma = 0.

    for i in range(m):
        for k in range(p):
            if normalize:
                mu = 0.
                sigma = 0.
                mean_sd_1d(x[i, :, k], &mu, &sigma)
                if sigma != 0.:
                    for j in range(1, n):
                      cid[i, k] += ((x[i, j, k] - x[i, j-1, k]) / sigma)**2
            else:  # if not normalizing
                for j in range(1, n):
                    cid[i, k] += (x[i, j, k] - x[i, j-1, k])**2
    
    for i in range(m):
        for k in range(p):
            cid[i, k] = sqrt(cid[i, k])
    
    return dist