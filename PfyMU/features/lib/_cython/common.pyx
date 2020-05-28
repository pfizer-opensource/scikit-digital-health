# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from libc.math cimport sqrt

# MEAN ONLY FUNCTIONS
cdef void mean_1d(const double[:] x, double* mean):
    cdef Py_ssize_t n = x.size, i

    for i in range(n):
        mean[0] += x[i]
    mean[0] /= n


# VARIANCE ONLY FUNCTIONS
cdef void variance_1d(const double[:] x, double* var, int ddof):
    cdef Py_ssize_t n = x.size, i

    cdef double k = x[0]
    cdef double Ex = 0., Ex2 = 0.

    for i in range(n):
        Ex += x[i] - k
        Ex2 += (x[i] - k)**2
    
    var[0] = (Ex2 - (Ex**2 / n)) / (n - ddof)


# MEAN & STD DEV FUNCTIONS
cdef void mean_sd_1d(const double[:] x, double* mean, double* std):
    cdef Py_ssize_t n = x.size, i

    cdef double k = x[0]
    cdef double Ex = 0., Ex2 = 0.
    mean[0] = 0.

    for i in range(n):
        mean[0] += x[i]
        Ex += x[i] - k
        Ex2 += (x[i] - k)**2
    
    std[0] = sqrt((Ex2 - (Ex**2 / n)) / (n - 1))
    mean[0] /= n
