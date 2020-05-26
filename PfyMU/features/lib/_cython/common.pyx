# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython

# MEAN ONLY FUNCTIONS
cdef void mean_1d(const double[:] x, double* mean):
    cdef Py_ssize_t n = x.size, i

    for i in range(n):
        mean[0] += x[i]
    mean[0] /= n