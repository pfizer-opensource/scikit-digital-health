# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as npy_double


def Autocorrelation(const double[:, :, :] x, int lag, bint normalize):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, j, k
    cdef double mean1, mean2, std1, std2

    autocorr = zeros((m, p), dtype=npy_double)
    cdef double[:, :] ac = autocorr

    if normalize:
        for i in range(m):
            for k in range(p):
                mean_sd_1d(x[i, :n-lag, k], &mean1, &std1)
                mean_sd_1d(x[i, lag:, k], &mean2, &std2)

                for j in range(n-lag):
                    ac[i, k] += (x[i, j, k] - mean1) * (x[i, j+lag, k] - mean2)
                ac[i, k] /= (n - lag) * std1 * std2
    else:
        for i in range(m):
            for k in range(p):
                mean_sd_1d(x[i, :n-lag, k], &mean1, &std1)
                mean_sd_1d(x[i, lag:, k], &mean2, &std2)

                for j in range(n-lag):
                    ac[i, k] += x[i, j, k] * x[i, j+lag, k]
                ac[i, k] /= std1 * std2
    
    return autocorr