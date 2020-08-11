# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as npy_double, intc, nanmin, nanmax
from libc.math cimport log, exp, ceil, sqrt, isnan
from .common cimport mean_sd_1d


cpdef hist(const double[:] signal, int ncells, double min_val, double max_val, Py_ssize_t N):
    cdef Py_ssize_t i
    counts = zeros(ncells, dtype=intc)

    cdef int[:] c_view = counts
    cdef int idx
    cdef double bin_width = (max_val - min_val) / <double>(ncells)

    if bin_width == 0.0:
        bin_width = 1.0  # prevent 0 division
    
    for i in range(N):
        if isnan(signal[i]):
            continue
        
        idx = <int>((signal[i] - min_val) / bin_width)
        if idx == ncells:
            idx -= 1
        
        c_view[idx] += 1
    
    return counts


cpdef histogram(const double[:] signal, double[:] descriptor):
    cdef Py_ssize_t N = signal.size
    cdef double min_val = nanmin(signal)
    cdef double max_val = nanmax(signal)
    cdef double delta = (max_val - min_val) / <double>(N - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ceil(sqrt(N))

    return hist(signal, <int>(descriptor[2]), min_val, max_val, N)


def SignalEntropy(const double[:, :, :] signal):
    cdef Py_ssize_t M = signal.shape[0], N = signal.shape[1], P = signal.shape[2]

    res = zeros((M, P), dtype=npy_double)

    cdef double[:, ::1] result = res
    cdef double[::1] d = zeros(3, dtype=npy_double)
    cdef double[::1] data_norm = zeros(N, dtype=npy_double)

    cdef double logf, nbias, count, estimate, h_n, std, mean
    cdef Py_ssize_t i, j, k, n

    for i in range(M):
        for k in range(P):
            std = 0.
            mean = 0.
            mean_sd_1d(signal[i, :, k], &mean, &std)
            
            if std == 0:
                std = 1.  # ensure no division by 0
            for j in range(N):
                data_norm[j] = signal[i, j, k] / std
            
            h = histogram(data_norm, d)

            if (d[0] == d[1]):  # data is constant
                result[i] = 0.0  # no information
                continue
        
            count = 0
            estimate = 0

            for n in range(<int>(d[2])):
                h_n = h[n]
                if h_n > 0:
                    logf = log(h_n)
                else:
                    logf = 0.0
            
                count += h_n
                estimate -= h_n * logf

            nbias = -(d[2] - 1) / (2 * count)
            estimate = estimate / count + log(count) + log((d[1] - d[0]) / d[2]) - nbias
            result[i, k] = exp(estimate**2) - 1 - 1

    return res


def SampleEntropy(const double[:, :, :] signal, int M, double r):
    cdef Py_ssize_t n = signal.shape[1], k = signal.shape[2], p = signal.shape[0]
    entropy = zeros((p, M, k), dtype=npy_double)
    cdef double[:, :, ::1] ent = entropy
    cdef long nj, j
    cdef long[:] run = zeros(n, dtype=int), lastrun = zeros(n, dtype=int)
    cdef long N, M1
    cdef double[:, :, :] A = zeros((p, M, k), dtype=npy_double), B = zeros((p, M, k), dtype=npy_double)
    cdef double Y1

    cdef Py_ssize_t i, jj, mm, kk, axis, wind
    for wind in range(p):
        for axis in range(k):
            run[:] = 0
            lastrun[:] = 0
            for i in range(n-1):
                nj = n - i - 1
                Y1 = signal[wind, i, axis]
                for jj in range(nj):
                    j = jj + i + 1
                    if (((signal[wind, j, axis] - Y1) < r) and ((Y1 - signal[wind, j, axis]) < r)):
                        run[jj] = lastrun[jj] + 1
                        M1 = M if (M < run[jj]) else run[jj]
                        for mm in range(M1):
                            A[wind, mm, axis] += 1
                            if (j < (n - 1)):
                                B[wind, mm, axis] += 1
                    else:
                        run[jj] = 0
                for kk in range(nj):
                    lastrun[kk] = run[kk]

    N = <long>(n * (n - 1) / 2)

    for wind in range(p):
        for axis in range(k):
            ent[wind, 0, axis] = -log(A[wind, 0, axis] / N)

            for mm in range(1, M):
<<<<<<< HEAD
                if (B[wind, mm-1, axis] == 0) or (A[wind, mm, axis] == 0):
                    ent[wind, mm, axis] = 100.0  # set to a large value. TODO evaluate this value
                else:
                    ent[wind, mm, axis] = -log(A[wind, mm, axis] / B[wind, mm - 1, axis])
=======
                ent[wind, mm, axis] = -log(A[wind, mm, axis] / B[wind, mm - 1, axis])
>>>>>>> c58b4fb23624c66aa098d7bd6127a9e5dd012ffe

    return entropy




            

