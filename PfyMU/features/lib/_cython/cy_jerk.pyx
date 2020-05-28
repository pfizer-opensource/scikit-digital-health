# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
cimport cython
from numpy import zeros, double as npy_double
from libc.math cimport abs, fmax

def cy_JerkMetric(const double[:, :, :] x, double fsample):
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, j, k
    cdef double amplitude, scale, jsum

    jerk_metric = zeros((m, p), dtype=npy_double)
    cdef double[:, ::1] jerk = jerk_metric

    for i in range(m):
        for k in range(p):
            jsum = 0.
            amplitude = abs(x[i, 0, k])
            for j in range(1, n):
                jsum += ((x[i, j, k] - x[i, j-1, k]) * fsample)**2
                if abs(x[i, j, k]) > amplitude:
                    amplitude = abs(x[i, j, k])
            
            scale = 360. * amplitude**2
            jerk[i, k] = jsum / (2 * scale * fsample)
    
    return jerk_metric


def cy_DimensionlessJerk(const double[:, :, :] x, unsigned int stype):
    # stype: 1 == 'velocity', 2 == 'acceleration', 3 == 'jerk'
    cdef Py_ssize_t m = x.shape[0], n = x.shape[1], p = x.shape[2], i, j, k

    dimless_jerk = zeros((m, p), dtype=npy_double, order='C')
    cdef double[:, ::1] jerk = dimless_jerk
    cdef double amplitude, jsum

    if stype == 1:
        for i in range(m):
            for k in range(p):
                jsum = 0.
                amplitude = fmax(abs(x[i, 0, k]), abs(x[i, n-1, k]))
                for j in range(1, n-1):
                    jsum += (x[i, j+1, k] - 2 * x[i, j, k] + x[i, j-1, k])**2
                    if abs(x[i, j, k]) > amplitude:
                        amplitude = abs(x[i, j, k])

                jerk[i, k] = -(n**3 * jsum) / amplitude**2
    elif stype == 2:
        for i in range(m):
            for k in range(p):
                jsum = 0.
                amplitude = abs(x[i, 0, k])
                for j in range(1, n):
                    jsum += (x[i, j, k] - x[i, j-1, k])**2
                    if abs(x[i, j, k]) > amplitude:
                        amplitude = abs(x[i, j, k])

                jerk[i, k] = -(n * jsum) / amplitude**2
    elif stype == 3:
        for i in range(m):
            for k in range(p):
                jsum = 0.
                amplitude = 0.
                for j in range(n):
                    jsum += x[i, j, k]**2
                    if abs(x[i, j, k]) > amplitude:
                        amplitude = abs(x[i, j, k])

                jerk[i, k] = -jsum / (n * amplitude**2)
    return dimless_jerk