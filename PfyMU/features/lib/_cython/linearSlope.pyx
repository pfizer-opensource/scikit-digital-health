# cython: infer_types = True
# cython: wraparound = True
# cython: boundscheck = False
cimport cython
from numpy import zeros, double as npy_double, reshape
from libc.math cimport sqrt
from .common cimport mean_1d


cpdef covariance(const double[:] x, const double[:] y, int ddof):
    cdef Py_ssize_t n = x.size, i

    cdef double kx = x[0], ky = y[0]
    cdef double Ex = 0., Ey = 0., Exy = 0., cov

    for i in range(n):
        Ex += (x[i] - kx)
        Ey += (y[i] - ky)
        Exy += (x[i] - kx) * (y[i] - ky)

    cov = (Exy - (Ex * Ey) / (n - ddof)) / (n - ddof)
    return cov


cpdef linregress(const double[:] x, const double[:] y):
    cdef Py_ssize_t n = x.size
    if n != y.size:
        raise ValueError('Inputs must be the same size.')
    if n < 2:
        raise ValueError('Inputs must have more than 1 element.')
    
    cdef double xmean, ymean, ssxm, ssxym, ssym
    cdef double slope, intercept

    mean_1d(x, &xmean)
    mean_1d(x, &ymean)

    # average sum of squares
    ssxm = covariance(x, x, 0)
    ssym = covariance(y, y, 0)
    ssxym = covariance(x, y, 0)

    slope = ssxym / ssxm
    intercept = ymean - slope * xmean

    return slope, intercept


cpdef LinRegression(const double[:] x, const double[:, :, :] y):
    cdef Py_ssize_t m = y.shape[0], p = y.shape[2], i, k

    slp = zeros((m, p), dtype=npy_double)
    itcpt = zeros((m, p), dtype=npy_double)
    cdef double[:, :] slope = slp, intercept = itcpt
    cdef double[:, :]  x2d

    for i in range(m):
        for k in range(p):
            slope[i, k], intercept[i, k] = linregress(x, y[i, :, k])
    
    return slp, itcpt

