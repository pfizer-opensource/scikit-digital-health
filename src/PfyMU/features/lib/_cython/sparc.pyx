# cython: infer_types = True
# cython: wraparound = False
# cython: boundscheck = False
cimport cython
from numpy import zeros, double as npy_double, arange, abs as npy_abs
from numpy.fft import rfft
from libc.math cimport log2, sqrt, ceil, abs, floor


cpdef sparc_1d(const double[:] x, double fsample, int padlevel, double fcut,
            double amp_thresh):
    cdef Py_ssize_t n = x.size, j, ixf, ixa0 = 0, ixa
    cdef double sal = 0., frange

    cdef int nfft = 2**(<int>(ceil(log2(n)) + padlevel))
    ixf = <Py_ssize_t>(floor(fcut / fsample * (nfft-1)))
    cdef double[:] freq = arange(0, fsample, fsample / nfft)

    # normalized magnitude spectrum
    cdef double[:] Mf = npy_abs(rfft(x, n=nfft))
    cdef double max_Mf = 0.
    for j in range(ixf + 1):
        if Mf[j] > max_Mf:
            max_Mf = Mf[j]
    amp_thresh *= max_Mf

    # indices to choose only the spectrum within the given cutoff frequency fcut
    # NOTE: this is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cutoff for
    # arc length calculation)
    ixa = ixf
    while Mf[ixa0] < amp_thresh and ixa0 < nfft:
        ixa0 += 1
    while Mf[ixa] < amp_thresh and ixa > 0:
        ixa -= 1

    frange = freq[ixa] - freq[ixa0]
    for j in range(ixa0+1, ixa+1):
        sal -= sqrt(((freq[j] - freq[j-1]) / frange)**2
                    + ((Mf[j] - Mf[j-1]) / max_Mf)**2)

    return sal


cpdef SPARC(const double[:, :, :] x, double fsample, int padlevel, double fcut,
               double amp_thresh):
    cdef Py_ssize_t m = x.shape[0], p = x.shape[2], i, k

    sparclen = zeros((m, p), dtype=npy_double, order='C')
    cdef double[:, ::1] sal = sparclen

    for i in range(m):
        for k in range(p):
            sal[i, k] = sparc_1d(x[i, :, k], fsample, padlevel, fcut, amp_thresh)

    return sparclen
