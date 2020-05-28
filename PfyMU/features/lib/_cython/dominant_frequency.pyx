# cython: infer_types = True
# cython: boundscheck = False
# cython: wraparound = False
from numpy import fft, sum as nsum, less_equal, zeros, conjugate, argmax, real
cimport cython
from libc.math cimport log, pow, exp, floor, ceil
from signal_features._extensions.common cimport mean_1d


cdef double gmean(const double[:] x):
    cdef Py_ssize_t n = x.size, i
    cdef double logsum = 0.0, prod = 1.0
    cdef double large = 1.e64, small=1.e-64

    for i in range(n):
        prod *= x[i]
        if (prod > large) or (prod < small):
            logsum += log(prod)
            prod = 1.

    return exp((logsum + log(prod)) / n)


cpdef linspace(double start, double stop, int N):
    cdef double[:] arr = zeros(N)
    cdef double step = (stop - start) / N
    cdef Py_ssize_t i

    for i in range(N):
        arr[i] = i * step + start

    return arr


def frequency_features_2d(const double[:, :] x, double fs, double low_cut, double hi_cut):
    # local variables
    cdef Py_ssize_t N=x.shape[0], P=x.shape[1], j, k
    cdef double invlog2 = 1 / log(2.)
    cdef double mean
    # return variables
    max_freq = zeros(P)
    max_freq_val = zeros(P)
    dom_freq_ratio = zeros(P)
    spectral_flatness = zeros(P)
    spectral_entropy = zeros(P)

    cdef double[:] maxf = max_freq, maxfv = max_freq_val
    cdef double[:] domf_ratio = dom_freq_ratio
    cdef double[:] spec_flat = spectral_flatness, spec_ent = spectral_entropy

    # function
    cdef int nfft = 2 ** (<int>(log(N) * invlog2))
    cdef double[:] freq = linspace(0.0, 0.5 * fs, nfft)

    cdef int ihcut = <int>(floor(hi_cut / (fs / 2) * (nfft - 1)) + 1)  # high cutoff
    cdef int ilcut = <int>(ceil(low_cut / (fs / 2) * (nfft - 1)))  # low cutoff
    if ihcut > nfft:
        ihcut = <int>(nfft)


    cdef complex[:, :] sp_hat = fft.fft(x, 2 * nfft, axis=0)
    cdef double[:, :] sp_norm = real(sp_hat[ilcut:ihcut, :]
                                     * conjugate(sp_hat[ilcut:ihcut, :]))
    sp_norm = sp_norm / nsum(sp_norm, axis=0, keepdims=True) + 1e-10

    imax = argmax(sp_norm, axis=0)
    for k in range(P):
        maxf[k] = freq[imax[k]]
        maxfv[k] = sp_norm[imax[k], k]

        # spectral flatness
        mean = 0.
        mean_1d(sp_norm[:, k], &mean)
        spec_flat[k] = 10. * log(gmean(sp_norm[:, k]) / mean) / log(10.)

    for j in range(ihcut - ilcut):
        for k in range(P):
            # dominant frequency ratio
            if ((maxf[k] - 0.5) < freq[j] < (maxf[k] + 0.5)):
                domf_ratio[k] += sp_norm[j, k]
            # spectral entropy estimate
            logps = log(sp_norm[j, k]) * invlog2
            spec_ent[k] -= logps * sp_norm[j, k]
    for k in range(P):
        spec_ent[k] /= log(ihcut - ilcut) * invlog2

    return max_freq, max_freq_val, dom_freq_ratio, spectral_flatness, spectral_entropy


def frequency_features_3d(const double[:, :, :] x, double fs, double low_cut, double hi_cut):
    # local variables
    cdef Py_ssize_t M = x.shape[0], N = x.shape[1], P = x.shape[2], i, j, k
    cdef double invlog2 = 1 / log(2.0)
    cdef double mean
    # return variables
    max_freq = zeros((M, P))
    max_freq_val = zeros((M, P))
    dom_freq_ratio = zeros((M, P))
    spectral_flatness = zeros((M, P))
    spectral_entropy = zeros((M, P))

    cdef double[:, :] maxf = max_freq, maxfv = max_freq_val
    cdef double[:, :] domf_ratio = dom_freq_ratio
    cdef double[:, :] spec_flat = spectral_flatness, spec_ent = spectral_entropy

    # function
    cdef int nfft = 2 ** (<int>(log(N) * invlog2))
    cdef double[:] freq = linspace(0.0, 0.5 * fs, nfft)

    cdef int ihcut = <int>(floor(hi_cut / (fs / 2) * (nfft - 1)) + 1)  # high cutoff
    cdef int ilcut = <int>(ceil(low_cut / (fs / 2) * (nfft - 1)))  # low cutoff
    if ihcut > nfft:
        ihcut = <int>(nfft)
    cdef double lic2 = log(ihcut - ilcut) * invlog2

    cdef complex[:, :, :] sp_hat = fft.fft(x, 2 * nfft, axis=1)
    cdef double[:, :, :] sp_norm = real(sp_hat[:, ilcut:ihcut, :]
                                        * conjugate(sp_hat[:, ilcut:ihcut, :]))
    sp_norm = sp_norm / nsum(sp_norm, axis=1, keepdims=True) + 1e-10

    imax = argmax(sp_norm, axis=1)
    for i in range(M):
        for k in range(P):
            maxf[i, k] = freq[imax[i, k]]
            maxfv[i, k] = sp_norm[i, imax[i, k], k]

            # spectral flatness
            mean = 0.
            mean_1d(sp_norm[i, :, k], &mean)
            spec_flat[i, k] = 10. * log(gmean(sp_norm[i, :, k]) / mean) / log(10.)

        for j in range(ihcut - ilcut):
            for k in range(P):
                # dominant frequency ratio
                if ((maxf[i, k] - 0.5) < freq[j] < (maxf[i, k] + 0.5)):
                    domf_ratio[i, k] += sp_norm[i, j, k]
                # spectral entropy
                logps = log(sp_norm[i, j, k]) * invlog2
                spec_ent[i, k] -= logps * sp_norm[i, j, k]
        for k in range(P):
            spec_ent[i, k] /= lic2

    return max_freq, max_freq_val, dom_freq_ratio, spectral_flatness, spectral_entropy
