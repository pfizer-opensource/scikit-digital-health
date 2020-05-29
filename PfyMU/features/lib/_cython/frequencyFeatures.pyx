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
    cdef double large = 1.e64, small = 1.e-64

    for i in range(n):
        prod *= x[i]
        if (prod > large) or (prod < small):
            logsum += log(prod)
            prod = 1.
    
    return exp((logsum + log(prod)) / n)


cdef linspace(double start, double stop, int N):
    cdef double[:] arr = zeros(N)
    cdef double step = (stop - start) / N
    cdef Py_ssize_t i

    for i in range(N):
        arr[i] = i * step + start
    
    return arr


cdef class FrequencyFeatures:
    # private attributes
    cdef bint base_run

    cdef Py_ssize_t M, N, P, i, j, k

    cdef double invlog2
    cdef double mean

    cdef double[:, :] maxf, maxfv, df_ratio, spec_flat, spec_ent

    cdef int nfft, ihcut, ilcut
    cdef long[:, :] imax
    cdef double[:] freq
    cdef double lic2, logps

    cdef complex[:, :, :] sp_hat
    cdef double[:, :, :] sp_norm
    
    def __init__(self):
        self.base_run = False
        self.invlog2 = 1 / log(2.0)

    cpdef _base_fn(self, const double[:, :, :] x, double fs, double low_cut, double hi_cut):
        # Book-keeping
        self.M = x.shape[0]
        self.N = x.shape[1]
        self.P = x.shape[2]

        self.maxf = zeros((self.M, self.P))
        self.maxfv = zeros((self.M, self.P))
        self.df_ratio = zeros((self.M, self.P))
        self.spec_flat = zeros((self.M, self.P))
        self.spec_ent = zeros((self.M, self.P))

        # function
        self.nfft = 2 ** (<int>(log(self.N) * self.invlog2))
        self.freq = linspace(0, 0.5 * fs, self.nfft)

        self.ihcut = <int>(floor(hi_cut / (fs / 2) * (self.nfft - 1)) + 1)  # high cutoff index
        self.ilcut = <int>(ceil(low_cut / (fs / 2) * (self.nfft - 1)))  # low cutoff index

        if self.ihcut > self.nfft:
            self.ihcut = <int>(self.nfft)
        self.lic2 = log(self.ihcut - self.ilcut) * self.invlog2

        self.sp_hat = fft.fft(x, 2 * self.nfft, axis=1)
        self.sp_norm = real(self.sp_hat[:, self.ilcut:self.ihcut, :] 
                            * conjugate(self.sp_hat[:, self.ilcut:self.ihcut, :]))
        
        self.sp_norm = self.sp_norm / nsum(self.sp_norm, axis=1, keepdims=True) + 1e-10

        self.imax = argmax(self.sp_norm, axis=1)

        # TODO uncomment, figure out how to implement
        # self.base_run = True

    cpdef get_dominant_freq(self, const double[:, :, :] x, double fs, double low_cut, double hi_cut):
        if not self.base_run:
            self._base_fn(x, fs, low_cut, hi_cut)
        
        for self.i in range(self.M):
            for self.k in range(self.P):
                self.maxf[self.i, self.k] = self.freq[self.imax[self.i, self.k]]
        
        return self.maxf
    
    cpdef get_dominant_freq_value(self, const double[:, :, :] x, double fs, double low_cut, double hi_cut):
        if not self.base_run:
            self._base_fn(x, fs, low_cut, hi_cut)
        
        for self.i in range(self.M):
            for self.k in range(self.P):
                self.maxfv[self.i, self.k] = self.sp_norm[self.i, self.imax[self.i, self.k], self.k]
        
        return self.maxfv
    
    cpdef get_spectral_flatness(self, const double[:, :, :] x, double fs, double low_cut, double hi_cut):
        if not self.base_run:
            self._base_fn(x, fs, low_cut, hi_cut)
        
        for self.i in range(self.M):
            for self.k in range(self.P):
                self.mean = 0.
                mean_1d(self.sp_norm[self.i, self.k], &self.mean)
                self.spec_flat[self.i, self.k] = 10. * log(gmean(self.sp_norm[self.i, :, self.k]) / self.mean) / log(10.0)
        
        return self.spec_flat

    cpdef get_power(self, const double[:, :, :] x, double fs, double low_cut, double hi_cut):
        if not self.base_run:
            self._base_fn(x, fs, low_cut, hi_cut)

        # TODO remove the base run setting
        self.base_run = True
        _ = self.get_max_freq(x, fs, low_cut, hi_cut)
        self.base_run = False  # turn off for now
        
        for self.i in range(self.M):
            for self.j in range(self.ihcut - self.ilcut):
                for self.k in range(self.P):
                    if ((self.maxf[self.i, self.k] - 0.5) < self.freq[self.j] < (self.maxf[self.i, self.k] + 0.5)):
                        self.df_ratio[self.i, self.k] += self.sp_norm[self.i, self.j, self.k]
    
        return self.df_ratio
    
    cpdef get_spectral_entropy(self, const double[:, :, :] x, double fs, double low_cut, double hi_cut):
        if not self.base_run:
            self._base_fn(x, fs, low_cut, hi_cut)
        
        for self.i in range(self.M):
            for self.j in range(self.ihcut - self.ilcut):
                for self.k in range(self.P):
                    self.logps = log(self.sp_norm[self.i, self.j, self.k]) * self.invlog2
                    self.spec_ent[self.i, self.k] -= self.logps * self.sp_norm[self.i, self.j, self.k]
            for self.k in range(self.P):
                self.spec_ent[self.i, self.k] /= self.lic2
        
        return self.spec_ent

