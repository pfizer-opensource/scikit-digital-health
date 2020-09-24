cdef void mean_1d(const double[:] x, double* mean)

cdef void variance_1d(const double[:] x, double* var, int ddof)

cdef void mean_sd_1d(const double[:] x, double* mean, double* std)