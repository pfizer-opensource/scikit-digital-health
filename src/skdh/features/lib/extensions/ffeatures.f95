! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.

! --------------------------------------------------------------------
! SUBROUTINE  autocorr_1d
!     Compute the autocorrelation of a signal with the specified lag
! 
!     Input
!     n            : integer(long)
!     x(n)         : real(double), array to compute autocorrelation for
!     lag          : integer(long), lag for the autocorrelation, in samples
!     normalize    : integer(int), normalize the autocorrelation
! 
!     Output
!     ac(n, p) : real(8)
! --------------------------------------------------------------------
subroutine autocorr_1d(n, x, lag, normalize, ac) bind(C, name="autocorr_1d")
    use, intrinsic :: iso_c_binding
    use utility, only : mean_sd_1d
    implicit none
    integer(c_long), intent(in) :: n, lag
    real(c_double), intent(in) :: x(n)
    integer(c_int), intent(in) :: normalize
    real(c_double), intent(out) :: ac
    ! local
    integer(c_long) :: i
    real(c_double) :: mean1, mean2, std1, std2
    
    ac = 0._c_double
    
    if (normalize == 1_c_int) then
        call mean_sd_1d(n - lag, x(:n - lag), mean1, std1)
        call mean_sd_1d(n - lag, x(lag + 1:), mean2, std2)
        
        do i=1_c_long, n - lag
            ac = ac + (x(i) - mean1) * (x(i + lag) - mean2)
        end do
        ac = ac / ((n - lag - 1) * std1 * std2)
    else
        call mean_sd_1d(n - lag, x(:n - lag), mean1, std1)
        call mean_sd_1d(n - lag, x(lag + 1:), mean2, std2)
        
        do i=1_c_long, n - lag
            ac = ac + x(i) * x(i + lag)
        end do
        ac = ac / (std1 * std2)
    end if
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  cid_1d
!     Compute the complexity invariant distance metric for a signal
! 
!     Input
!     n            : integer(long)
!     x(n)         : real(double), array to compute over
!     normalize    : integer(int), normalize the complexity invariant distance
! 
!     Output
!     cid : real(double)
! --------------------------------------------------------------------
subroutine cid_1d(n, x, normalize, cid) bind(C, name='cid_1d')
    use, intrinsic :: iso_c_binding
    use utility, only : mean_sd_1d
    implicit none
    integer(c_long), intent(in) :: n
    real(c_double), intent(in) :: x(n)
    integer(c_int), intent(in) :: normalize
    real(c_double), intent(out) :: cid
    ! local
    integer(c_long) :: i
    real(c_double) :: mean, sd

    cid = 0._c_double;

    do i=1, n - 1
        cid = cid + (x(i + 1) - x(i))**2
    end do
    if (normalize == 1) then
        call mean_sd_1d(n, x, mean, sd)
        cid = cid / sd**2
    end if
    cid = sqrt(cid)
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  permutation_entropy_1d
!     Compute the permutation entropy of a 1d array
! 
!     In
!     n           : integer(long)
!     x(n)        : real(double), array of values of length n
!     order       : integer(long), order (number of values) of each embedding vector
!     delay       : integer(long), number of samples to skip when extracting each embedding vector
!     normalize   : integer(int), normalize the permutation entropy
! 
!     Out
!     pe  : real(double), computed permutation entropy
! --------------------------------------------------------------------
subroutine permutation_entropy_1d(n, x, order, delay, normalize, pe) bind(C, name="permutation_entropy_1d")
    use, intrinsic :: iso_c_binding
    use utility, only : embed_sort, unique
    implicit none
    integer(c_long), intent(in) :: n, order, delay
    real(c_double), intent(in) :: x(n)
    integer(c_int), intent(in) :: normalize
    real(c_double), intent(out) :: pe
    ! local
    integer(c_long) :: i, nsi, n_unq
    integer(c_long) :: sorted_idx(n - (order - 1) * delay, order)
    real(c_double) :: hashval(n - (order - 1) * delay)
    real(c_double) :: unq_vals(n - (order - 1) * delay)
    real(c_double) :: unq_counts(n - (order - 1) * delay)
    real(c_double), parameter :: log2 = dlog(2._c_double)

    nsi = n - (order - 1) * delay

    call embed_sort(n, nsi, x, order, delay, sorted_idx)

    hashval = 0._c_double
    do i=1, order
        hashval = hashval + order**(i-1) * sorted_idx(:, i)
    end do

    unq_counts = 0._c_double
    call unique(nsi, hashval, unq_vals, unq_counts, n_unq)

    unq_counts = unq_counts / sum(unq_counts)

    pe = -sum(unq_counts(:n_unq) * dlog(unq_counts(:n_unq)) / log2)

    if (normalize == 1) then
        pe = pe / (dlog(product((/(real(i, 8), i=1, order)/))) / log2)
    end if
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  sample_entropy_1d
!     Compute the sample entropy of a signal
! 
!     Input
!     n      : integer(long)
!     x(n)   : real(double), array to compute sample entropy on
!     L      : integer(long), length of sets to compare
!     r      : real(double), maximum set distance
! 
!     Output
!     samp_ent : real(double), sample entropy
! --------------------------------------------------------------------
subroutine sample_entropy_1d(n, x, L, r, samp_ent) bind(C, name="sample_entropy_1d")
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: stdout=>output_unit
    implicit none
    integer(c_long), intent(in) :: n, L
    real(c_double), intent(in) :: x(n), r
    real(c_double), intent(out) :: samp_ent
    ! local
    real(c_double) :: x1, A, B
    integer(c_long) :: i, ii, i2, run(n)

    A = 0._c_double
    B = 0._c_double
    run = 0_c_long
    do i = 1, n - 1
        x1 = x(i)
        do ii = 1, n - i
            i2 = ii + i
            if (abs(x(i2) - x1) < r) then
                run(ii) = run(ii) + 1

                if (run(ii) >= L) then
                    A = A + 1._c_double
                end if
                if (run(ii) >= (L - 1)) then
                    if (i2 < n) then
                        B = B + 1._c_double
                    end if
                end if
            else
                run(ii) = 0_c_long
            end if
        end do
    end do

    if (L == 1) then
        samp_ent = -log(A / (n * (n - 1) / 2._c_double))
    else
        samp_ent = -log(A / B)
    end if
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  signal_entropy_1d
!     Compute the signal entropy of a 1d signal
! 
!     Input
!     n      : integer(long)
!     x(n)   : real(double), array to compute signal entropy for
! 
!     Output
!     ent : real(double)
! --------------------------------------------------------------------
subroutine signal_entropy_1d(n, x, ent) bind(C, name="signal_entropy_1d")
    use, intrinsic :: iso_c_binding
    use utility, only : histogram, mean_sd_1d
    implicit none
    integer(c_long), intent(in) :: n
    real(c_double), intent(out) :: x(n)
    real(c_double), intent(out) :: ent
    ! local
    real(c_double) :: d(3), nbias, cnt
    real(c_double) :: estimate, stdev, mean, logf
    integer(c_long) :: i, sqrt_n
    integer(c_long) :: h(ceiling(sqrt(real(n))))

    sqrt_n = ceiling(sqrt(real(n, c_double)))

    call mean_sd_1d(n, x, mean, stdev)

    if (stdev == 0._c_double) then
        stdev = 1._c_double  ! ensure no 0 division
    end if

    call histogram(n, sqrt_n, x, d, h)
    d(1:2) = d(1:2) / stdev

    if (d(1) == d(2)) then
        ent = 0._c_double
        return
    end if

    cnt = 0._c_double
    estimate = 0._c_double

    do i = 1, int(d(3), c_long)
        if (h(i) > 0) then
            logf = log(real(h(i), c_double))
        else
            logf = 0._c_double
        end if

        cnt = cnt + h(i)
        estimate = estimate - h(i) * logf
    end do

    nbias = -(d(3) - 1._c_double) / (2._c_double * cnt)
    estimate = estimate / cnt + log(cnt) + log((d(2) - d(1)) / d(3)) - nbias

    ent = exp(estimate**2) - 2._c_double
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  dominant_freq_1d
!     Compute the dominant frequency of a signal, in the specified range
! 
!     Input
!     n        : integer(long)
!     x(n)     : real(double), array to compute signal entropy for
!     nfft     : integer(long), number of points to use in the FFT computation
!     fs       : real(double), sampling frequency in Hz
!     low_cut  : real(double), low frequency cutoff for the range to use
!     hi_cut   : real(double), high frequency cutoff for the range to use
! 
!     Output
!     df : real(double)
! --------------------------------------------------------------------
subroutine dominant_freq_1d(n, x, fs, nfft, low_cut, hi_cut, df) bind(C, name="dominant_freq_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long) :: n, nfft
    real(c_double), intent(in) :: x(n), low_cut, hi_cut, fs
    real(c_double), intent(out) :: df
    ! local
    real(c_double), parameter :: log2 = log(2._c_double)
    integer(c_long) :: ihcut, ilcut, imax, ier
    real(c_double) :: sp_norm(nfft + 1)
    real(c_double) :: sp_hat(2 * nfft + 2), y(2 * nfft)

    ! find the cutoff indices for the high and low cutoffs
    ihcut = min(floor(hi_cut / (fs / 2) * (nfft - 1) + 1, c_long), nfft + 1)
    ilcut = max(ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, c_long), 1_c_long)

    if (ihcut > nfft) then
        ihcut = nfft
    end if

    y = 0._c_double
    y(:n) = x
    sp_hat = 0._c_double
    call execute_real_forward(2 * nfft, y, 1.0_c_double, sp_hat, ier)

    sp_norm = sp_hat(1:2 * nfft + 2:2)**2 + sp_hat(2:2 * nfft + 2:2)**2
    sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) + 1.d-10

    ! find the maximum index
    imax = maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1

    df = fs * (imax - 1._c_double) / nfft / 2._c_double

end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  dominant_freq_value_1d
!     Compute the dominant frequency value (spectral power) of a signal, in the specified range
! 
!     Input
!     n        : integer(long)
!     x(n)     : real(double), array to compute signal entropy for
!     nfft     : integer(long), number of points to use in the FFT computation
!     fs       : real(double), sampling frequency in Hz
!     low_cut  : real(double), low frequency cutoff for the range to use
!     hi_cut   : real(double), high frequency cutoff for the range to use
! 
!     Output
!     dfval : real(double)
! --------------------------------------------------------------------
subroutine dominant_freq_value_1d(n, x, fs, nfft, low_cut, hi_cut, dfval) bind(C, name="dominant_freq_value_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long) :: n, nfft
    real(c_double), intent(in) :: x(n), low_cut, hi_cut, fs
    real(c_double), intent(out) :: dfval
    ! local
    real(c_double), parameter :: log2 = log(2._c_double)
    integer(c_long) :: ihcut, ilcut, ier
    real(c_double) :: sp_norm(nfft + 1)
    real(c_double) :: sp_hat(2 * nfft + 2), y(2 * nfft)

    ! find the cutoff indices for the high and low cutoffs
    ihcut = min(floor(hi_cut / (fs / 2) * (nfft - 1) + 1, c_long), nfft + 1)
    ilcut = max(ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, c_long), 1_c_long)

    if (ihcut > nfft) then
        ihcut = nfft
    end if

    y = 0._c_double
    y(:n) = x
    sp_hat = 0._c_double
    call execute_real_forward(2 * nfft, y, 1.0_c_double, sp_hat, ier)

    sp_norm = sp_hat(1:2 * nfft + 2:2)**2 + sp_hat(2:2 * nfft + 2:2)**2
    sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) + 1.d-10
    ! find the maximum value
    dfval = maxval(sp_norm(ilcut:ihcut))
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  power_spectral_sum_1d
!     Compute the sum of the spectral power in a 1hz range (+- 0.5hz) around the 
!     dominant frequency
! 
!     Input
!     n        : integer(long)
!     x(n)     : real(double), array to compute signal entropy for
!     nfft     : integer(long), number of points to use in the FFT computation
!     fs       : real(double), sampling frequency in Hz
!     low_cut  : real(double), low frequency cutoff for the range to use
!     hi_cut   : real(double), high frequency cutoff for the range to use
! 
!     Output
!     pss : real(double)
! --------------------------------------------------------------------
subroutine power_spectral_sum_1d(n, x, fs, nfft, low_cut, hi_cut, pss) bind(C, name="power_spectral_sum_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long), intent(in) :: n, nfft
    real(c_double), intent(in) :: x(n), fs, low_cut, hi_cut
    real(c_double), intent(out) :: pss
    ! local
    real(c_double), parameter :: log2 = log(2._c_double)
    integer(c_long) :: i, ihcut, ilcut, ier, imax
    real(c_double) :: sp_norm(nfft + 1)
    real(c_double) :: sp_hat(2 * nfft + 2), y(2 * nfft)

    ! find the cutoff indices for the high and low cutoffs
    ihcut = min(floor(hi_cut / (fs / 2) * (nfft - 1) + 1, c_long), nfft + 1)
    ilcut = max(ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, c_long), 1_c_long)

    if (ihcut > nfft) then
        ihcut = nfft
    end if

    pss = 0._c_double  ! ensure starts at 0

    y = 0._c_double
    y(:n) = x
    sp_hat = 0._c_double
    call execute_real_forward(2 * nfft, y, 1.0_c_double, sp_hat, ier)

    sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
    sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) + 1.d-10

    ! find the maximum index.  minus 2 to account for adding min 1 (ilcut) and then shifting to 0
    ! at the first index
    imax = maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1

    ! adjust ilcut and ihcut so they correspond to fmax +- 0.5Hz
    ilcut = max(imax - ceiling(0.5 * real(nfft, c_double) / fs * 2._c_double), 1_c_long)
    ihcut = min(imax + floor(0.5 * real(nfft, c_double) / fs * 2._c_double), nfft)

    do i=ilcut, ihcut
        pss = pss + sp_norm(i)
    end do
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  range_power_sum_1d
!     Compute the sum of the spectral power in the range specified.
! 
!     Input
!     n         : integer(long)
!     x(n)      : real(double), array to compute signal entropy for
!     nfft      : integer(long), number of points to use in the FFT computation
!     fs        : real(double), sampling frequency in Hz
!     low_cut   : real(double), low frequency cutoff for the range to use
!     hi_cut    : real(double), high frequency cutoff for the range to use
!     normalize : integer(int), normalize to the power sum across the whole range
! 
!     Output
!     pss : real(double)
! --------------------------------------------------------------------
subroutine range_power_sum_1d(n, x, fs, nfft, low_cut, hi_cut, normalize, rps) bind(C, name="range_power_sum_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long), intent(in) :: n, nfft
    real(c_double), intent(in) :: x(n), fs, low_cut, hi_cut
    integer(c_int), intent(in) :: normalize
    real(c_double), intent(out) :: rps
    ! local
    real(c_double), parameter :: log2 = log(2._c_double)
    integer(c_long) :: i, ihcut, ilcut, ier
    real(c_double) :: sp_norm(nfft + 1)
    real(c_double) :: sp_hat(2 * nfft + 2), y(2 * nfft)

    ! find the cutoff indices for the high and low cutoffs
    ihcut = min(floor(hi_cut / (fs / 2) * (nfft - 1) + 1, c_long), nfft + 1)
    ilcut = max(ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, c_long), 1_c_long)

    if (ihcut > nfft) then
        ihcut = nfft
    end if

    rps = 0._c_double  ! ensure starts at 0

    y = 0._c_double
    y(:n) = x
    sp_hat = 0._c_double
    call execute_real_forward(2 * nfft, y, 1.0_c_double, sp_hat, ier)

    sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
    ! sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) + 1.d-10

    if (normalize == 1_c_int) then
        rps = sum(sp_norm(ilcut:ihcut)) / sum(sp_norm(1:nfft))
    else
        rps = sum(sp_norm(ilcut:ihcut))
    end if
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  spectral_entropy_1d
!     Compute the spectral entropy of the specified frequency range
! 
!     Input
!     n        : integer(long)
!     x(n)     : real(double), array to compute signal entropy for
!     nfft     : integer(long), number of points to use in the FFT computation
!     fs       : real(double), sampling frequency in Hz
!     low_cut  : real(double), low frequency cutoff for the range to use
!     hi_cut   : real(double), high frequency cutoff for the range to use
! 
!     Output
!     sEnt : real(double)
! --------------------------------------------------------------------
subroutine spectral_entropy_1d(n, x, fs, nfft, low_cut, hi_cut, sEnt) bind(C, name="spectral_entropy_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long), intent(in) :: n, nfft
    real(c_double), intent(in) :: x(n), low_cut, hi_cut, fs
    real(c_double), intent(out) :: sEnt
    ! local
    real(c_double), parameter :: log2 = log(2._c_double)
    integer(c_long) :: i, ihcut, ilcut, ier
    real(c_double) :: sp_norm(nfft + 1), sp_hat(2 * nfft + 2), log2_idx_cut
    real(c_double) :: y(2 * nfft)

    ! find the cutoff indices for the high and low cutoffs
    ihcut = min(floor(hi_cut / (fs / 2) * (nfft - 1) + 1, c_long), nfft + 1)
    ilcut = max(ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, c_long), 1_c_long)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    log2_idx_cut = log(real(ihcut - ilcut + 1, c_double)) / log2
    
    sEnt = 0._c_double
    y = 0._c_double
    y(:n) = x
    sp_hat = 0._c_double
    call execute_real_forward(2 * nfft, y, 1.0_c_double, sp_hat, ier)

    sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
    sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) + 1.d-10

    do i=ilcut, ihcut
        sEnt = sEnt - log(sp_norm(i)) / log2 * sp_norm(i)
    end do
    sEnt = sEnt / log2_idx_cut
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  spectral_flatness_1d
!     Compute the spectral flatness of the specified frequency range
! 
!     Input
!     n        : integer(long)
!     x(n)     : real(double), array to compute signal entropy for
!     nfft     : integer(long), number of points to use in the FFT computation
!     fs       : real(double), sampling frequency in Hz
!     low_cut  : real(double), low frequency cutoff for the range to use
!     hi_cut   : real(double), high frequency cutoff for the range to use
! 
!     Output
!     df : real(double)
! --------------------------------------------------------------------
subroutine spectral_flatness_1d(n, x, fs, nfft, low_cut, hi_cut, sFlat) bind(C, name="spectral_flatness_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    use utility, only : gmean
    implicit none
    integer(c_long), intent(in) :: n, nfft
    real(c_double), intent(in) :: x(n), low_cut, hi_cut, fs
    real(c_double), intent(out) :: sFlat
    ! local
    real(c_double), parameter :: log2 = log(2._c_double)
    integer(c_long) :: ihcut, ilcut, ier
    real(c_double) :: sp_norm(nfft+1)
    real(c_double) :: sp_hat(2 * nfft+2), y(2*nfft), mean

    ! find the cutoff indices for the high and low cutoffs
    ihcut = min(floor(hi_cut / (fs / 2._c_double) * (nfft - 1) + 1, c_long), nfft + 1)
    ilcut = max(ceiling(low_cut / (fs / 2._c_double) * (nfft - 1) + 1, c_long), 1_c_long)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if

    y = 0._c_double
    y(:n) = x
    sp_hat = 0._c_double
    call execute_real_forward(2 * nfft, y, 1._c_double, sp_hat, ier)

    sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
    sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) + 1.d-10

    mean = sum(sp_norm(ilcut:ihcut)) / (ihcut - ilcut + 1)
    call gmean(ihcut - ilcut + 1, sp_norm(ilcut:ihcut), sFlat)
    sFlat = 10._c_double * log(sFlat / mean) / log(10._c_double)
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  jerk_1d
!     Compute the jerk metric for a 1d signal
! 
!     Input
!     n      : integer(long)
!     x(n)   : real(double), array to compute for
!     fs     : real(double), sampling frequency, in Hz
! 
!     Output
!     jerk : real(double)
! --------------------------------------------------------------------
subroutine jerk_1d(n, x, fs, jerk) bind(C, name="jerk_1d")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n
    real(c_double), intent(in) :: x(n), fs
    real(c_double), intent(out) :: jerk
    ! local
    integer(c_long) :: i
    real(c_double) :: amp, jsum, xold

    jsum = 0._c_double
    xold = x(1)
    amp = abs(xold)
    do i=2, n
        jsum = jsum + (x(i) - xold)**2
        if (abs(x(i)) > amp) then
            amp = abs(x(i))
        end if
        xold = x(i)
    end do
    jerk = jsum / (720._c_double * amp**2) * fs
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  dimensionless_jerk_1d
!     Compute the jerk metric, but dimensionless, for a signal
! 
!     Input
!     n      : integer(long), axis dimension
!     x(n)   : real(double), array to compute for
!     stype  : integer(long), type of signal input. 1=velocity, 2=acceleration, 3=jerk
! 
!     Output
!     djerk : real(double)
! --------------------------------------------------------------------
subroutine dimensionless_jerk_1d(n, x, stype, djerk) bind(C, name="dimensionless_jerk_1d")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, stype
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: djerk
    ! local
    integer(c_long) :: i
    real(c_double) :: amp, jsum

    jsum = 0._c_double
    amp = abs(x(1))

    if (stype == 1) then
        do i=2, n-1
            jsum = jsum + (x(i-1) - 2._c_double * x(i) + x(i+1))**2
            if (abs(x(i)) > amp) then
                amp = abs(x(i))
            end if
        end do
        if (abs(x(n)) > amp) then
            amp = abs(x(n))
        end if
        djerk = -(n**3 * jsum) / amp**2
    else if (stype == 2) then
        do i=2, n
            jsum = jsum + (x(i) - x(i-1))**2
            if (abs(x(i)) > amp) then
                amp = abs(x(i))
            end if
        end do
        djerk = -(n * jsum) / amp**2
    else if (stype == 3) then
        do i=1, n
            jsum = jsum + x(i)**2
            if (abs(x(i)) > amp) then
                amp = abs(x(i))
            end if
        end do
        djerk = -jsum / (n * amp**2)
    end if
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  range_count_1d
!     Compute the percentage of samples that lie within the specified range
! 
!     Input
!     n      : integer(long)
!     x(n)   : real(double), array to compute for
!     xmin   : real(double), range min value
!     xmax   : real(double), range max value
! 
!     Output
!     rcp : real(double)
! --------------------------------------------------------------------
subroutine range_count_1d(n, x, xmin, xmax, rcp) bind(C, name="range_count_1d")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n
    real(c_double), intent(in) :: x(n), xmin, xmax
    real(c_double), intent(out) :: rcp
    ! local
    integer(c_long) :: i

    rcp = 0._c_double
    do i=1, n
        if ((x(i) >= xmin) .AND. (x(i) < xmax)) then
            rcp = rcp + 1._c_double
        end if
    end do
    rcp = rcp / n

end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  ratio_beyond_r_sigma_1d
!     Compute the percentage of samples farther away than r * SD from the mean
! 
!     Input
!     n      : integer(long)
!     x(n)   : real(double), array to compute for
!     r      : real(double), factor to multiply SD by
! 
!     Output
!     rbrs : real(double)
! --------------------------------------------------------------------
subroutine ratio_beyond_r_sigma_1d(n, x, r, rbrs) bind(C, name="ratio_beyond_r_sigma_1d")
    use, intrinsic :: iso_c_binding
    use utility, only : mean_sd_1d
    implicit none
    integer(c_long), intent(in) :: n
    real(c_double), intent(in) :: x(n), r
    real(c_double), intent(out) :: rbrs
    ! local
    real(c_double) :: mean, stdev

    call mean_sd_1d(n, x, mean, stdev)

    stdev = stdev * r

    rbrs = real(count(abs(x - mean) > stdev), c_double) / n
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  linear_regression_1d
!     Compute the linear regression and the slope
! 
!     Input
!     n      : integer(long)
!     y(n)   : real(double), array to compute for
!     fs     : real(double), sampling frequency for the array
! 
!     Output
!     slope : real(double)
! --------------------------------------------------------------------
subroutine linear_regression_1d(n, y, fs, slope) bind(C, name="linear_regression_1d")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n
    real(c_double), intent(in) :: y(n), fs
    real(c_double), intent(out) :: slope
    ! local
    integer(c_long) :: i
    real(c_double) :: ssxm, ssxym, ky, Ex, Ey, Exy

    ! The variance of a time series with N elements and sampling frequency fs
    ! can be expressed using the sum of natural numbers up to n-1 divided by the fs
    ! and the sum of squares natural numbers up to n-1 divided by fs**2
    ! kx = 0  ! since the first time value is 0
    ! Ex = (n - 1) * n / (2 * fs)
    ! Exy = ((n - 1) * n * (2 * n - 1)) / (6 * fs**2)
    ! ssxm = (Exy - Ex**2 / n) / n  ! biased estimator

    ! ssxm = (((n - 1) * n * (2 * n - 1) / (6 * fs**2)) - (((n-1)**2 * n**2) / (4 * n * fs**2))) / n
    ! ssxm = ((n - 1) * n * (2 * n - 1) / (6 * n * fs**2)) - (((n-1)**2 * n**2) / (4 * n**2 * fs**2))
    ! ssxm = ((n - 1) * (2 * n - 1) / (6 * fs**2)) - ((n-1)**2 / (4 * fs**2))

    ! ssxm = (2*n**2 - 3*n + 1) / (6 * fs**2) - ((n**2 - 2 * n + 1) / (4 * fs**2))
    ! ssxm = (4 * n**2 - 6 * n + 2) / (12 * fs**2) - ((3 * n**2 - 6 * n + 3) / (12 * fs**2))
    ! ssxm = (4 * n**2 - 6 * n + 2 - 3 * n**2 + 6 * n - 3) / (12 * fs**2)
    ssxm = (n**2 - 1) / (12._c_double * fs**2)
    ! need Ex though for linear regression computation
    Ex = (n - 1) * n / (2._c_double * fs)

    ky = y(1)
    Ey = 0._c_double
    Exy = 0._c_double

    do i=2, n
        Ey = Ey + (y(i) - ky)
        Exy = Exy + ((i-1) / fs) * (y(i) - ky)
    end do
    ssxym = (Exy - (Ex * Ey) / n) / n

    slope = ssxym / ssxm
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  SPARC
!     Compute the spectral arc length measure of smoothness
! 
!     Input
!     n            : integer(long), axis dimension
!     x(n)         : real(double), array to compute for
!     fs           : real(double), sampling frequency in Hz
!     padlevel     : integer(long), amount of zero-padding for the FFT
!     fc           : real(double), frequency cutoff (Hz) to lowpass filter
!     amp_thresh   : real(double), normalized power spectra threshold for the arc
!                                  length calculation
! 
!     Output
!     sal   : real(double)
! --------------------------------------------------------------------
subroutine sparc_1d(n, x, fs, padlevel, fc, amp_thresh, sal) bind(C, name="sparc_1d")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long), intent(in) :: n, padlevel
    real(c_double), intent(in) :: x(n), fs, fc, amp_thresh
    real(c_double), intent(out) :: sal
    ! local
    integer(c_long) :: nfft, i, ixf, inxi, inxf, ier
    real(c_double) :: freq_factor
    real(c_double) :: Mf(2**(ceiling(log(real(n))/log(2.0)) + padlevel-1)+1)
    real(c_double) :: y(2**(ceiling(log(real(n))/log(2.0)) + padlevel))
    real(c_double) :: sp_hat(2**(ceiling(log(real(n))/log(2.0)) + padlevel)+2)

    ier = 0_c_long

    nfft = 2**(ceiling(log(real(n))/log(2.0)) + padlevel)

    ! frequency cutoff index. This will essentially function as a low-pass filter
    ! to remove high frequency noise from affecting the next step
    ixf = min(ceiling(fc / (0.5_c_double * fs) * (nfft / 2)), int(nfft / 2 + 1, c_long))

    sal = 0._c_double

    ! compute fft
    sp_hat = 0._c_double
    y = 0._c_double
    y(:n) = x
    call execute_real_forward(nfft, y, 1._c_double, sp_hat, ier)
    if (ier /= 0_c_long) return

    ! normalize the FFT response
    do i = 1, nfft + 2, 2
        Mf((i + 1) / 2) = sqrt(sp_hat(i)**2 + sp_hat(i + 1)**2)
    end do
    Mf = Mf / maxval(Mf)

    inxi = 1_c_long
    inxf = ixf

    do while ((Mf(inxi) < amp_thresh) .AND. (inxi < (nfft / 2 + 1)))
        inxi = inxi + 1
    end do
    do while ((Mf(inxf) < amp_thresh) .AND. (inxf > 1))
        inxf = inxf - 1
    end do

    ! compute the arc length
    freq_factor = 1._c_double / real(inxf - inxi, c_double)**2

    do i = inxi + 1, inxf
        sal = sal - sqrt(freq_factor + (Mf(i) - Mf(i - 1))**2)
    end do
end subroutine
