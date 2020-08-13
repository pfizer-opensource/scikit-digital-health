! -*- f90 -*-
include "fftpack3.f90"
include "common.f90"


subroutine fft(n, x, nfft, res)
    implicit none
    integer(4), intent(in) :: n, nfft
    real(4), intent(in) :: x(n)
    complex(4), intent(out) :: res(2 * nfft)
!f2py intent(hide) :: n
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav
    real(4) :: wrk(2 * 2 * nfft)
    integer(4) :: lenwrk
    integer(4) :: ier
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    call cfft1i(2 * nfft, wsave, lensav, ier)

    res = complex(0._4, 0._4)
    res(:n) = cmplx(x, kind=4)
    
    call cfft1f(2 * nfft, 1_4, res, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
end subroutine


subroutine dominantFrequency(m, n, p, x, nfft, fs, low_cut, hi_cut, dFreq)
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: dFreq(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, imax, j, k
    real(8) :: freq(nfft), sp_norm(2 * nfft)
    complex(4) :: sp_hat(2 * nfft)
    ! for the FFT
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav, lenwrk, ier
    real(4) :: wrk(2 * 2 * nfft)
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    ! compute the resulting frequency array
    freq = (/ (fs * (i - 1) / nfft, i=1, nfft) /) / 2._8
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    ! compute the FFT
    call cfft1i(2 * nfft, wsave, lensav, ier)  ! initialize
    
    do k=1, p
        do j=1, n
            sp_hat = complex(0._4, 0._4)
            sp_hat(:m) = cmplx(x(:, j, k), kind=4)
            call cfft1f(2 * nfft, 1_4, sp_hat, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
            sp_norm = real(sp_hat * conjg(sp_hat), 8)
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            ! find the maximum index
            imax = maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1
            
            dFreq(j, k) = freq(imax)
        end do
    end do
end subroutine


subroutine dominantFrequencyValue(m, n, p, x, nfft, fs, low_cut, hi_cut, dFreqVal)
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: dFreqVal(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, j, k
    real(8) :: freq(nfft), sp_norm(2 * nfft)
    complex(4) :: sp_hat(2 * nfft)
    ! for the FFT
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav, lenwrk, ier
    real(4) :: wrk(2 * 2 * nfft)
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    ! compute the resulting frequency array
    freq = (/ (fs * (i - 1) / nfft, i=1, nfft) /) / 2._8
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    ! compute the FFT
    call cfft1i(2 * nfft, wsave, lensav, ier)  ! initialize
    
    do k=1, p
        do j=1, n
            sp_hat = complex(0._4, 0._4)
            sp_hat(:m) = cmplx(x(:, j, k), kind=4)
            call cfft1f(2 * nfft, 1_4, sp_hat, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
            sp_norm = real(sp_hat * conjg(sp_hat), 8)
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut)) 
            
            dFreqVal(j, k) = maxval(sp_norm(ilcut:ihcut))
        end do
    end do
end subroutine


subroutine powerSpectralSum(m, n, p, x, nfft, fs, low_cut, hi_cut, pss)
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: pss(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, j, k
    real(8) :: freq(nfft), sp_norm(2 * nfft)
    real(8) :: fmax
    complex(4) :: sp_hat(2 * nfft)
    ! for the FFT
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav, lenwrk, ier
    real(4) :: wrk(2 * 2 * nfft)
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    ! compute the resulting frequency array
    freq = (/ (fs * (i - 1) / nfft, i=1, nfft) /) / 2._8
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    ! compute the FFT
    call cfft1i(2 * nfft, wsave, lensav, ier)  ! initialize
    
    do k=1, p
        do j=1, n
            sp_hat = complex(0._4, 0._4)
            sp_hat(:m) = cmplx(x(:, j, k), kind=4)
            call cfft1f(2 * nfft, 1_4, sp_hat, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
            sp_norm = real(sp_hat * conjg(sp_hat), 8)
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            ! find the maximum index
            fmax = freq(maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1)
            
            pss(j, k) = 0._8
            do i=ilcut, ihcut
                if ((freq(i) > (fmax - 0.5)) .AND. (freq(i) < (fmax + 0.5))) then
                    pss(j, k) = pss(j, k) + sp_norm(i)
                end if
            end do
        end do
    end do
end subroutine


subroutine spectralEntropy(m, n, p, x, nfft, fs, low_cut, hi_cut, sEnt)
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: sEnt(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, j, k
    real(8) :: freq(nfft), sp_norm(2 * nfft)
    real(8) :: log2_idx_cut
    complex(4) :: sp_hat(2 * nfft)
    ! for the FFT
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav, lenwrk, ier
    real(4) :: wrk(2 * 2 * nfft)
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    ! compute the resulting frequency array
    freq = (/ (fs * (i - 1) / nfft, i=1, nfft) /) / 2._8
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    log2_idx_cut = log(real(ihcut - ilcut + 1, 8)) / log2
    
    ! compute the FFT
    call cfft1i(2 * nfft, wsave, lensav, ier)  ! initialize
    
    do k=1, p
        do j=1, n
            sp_hat = complex(0._4, 0._4)
            sp_hat(:m) = cmplx(x(:, j, k), kind=4)
            call cfft1f(2 * nfft, 1_4, sp_hat, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
            sp_norm = real(sp_hat * conjg(sp_hat), 8)
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            sEnt(j, k) = 0._8
            do i=ilcut, ihcut
                sEnt(j, k) = sEnt(j, k) - log(sp_norm(i)) / log2 * sp_norm(i)
            end do
        end do
    end do
    sEnt = sEnt / log2_idx_cut
end subroutine


subroutine spectralFlatness(m, n, p, x, nfft, fs, low_cut, hi_cut, sFlat)
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: sFlat(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, j, k
    real(8) :: freq(nfft), sp_norm(2 * nfft)
    real(8) :: mean
    complex(4) :: sp_hat(2 * nfft)
    ! for the FFT
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav, lenwrk, ier
    real(4) :: wrk(2 * 2 * nfft)
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    ! compute the resulting frequency array
    freq = (/ (fs * (i - 1) / nfft, i=1, nfft) /) / 2._8
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    ! compute the FFT
    call cfft1i(2 * nfft, wsave, lensav, ier)  ! initialize
    
    do k=1, p
        do j=1, n
            sp_hat = complex(0._4, 0._4)
            sp_hat(:m) = cmplx(x(:, j, k), kind=4)
            call cfft1f(2 * nfft, 1_4, sp_hat, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
            sp_norm = real(sp_hat * conjg(sp_hat), 8)
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            mean = sum(sp_norm(ilcut:ihcut)) / (ihcut - ilcut + 1)
            call gmean(ihcut - ilcut + 1, sp_norm(ilcut:ihcut), sFlat(j, k))
            sFlat(j, k) = 10._8 * log(sFlat(j, k) / mean) / log(10._8)
        end do
    end do
end subroutine