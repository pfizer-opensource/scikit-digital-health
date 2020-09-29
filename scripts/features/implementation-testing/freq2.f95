! -*- f90 -*-
include "common.f90"

subroutine dominantFrequency(m, n, p, x, nfft, fs, low_cut, hi_cut, dFreq)
    use real_fft, only : execute_real_forward, destroy_plan
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: dFreq(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: ihcut, ilcut, imax, j, k, ier
    real(8) :: sp_norm(nfft+1)
    real(8) :: sp_hat(2 * nfft+2), y(2*nfft)
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    do k=1, p
        do j=1, n
            y = 0._8
            y(:m) = x(:, j, k)
            sp_hat = 0._8
            call execute_real_forward(2*nfft, y, 1.0_8, sp_hat, ier)
            
            sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            ! find the maximum index
            imax = maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1
            
            dFreq(j, k) = fs * (imax - 1._8) / nfft / 2._8
        end do
    end do
    call destroy_plan()  ! deallocate/unassociate FFT plan
end subroutine


subroutine dominantFrequencyValue(m, n, p, x, nfft, fs, low_cut, hi_cut, dFreqVal)
    use real_fft, only : execute_real_forward, destroy_plan
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: dFreqVal(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: ihcut, ilcut, j, k, ier
    real(8) :: sp_norm(nfft+1)
    real(8) :: sp_hat(2 * nfft+2), y(2*nfft)
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    do k=1, p
        do j=1, n
            y = 0._8
            y(:m) = x(:, j, k)
            sp_hat = 0._8
            call execute_real_forward(2*nfft, y, 1.0_8, sp_hat, ier)
            
            sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            ! find the maximum value
            dFreqVal(j, k) = maxval(sp_norm(ilcut:ihcut))
        end do
    end do
    call destroy_plan()  ! deallocate/unassociate FFT plan
end subroutine


subroutine powerSpectralSum(m, n, p, x, nfft, fs, low_cut, hi_cut, pss)
    use real_fft, only : execute_real_forward, destroy_plan
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: pss(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, j, k, ier
    real(8) :: freq(nfft+1), sp_norm(nfft+1)
    real(8) :: sp_hat(2 * nfft+2), y(2*nfft), fmax
    
    ! compute the resulting frequency array
    freq = (/ (fs * (i - 1) / nfft, i=1, nfft+1) /) / 2._8
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    pss = 0._8  ! ensure set to 0
    
    do k=1, p
        do j=1, n
            y = 0._8
            y(:m) = x(:, j, k)
            sp_hat = 0._8
            call execute_real_forward(2*nfft, y, 1.0_8, sp_hat, ier)
            
            sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            ! find the maximum index
            fmax = freq(maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1)
            
            do i=ilcut, ihcut
                if ((freq(i) > (fmax - 0.5)) .AND. (freq(i) < (fmax + 0.5))) then
                    pss(j, k) = pss(j, k) + sp_norm(i)
                end if
            end do
        end do
    end do
    call destroy_plan()  ! deallocate/unassociate FFT plan
end subroutine


subroutine spectralEntropy(m, n, p, x, nfft, fs, low_cut, hi_cut, sEnt)
    use real_fft, only : execute_real_forward, destroy_plan
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: sEnt(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: i, ihcut, ilcut, j, k, ier
    real(8) :: sp_norm(nfft+1)
    real(8) :: sp_hat(2 * nfft+2), y(2*nfft), log2_idx_cut
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    log2_idx_cut = log(real(ihcut - ilcut + 1, 8)) / log2
    
    sEnt = 0._8
    
    do k=1, p
        do j=1, n
            y = 0._8
            y(:m) = x(:, j, k)
            sp_hat = 0._8
            call execute_real_forward(2*nfft, y, 1.0_8, sp_hat, ier)
            
            sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            do i=ilcut, ihcut
                sEnt(j, k) = sEnt(j, k) - log(sp_norm(i)) / log2 * sp_norm(i)
            end do
        end do
    end do
    sEnt = sEnt / log2_idx_cut
    call destroy_plan()  ! deallocate/unassociate FFT plan
end subroutine




subroutine spectralFlatness(m, n, p, x, nfft, fs, low_cut, hi_cut, sFlat)
    use real_fft, only : execute_real_forward, destroy_plan
    implicit none
    integer(8), intent(in) :: m, n, p, nfft
    real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
    real(8), intent(out) :: sFlat(n, p)
!f2py intent(hide) :: m, n, p
    real(8), parameter :: log2 = log(2._8)
    integer(8) :: ihcut, ilcut, j, k, ier
    real(8) :: sp_norm(nfft+1)
    real(8) :: sp_hat(2 * nfft+2), y(2*nfft), mean
    
    ! find the cutoff indices for the high and low cutoffs
    ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
    ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)
    
    if (ihcut > nfft) then
        ihcut = nfft
    end if
    
    do k=1, p
        do j=1, n
            y = 0._8
            y(:m) = x(:, j, k)
            sp_hat = 0._8
            call execute_real_forward(2*nfft, y, 1.0_8, sp_hat, ier)
            
            sp_norm = sp_hat(1:2*nfft+2:2)**2 + sp_hat(2:2*nfft+2:2)**2
            sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))
            
            mean = sum(sp_norm(ilcut:ihcut)) / (ihcut - ilcut + 1)
            call gmean(ihcut - ilcut + 1, sp_norm(ilcut:ihcut), sFlat(j, k))
            sFlat(j, k) = 10._8 * log(sFlat(j, k) / mean) / log(10._8)
        end do
    end do
    call destroy_plan()  ! deallocate/unassociate FFT plan
end subroutine