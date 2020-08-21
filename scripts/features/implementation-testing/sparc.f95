! -*- f95 -*-

! --------------------------------------------------------------------
! SUBROUTINE  SPARC
!     Compute the spectral arc length measure of smoothness
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute for
!     fs           : real(8), sampling frequency in Hz
!     padlevel     : integer(8), amount of zero-padding for the FFT
!     fc           : real(8), frequency cutoff (Hz) to lowpass filter
!     amp_thresh   : real(8), normalized power spectra threshold for the arc
!                             length calculation
! 
!     Output
!     sal(n, p) : real(8)
! --------------------------------------------------------------------
subroutine SPARC(m, n, p, x, fs, padlevel, fc, amp_thresh, sal)
    use real_fft, only : execute_real_forward, destroy_plan
    implicit none
    integer(8), intent(in) :: m, n, p, padlevel
    real(8), intent(in) :: x(m, n, p), fs, fc, amp_thresh
    real(8), intent(out) :: sal(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: j, k, nfft, i, ixf, inxi, inxf, ier
    real(8) :: freq_factor
    real(8) :: Mf(2**(ceiling(log(real(m))/log(2.0)) + padlevel-1)+1)
    real(8) :: y(2**(ceiling(log(real(m))/log(2.0)) + padlevel))
    real(8) :: sp_hat(2**(ceiling(log(real(m))/log(2.0)) + padlevel)+2)
    ier = 0_8
    
    nfft = 2**(ceiling(log(real(m))/log(2.0)) + padlevel)
    
    ! frequency cutoff index.  This will essentially function as a low-pass filter
    ! to remove high frequency noise from affecting the next step
    ixf = ceiling(fc / (0.5 * fs) * (nfft / 2))
    
    ! loop over axes
    sal = 0._8
    do k=1, p
        do j=1, n
            ! compute the fft
            sp_hat = 0._8
            y = 0._8
            y(:m) = x(:, j, k)
            call execute_real_forward(nfft, y, 1.0_8, sp_hat, ier)
            if (ier .NE. 0) return
            
            ! normalize the fft response
            do i=1, nfft+2, 2
                Mf((i+1)/2) = sqrt(sp_hat(i)**2 + sp_hat(i+1)**2)
            end do
            Mf = Mf / maxval(Mf)
            
            inxi = 1_8
            inxf = ixf
            do while ((Mf(inxi) < amp_thresh) .AND. (inxi < (nfft/2+1)))
                inxi = inxi + 1
            end do
            do while ((Mf(inxf) < amp_thresh) .AND. (inxf > 1))
                inxf = inxf - 1
            end do
            
            ! compute the arc length
            freq_factor = 1._8 / (inxf - inxi)**2
            do i=inxi+1, inxf
                sal(j, k) = sal(j, k) - sqrt(freq_factor + (Mf(i) - Mf(i-1))**2)
            end do
        end do
    end do
    call destroy_plan()
end subroutine
    