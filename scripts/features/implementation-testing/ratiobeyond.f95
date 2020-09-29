! -*- f95 -*-
include "common.f90"


! --------------------------------------------------------------------
! SUBROUTINE  ratioBeyondRSigma
!     Compute the percentage of samples farther away than r * SD from the mean
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute for
!     r            : real(8), factor to multiply SD by
! 
!     Output
!     rbrs(n, p) : real(8)
! --------------------------------------------------------------------
subroutine ratioBeyondRSigma(m, n, p, x, r, rbrs)
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: x(m, n, p), r
    real(8), intent(out) :: rbrs(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    real(8) :: mean, sd
    
    rbrs = 0._8
    
    do k=1, p
        do j=1, n
            call mean_sd_1d(m, x(:, j, k), mean, sd)
            sd = sd * r
            
            rbrs(j, k) = count(abs(x(:, j, k) - mean) > sd)
        end do
    end do
    rbrs = rbrs / m
end subroutine
            