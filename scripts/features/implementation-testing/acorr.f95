! -*- f95 -*-
include "common.f90"


! --------------------------------------------------------------------
! SUBROUTINE  autocorrelation
!     Compute the autocorrelation of a signal with the specified lag
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute autocorrelation for
!     lag          : integer(8), lag for the autocorrelation, in samples
!     normalize    : logical, normalize the autocorrelation
! 
!     Output
!     ac(n, p) : real(8)
! --------------------------------------------------------------------
subroutine autocorrelation(m, n, p, x, lag, normalize, ac) bind(C, name='fautocorr')
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: m, n, p, lag
    real(c_double), intent(in) :: x(m, n, p)
    integer(c_int), intent(in) :: normalize
    real(c_double), intent(out) :: ac(n, p)
!f2py intent(hide) :: m, n, p
    integer(c_long) :: i, j, k
    real(c_double) :: mean1, mean2, std1, std2
    
    ac = 0._c_double
    
    if (normalize == 1_c_int) then
        do k=1_c_long, p
            do j=1_c_long, n
                call mean_sd_1d(m-lag, x(:m-lag-1, j, k), mean1, std1)
                call mean_sd_1d(m-lag, x(lag+1:, j, k), mean2, std2)
                
                do i=1_c_long, m-lag
                    ac(j, k) = ac(j, k) + (x(i, j, k) - mean1) * (x(i+lag, j, k) - mean2)
                end do
                ac(j, k) = ac(j, k) / ((m - lag) * std1 * std2)
            end do
        end do
    else
        do k=1_c_long, p
            do j=1_c_long, n
                call mean_sd_1d(m-lag, x(:m-lag-1, j, k), mean1, std1)
                call mean_sd_1d(m-lag, x(lag+1:, j, k), mean2, std2)
                
                do i=1_c_long, m-lag
                    ac(j, k) = ac(j, k) + x(i, j, k) * x(i+lag, j, k)
                end do
                ac(j, k) = ac(j, k) / (std1 * std2)
            end do
        end do
    end if
end subroutine