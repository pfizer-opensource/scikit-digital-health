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
!     x(m, n, p)   : real(8), array to compute signal entropy for
!     lag          : integer(8), lag for the autocorrelation, in samples
!     normalize    : logical, normalize the autocorrelation
! 
!     Output
!     ac(n, p) : real(8)
! --------------------------------------------------------------------
subroutine autocorrelation(m, n, p, x, lag, normalize, ac)
    implicit none
    integer(8), intent(in) :: m, n, p, lag
    real(8), intent(in) :: x(m, n, p)
    logical, intent(in) :: normalize
    real(8), intent(out) :: ac(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    real(8) :: mean1, mean2, std1, std2
    
    ac = 0._8
    
    if (normalize) then
        do k=1, p
            do j=1, n
                call mean_sd_1d(m-lag, x(:m-lag-1, j, k), mean1, std1)
                call mean_sd_1d(m-lag, x(lag+1:, j, k), mean2, std2)
                
                do i=1, m-lag
                    ac(j, k) = ac(j, k) + (x(i, j, k) - mean1) * (x(i+lag, j, k) - mean2)
                end do
                ac(j, k) = ac(j, k) / ((m - lag) * std1 * std2)
            end do
        end do
    else
        do k=1, p
            do j=1, n
                call mean_sd_1d(m-lag, x(:m-lag-1, j, k), mean1, std1)
                call mean_sd_1d(m-lag, x(lag+1:, j, k), mean2, std2)
                
                do i=1, m-lag
                    ac(j, k) = ac(j, k) + x(i, j, k) * x(i+lag, j, k)
                end do
                ac(j, k) = ac(j, k) / (std1 * std2)
            end do
        end do
    end if
end subroutine