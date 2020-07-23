! -*- f90 -*-

! --------------------------------------------------------------------
! SUBROUTINE  mean_sd_1d
!     Compute the mean and sample standard deviation of an array
! 
!     Input
!     n  : integer(8), number of samples in x
!     x  : real(8), array of values of length n
! 
!     Output
!     mn : real(8), mean of x
!     sd : real(8), sample standard deviation of x
! --------------------------------------------------------------------
subroutine mean_sd_1d(n, x, mn, sd)
    implicit none
    integer(8), intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: mn, sd
!f2py intent(hide) :: n
    real(8) :: Ex, Ex2
    integer(8) :: i
    
    Ex2 = 0._8
    mn = 0._8
    
    do i=1, n
        mn = mn + x(i)
        Ex2 = Ex2 + (x(i) - x(1))**2
    end do
    
    ! can compute outside loop since mean is the same computation
    Ex = mn - (n * x(1))
    
    sd = sqrt((Ex2 - (Ex**2 / n)) / (n - 1))
    mn = mn / n
end subroutine

    