! -*- f95 -*-


! --------------------------------------------------------------------
! SUBROUTINE  linearRegression
!     Compute the linear regression and the slope
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute for
! 
!     Output
!     slope(n, p) : real(8)
! --------------------------------------------------------------------
subroutine linearRegression(m, n, p, x, y, slope)
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: x(m), y(m, n, p)
    real(8), intent(out) :: slope(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    real(8) :: ssxm, ssxym, kx, ky, Ex, Ey, Exy
    
    ! only need to call 1 time
    call covariance(m, x, x, 0_8, ssxm)
    ! start computation for covariance with unchanging x values
    kx = x(1)
    Ex = 0._8
    do i=2, m
        Ex = Ex + (x(i) - kx)
    end do
    
    do k=1, p
        do j=1, n
            ! compute covariance of x and y(:, j, k)
            ky = y(1, j, k)
            Ey = 0._8
            Exy = 0._8
            do i=2, m
                Ey = Ey + (y(i, j, k) - ky)
                Exy = Exy + (x(i) - kx) * (y(i, j, k) - ky)
            end do
            ssxym = (Exy - (Ex * Ey) / m) / m
            
            ! compute best fit slope
            slope(j, k) = ssxym / ssxm
        end do
    end do
end subroutine
    