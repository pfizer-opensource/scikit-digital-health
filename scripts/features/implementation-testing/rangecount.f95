! -*- f95 -*-

! --------------------------------------------------------------------
! SUBROUTINE  rangeCount
!     Compute the percentage of samples that lie within the specified range
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute for
!     xmin         : real(8), range min value
!     xmax         : real(8), range max value
! 
!     Output
!     rcp(n, p) : real(8)
! --------------------------------------------------------------------
subroutine rangeCount(m, n, p, x, xmin, xmax, rcp)
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: x(m, n, p), xmin, xmax
    real(8), intent(out) :: rcp(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    
    do k=1, p
        do j=1, n
            rcp(j, k) = count((x(:, j, k) > xmin) .AND. (x(:, j, k) < xmax))
        end do
    end do
    rcp = rcp / m
end subroutine