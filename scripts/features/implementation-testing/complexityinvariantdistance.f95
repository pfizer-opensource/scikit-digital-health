! -*- f95 -*-
include "common.f90"

! --------------------------------------------------------------------
! SUBROUTINE  complexityinvariantdistance
!     Compute the complexity invariant distance metric for a signal
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute over
!     normalize    : logical, normalize the complexity invariant distance
! 
!     Output
!     cid(n, p) : real(8)
! --------------------------------------------------------------------
subroutine complexityinvariantdistance(m, n, p, x, normalize, cid)
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: x(m, n, p)
    logical, intent(in) :: normalize
    real(8), intent(out) :: cid(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    real(8) :: mean, sd
    
    cid = 0._8
    
    do k=1, p
        do j=1, n
            do i=1, m-1
                cid(j, k) = cid(j, k) + (x(i+1, j, k) - x(i, j, k))**2
            end do
            if (normalize) then
                call mean_sd_1d(m, x(:, j, k), mean, sd)
                cid(j, k) = cid(j, k) / sd**2
            end if
        end do
    end do
    cid = sqrt(cid)
end subroutine
                    