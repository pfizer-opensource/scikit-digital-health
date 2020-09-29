! -*- f95 -*-

! --------------------------------------------------------------------
! SUBROUTINE  jerkMetric
!     Compute the jerk metric for a signal
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute for
!     fs           : real(8), sampling frequency, in Hz
! 
!     Output
!     jerk(n, p) : real(8)
! --------------------------------------------------------------------
subroutine jerkMetric(m, n, p, x, fs, jerk)
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: x(m, n, p), fs
    real(8), intent(out) :: jerk(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    real(8) :: amp, jsum, xold
    
    do k=1, p
        do j=1, n
            jsum = 0._8
            xold = x(1, j, k)
            amp = abs(xold)
            do i=2, m
                jsum = jsum + (x(i, j, k) - xold)**2
                if (abs(x(i, j, k)) > amp) then
                    amp = abs(x(i, j, k))
                end if
                xold = x(i, j, k)
            end do
            
            jerk(j, k) = jsum / (720._8 * amp**2) * fs
        end do
    end do
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  dimensionlessJerkMetric
!     Compute the jerk metric, but dimensionless, for a signal
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute for
!     stype        : integer(8), type of signal input. 1=velocity, 2=acceleration, 3=jerk
! 
!     Output
!     djerk(n, p) : real(8)
! --------------------------------------------------------------------
subroutine dimensionlessJerkMetric(m, n, p, x, stype, djerk)
    implicit none
    integer(8), intent(in) :: m, n, p, stype
    real(8), intent(in) :: x(m, n, p)
    real(8), intent(out) :: djerk(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k
    real(8) :: amp, jsum
    
    if (stype == 1) then
        do k=1, p
            do j=1, n
                jsum = 0._8
                amp = abs(x(1, j, k))
                do i=2, m-1
                    jsum = jsum + (x(i-1, j, k) - 2 * x(i, j, k) + x(i+1, j, k))**2
                    if (abs(x(i, j, k)) > amp) then
                        amp = abs(x(i, j, k))
                    end if
                end do
                if (abs(x(m, j, k)) > amp) then
                    amp = abs(x(m, j, k))
                end if
                
                djerk(j, k) = -(m**3 * jsum) / amp**2
            end do
        end do
    else if (stype == 2) then
        do k=1, p
            do j=1, n
                jsum = 0._8
                amp = abs(x(1, j, k))
                do i=2, m
                    jsum = jsum + (-x(i-1, j, k) + x(i, j, k))**2
                    if (abs(x(i, j, k)) > amp) then
                        amp = abs(x(i, j, k))
                    end if
                end do
                
                djerk(j, k) = -(m * jsum) / amp**2
            end do
        end do
    else if (stype == 3) then
        do k=1, p
            do j=1, n
                jsum = 0._8
                amp = 0._8
                do i=1, m
                    jsum = jsum + x(i, j, k)**2
                    if (abs(x(i, j, k)) > amp) then
                        amp = abs(x(i, j, k))
                    end if
                end do
                
                djerk(j, k) = -jsum / (m * amp**2)
            end do
        end do
    end if
end subroutine