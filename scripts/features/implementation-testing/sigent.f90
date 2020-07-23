! -*- f95 -*-
include "common.f90"

! --------------------------------------------------------------------
! SUBROUTINE  hist
!     Count the values of x in bins of equal width
! 
!     Input
!     n       : integer(8), number of samples in x
!     x       : real(8), array of values of length n
!     ncells  : integer(8), number of cells
!     min_val : real(8), minimum value in x
!     max_val : real(8), maximum value in x
! 
!     Output
!     counts  : integer(8), counts in each bin
! --------------------------------------------------------------------
subroutine hist(n, x, ncells, min_val, max_val, counts)
    implicit none
    integer(8), intent(in) :: n, ncells
    real(8), intent(in) :: x(n), min_val, max_val
    integer(8), intent(out) :: counts(ncells)
!f2py intent(hide) :: n
    integer(8) :: i, idx
    real(8) :: bin_width
    
    counts = 0_8
    
    bin_width = (max_val - min_val) / ncells
    
    ! prevent counting when no variation
    if (bin_width == 0.0_8) then
        bin_width = 1.0_8  ! prevent 0 division
    end if
    
    do i=1, n
        if (.NOT. isnan(x(i))) then
            idx = int((x(i) - min_val) / bin_width, 8) + 1_8
            if (idx > ncells) then
                idx = ncells
            end if
            
            counts(idx) = counts(idx) + 1
        end if
    end do
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  histogram
!     Setup values for computing a histogram on x
! 
!     Input
!     n          : integer(8), number of samples in x
!     k          : integer(8), number of cells, =ceiling(sqrt(n))
!     x          : real(8), array of values of length n
! 
!     Output
!     descriptor : integer(8), length 3 array of description of the histogram
!     counts     : integer(8), counts in each bin
! --------------------------------------------------------------------
subroutine histogram(n, k, x, descriptor, counts)
    implicit none
    integer(8), intent(in) :: n, k
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: descriptor(3)
    integer(8), intent(out) :: counts(k)
!f2py intent(hide) :: n
    real(8) :: min_val, max_val, delta
    
    min_val = minval(x)
    max_val = maxval(x)
    delta = (max_val - min_val) / (n - 1)
    
    descriptor(1) = min_val - delta / 2
    descriptor(2) = max_val + delta / 2
    descriptor(3) = real(k, 8)
    
    call hist(n, x, k, min_val, max_val, counts)
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  signalEntropy
!     Compute the signal entropy of a 3d signal x, along the 1st axis
! 
!     Input
!     m            : integer(8), signal dimension
!     n            : integer(8), axis dimension
!     p            : integer(8), window dimension
!     x(m, n, p)   : real(8), array to compute signal entropy for
! 
!     Output
!     sigEnt(n, p) : real(8)
! --------------------------------------------------------------------
subroutine signalEntropy(m, n, p, x, sigEnt)
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: x(m, n, p)
    real(8), intent(out) :: sigEnt(n, p)
!f2py intent(hide) :: m, n, p
    real(8) :: d(3), nbias, cnt
    real(8) :: estimate, stdev, mean, logf
    integer(8) :: i, j, k, sqm
    integer(8) :: h(ceiling(sqrt(real(m))))
    
    sqm = ceiling(sqrt(real(m)))
    
    do k=1, p
        do j=1, n
            call mean_sd_1d(m, x(:, j, k), mean, stdev)
            
            if (stdev == 0.0_8) then
                stdev = 1._8  ! ensure no 0-division
            end if
            
            call histogram(m, sqm, x(:, j, k), d, h)
            d(1:2) = d(1:2) / stdev
            
            if (d(1) == d(2)) then
                sigEnt(j, k) = 0._8
            else
                cnt = 0._8
                estimate = 0._8
                
                do i=1, int(d(3), 8)
                    if (h(i) > 0) then
                        logf = log(real(h(i), 8))
                    else
                        logf = 0._8
                    end if
                    
                    cnt = cnt + h(i)
                    estimate = estimate - h(i) * logf
                end do
                
                nbias = -(d(3) - 1._8) / (2._8 * cnt)
                estimate = estimate / cnt + log(cnt) + log((d(2) - d(1)) / d(3)) - nbias
                
                sigEnt(j, k) = exp(estimate**2) - 2._8
            end if
        end do
    end do
end subroutine