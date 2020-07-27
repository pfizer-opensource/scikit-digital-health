! -*- f90 -*-

! --------------------------------------------------------------------
! SUBROUTINE  mean_sd_1d
!     Compute the mean and sample standard deviation of an array
! 
!     In
!     n  : integer(8), number of samples in x
!     x  : real(8), array of values of length n
! 
!     Out
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


! --------------------------------------------------------------------
! SUBROUTINE  unique
!     Compute the unique values, and their counts in an array
! 
!     In
!     n     : integer(8), number of samples in x
!     x(n)  : real(8), array of values of length n
! 
!     Out
!     uniq(n)   : real(8), first j values are the unique values of x
!     counts(n) : integer(8), first j values are the counts of the unique values
!     j         : integer(8), number of unique values in x
! --------------------------------------------------------------------
subroutine unique(n, x, uniq, counts, j)
    implicit none
    integer(8), intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: uniq(n), counts(n)
    integer(8), intent(out) :: j
!f2py intent(hide) :: n
    integer(8) :: i
    
    ! not worried about modifying x in this case
    call dpqsort_no_idx(n, x)
    
    counts = 0._8
    
    uniq(1) = x(1)
    counts(1) = 1._8
    j = 2
    
    do i=2, n
        if (x(i) .NE. x(i-1)) then
            uniq(j) = x(i)
            counts(j) = counts(j) + 1
            j = j + 1
        else
            counts(j - 1) = counts(j - 1) + 1
        end if
    end do
    j = j - 1  ! make length of unique samples valid
end subroutine

    