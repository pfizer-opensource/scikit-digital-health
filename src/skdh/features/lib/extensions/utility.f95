! -*- f95 -*-

module utility
    use, intrinsic :: iso_c_binding
    use sort, only : quick_sort, insertion_sort_2d
    implicit none
contains
    ! --------------------------------------------------------------------
    ! SUBROUTINE  mean_sd_1d
    !     Compute the mean and sample standard deviation of an array
    ! 
    !     In
    !     n  : integer(long), number of samples in x
    !     x  : real(double), array of values of length n
    ! 
    !     Out
    !     mn : real(double), mean of x
    !     sd : real(double), sample standard deviation of x
    ! --------------------------------------------------------------------
    subroutine mean_sd_1d(n, x, mn, sd) bind(C)
        integer(c_long), intent(in) :: n
        real(c_double), intent(in) :: x(n)
        real(c_double), intent(out) :: mn, sd
        ! local
        real(c_double) :: Ex, Ex2
        integer(c_long) :: i
        
        Ex2 = 0._c_double
        mn = 0._c_double
        
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
    !     n     : integer(long), number of samples in x
    !     x(n)  : real(double), array of values of length n
    ! 
    !     Out
    !     uniq(n)   : real(double), first j values are the unique values of x
    !     counts(n) : integer(long), first j values are the counts of the unique values
    !     j         : integer(long), number of unique values in x
    ! --------------------------------------------------------------------
    subroutine unique(n, x, uniq, counts, j) bind(C)
        integer(c_long), intent(in) :: n
        real(c_double), intent(in) :: x(n)
        real(c_double), intent(out) :: uniq(n), counts(n)
        integer(c_long), intent(out) :: j
    !f2py intent(hide) :: n
        integer(c_long) :: i
        real(c_double) :: y(n)
        
        y = x
        call quick_sort(n, y)
        
        counts = 0._c_double
        
        uniq(1) = y(1)
        counts(1) = 1._c_double
        j = 2_c_long
        
        do i=2_c_long, n
            if (y(i) .NE. y(i-1)) then
                uniq(j) = y(i)
                counts(j) = counts(j) + 1
                j = j + 1
            else
                counts(j - 1) = counts(j - 1) + 1
            end if
        end do
        j = j - 1  ! make length of unique samples valid
    end subroutine


    ! --------------------------------------------------------------------
    ! SUBROUTINE  gmean
    !     Compute the geometric mean of a series
    ! 
    !     In
    !     n     : integer(long), number of samples in x
    !     x(n)  : real(double), array of values of length n
    ! 
    !     Out
    !     gm : real(double), geometric mean of the series x
    ! --------------------------------------------------------------------
    subroutine gmean(n, x, gm) bind(C)
        integer(c_long), intent(in) :: n
        real(c_double), intent(in) :: x(n)
        real(c_double), intent(out) :: gm
    !f2py intent(hide) :: n
        real(c_double) :: logsum, prod
        real(c_double), parameter :: large=1.d64, small=1.d-64
        integer(c_long) :: i
        
        logsum = 0._c_double
        prod = 1._c_double
        
        do i=1, n
            prod = prod * x(i)
            if ((prod > large) .OR. (prod < small)) then
                logsum = logsum + log(prod)
                prod = 1._c_double
            end if
        end do
        gm = exp((logsum + log(prod)) / n)
    end subroutine

    ! --------------------------------------------------------------------
    ! SUBROUTINE  embed_sort
    !     Create embedding vectors from the array x, and get the indices of the sorted embedding vectors
    ! 
    !     In
    !     n, m   : integer(long)
    !     x(n)   : real(double), array of values of length n
    !     order  : integer(long), order (number of values) of each embedding vector
    !     delay  : integer(long), number of samples to skip when extracting each embedding vector
    ! 
    !     Out
    !     res(m, order)  : integer(long), sorted embedding vectors
    ! --------------------------------------------------------------------
    subroutine embed_sort(n, m, x, order, delay, res) bind(C, name="embed_sort")
        integer(c_long), intent(in) :: n, m, order, delay
        real(c_double), intent(in) :: x(n)
        integer(c_long), intent(out) :: res(m, order)
        ! local
        integer(c_long) :: j
        real(c_double) :: temp(m, order)

        do j=0, order-1
            res(:, j+1) = j
        end do

        do j=0, order-1
            temp(:, j+1) = x(j * delay + 1:j * delay + 1 + m)
        end do

        call insertion_sort_2d(m, order, temp, res)
    end subroutine
    
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
    subroutine hist(n, x, ncells, min_val, max_val, counts) bind(C)
        integer(c_long), intent(in) :: n, ncells
        real(c_double), intent(in) :: x(n), min_val, max_val
        integer(c_long), intent(out) :: counts(ncells)
    !f2py intent(hide) :: n
        integer(c_long) :: i, idx
        real(c_double) :: bin_width
        
        counts = 0_c_long
        
        bin_width = (max_val - min_val) / ncells
        
        ! prevent counting when no variation
        if (bin_width == 0._c_double) then
            bin_width = 1._c_double  ! prevent 0 division
        end if
        
        do i=1, n
            if (.NOT. isnan(x(i))) then
                idx = int((x(i) - min_val) / bin_width, c_long) + 1
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
    subroutine histogram(n, k, x, descriptor, counts) bind(C)
        integer(c_long), intent(in) :: n, k
        real(c_double), intent(in) :: x(n)
        real(c_double), intent(out) :: descriptor(3)
        integer(c_long), intent(out) :: counts(k)
    !f2py intent(hide) :: n
        real(c_double) :: min_val, max_val, delta
        
        min_val = minval(x)
        max_val = maxval(x)
        delta = (max_val - min_val) / (n - 1)
        
        descriptor(1) = min_val - delta / 2._c_double
        descriptor(2) = max_val + delta / 2._c_double
        descriptor(3) = real(k, c_double)
        
        call hist(n, x, k, min_val, max_val, counts)
    end subroutine
end module utility