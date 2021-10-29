! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.

module sort
    use, intrinsic :: iso_c_binding
    implicit none

    interface quick_sort
        ! subroutine quick_argsort_(n, x, idx)
        !     integer(c_long), intent(in) :: n
        !     real(c_double), intent(inout) :: x(n)
        !     integer(c_long), intent(inout) :: idx(n)
        ! end subroutine quick_argsort_
        ! subroutine quick_sort_(n, x)
        !     integer(c_long), intent(in) :: n
        !     real(c_double), intent(inout) :: x(n)
        ! end subroutine quick_sort_
        module procedure quick_argsort_, quick_sort_
    end interface quick_sort
contains
    ! --------------------------------------------------------------------
    ! SUBROUTINE  insertion_sort_2d
    !     Perform an insertion sort on a 2D array. Specifically for embedding
    !     for permutation entropy
    !
    !     adapted from https://www.mjr19.org.uk/IT/sorts/
    ! 
    !     In
    !     m, order  : integer(8)
    !
    !     Inout
    !     x(m, order)    : real(8), array of values to sort (inplace) in the order dimension
    !     idx(m, order)  : integer(8), index of values to be rearranged (inplace) to match the sorting of x
    ! --------------------------------------------------------------------
    subroutine insertion_sort_2d(m, order, x, idx) bind(C, name="insertion_sort_2d")
        integer(c_long), intent(in) :: m, order
        real(c_double), intent(inout) :: x(m, order)
        integer(c_long), intent(inout) :: idx(m, order)
        !
        integer(c_long) :: i, j, jj, itemp(m)
        real(c_double) :: temp(m)

        if ( order .LT. 40) then
            do j=2, order
                do i=1, m
                    temp(i) = x(i, j)
                    itemp(i) = idx(i, j)
                    
                    do jj=j-1, 1, -1
                        if (x(i, jj) .le. temp(i)) exit
                        x(i, jj+1) = x(i, jj)
                        idx(i, jj+1) = idx(i, jj)
                    end do
                    x(i, jj+1) = temp(i)
                    idx(i, jj+1) = itemp(i)
                end do
            end do
            return 
        end if
    end subroutine

    ! --------------------------------------------------------------------
    ! SUBROUTINE  quick_argsort_
    !     Perform a double pivot quicksort, inplace, on a 1d array and its indices
    !
    !     adapted from https://www.mjr19.org.uk/IT/sorts/
    ! 
    !     In
    !     n  : integer(8)
    !
    !     Inout
    !     x(n)    : real(8), array of values to sort (inplace)
    !     idx(n)  : integer(8), index of values to be rearranged (inplace) to match the sorting of x
    ! --------------------------------------------------------------------
    recursive subroutine quick_argsort_(n, x, idx) bind(C, name="quick_argsort_")
        integer(c_long), intent(in) :: n
        real(c_double), intent(inout) :: x(n)
        integer(c_long), intent(inout) :: idx(n)
        ! local
        integer(c_long) :: i, j, k, l, g, itemp, ip1, ip2
        real(c_double) :: temp, p1, p2
        
        ! use insertion sort on small arrays
        if (n .lt. 40) then
            do i=2, n
                temp = x(i)
                itemp = idx(i)
                do j=i-1, 1, -1
                    if (x(j) .le. temp) exit
                    x(j+1) = x(j)
                    idx(j+1) = idx(j)
                end do
                x(j+1) = temp
                idx(j+1) = itemp
            end do
            return
        end if
        
        ! use quicksort on larger arrays
        ip1 = idx(n / 3)
        ip2 = idx(2 * n / 3)
        p1 = x(n / 3)
        p2 = x(2 * n / 3)
        if (p2 .lt. p1) then
            temp = p1
            p1 = p2
            p2 = temp
            ! index
            itemp = ip1
            ip1 = ip2
            ip2 = itemp
        end if
        
        ! put pivots at front and end
        x(n / 3) = x(1)
        x(1) = p1
        x(2 * n / 3) = x(n)
        x(n) = p2
        
        ! move indices
        idx(n / 3) = idx(1)
        idx(1) = ip1
        idx(2 * n / 3) = idx(n)
        idx(n) = ip2
        
        g = n
        l = 2
        do while (x(l) .lt. p1)
            l = l + 1
        end do
        k = l
        
        do while (k .lt. g)
            temp = x(k)
            itemp = idx(k)
            if (temp .lt. p1) then
                x(k) = x(l)
                x(l) = temp
                ! index
                idx(k) = idx(l)
                idx(l) = itemp
                ! increment
                l = l + 1
            else if (temp .gt. p2) then
                do while (x(g - 1) .gt. p2)
                    g = g - 1
                end do
                if (k .ge. g) exit
                
                g = g - 1
                if (x(g) .lt. p1) then
                    x(k) = x(l)
                    x(l) = x(g)
                    x(g) = temp
                    ! index
                    idx(k) = idx(l)
                    idx(l) = idx(g)
                    idx(g) = itemp
                    ! increment
                    l = l + 1
                else
                    x(k) = x(g)
                    x(g) = temp
                    ! index
                    idx(k) = idx(g)
                    idx(g) = itemp
                end if
            end if
            k = k + 1
        end do
        
        if (l .gt. 2) then
            x(1) = x(l - 1)
            x(l - 1) = p1
            ! index
            idx(1) = idx(l - 1)
            idx(l - 1) = ip1
            call quick_argsort_(l-2, x(1:l-2), idx(1:l-2))
        end if
        call quick_argsort_(g-l, x(l:g-1), idx(l:g-1))
        if (g .lt. n) then
            x(n) = x(g)
            x(g) = p2
            idx(n) = idx(g)
            idx(g) = ip2
            call quick_argsort_(n-g, x(g+1:n), idx(g+1:n))
        end if   
    end subroutine


    ! --------------------------------------------------------------------
    ! SUBROUTINE  quick_sort_
    !     Perform a double pivot quicksort, inplace, on a 1d array without indices
    !
    !     adapted from https://www.mjr19.org.uk/IT/sorts/
    ! 
    !     In
    !     n  : integer(8)
    !
    !     Inout
    !     x(n)  : real(8), array of values to sort (inplace)
    ! --------------------------------------------------------------------
    recursive subroutine quick_sort_(n, x) bind(C, name="quick_sort_")
        integer(c_long), intent(in) :: n
        real(c_double), intent(inout) :: x(n)
    !f2py intent(hide) :: n
        integer(c_long) :: i, j, k, l, g
        real(c_double) :: temp, p1, p2
        
        ! use insertion sort on small arrays
        if (n .lt. 40) then
            do i=2, n
                temp = x(i)
                do j=i-1, 1, -1
                    if (x(j) .le. temp) exit
                    x(j+1) = x(j)
                end do
                x(j+1) = temp
            end do
            return
        end if
        
        ! use quicksort on larger arrays
        p1 = x(n / 3)
        p2 = x(2 * n / 3)
        if (p2 .lt. p1) then
            temp = p1
            p1 = p2
            p2 = temp
        end if
        
        ! put pivots at front and end
        x(n / 3) = x(1)
        x(1) = p1
        x(2 * n / 3) = x(n)
        x(n) = p2
        
        g = n
        l = 2
        do while (x(l) .lt. p1)
            l = l + 1
        end do
        k = l
        
        do while (k .lt. g)
            temp = x(k)
            if (temp .lt. p1) then
                x(k) = x(l)
                x(l) = temp
                ! increment
                l = l + 1
            else if (temp .gt. p2) then
                do while (x(g - 1) .gt. p2)
                    g = g - 1
                end do
                if (k .ge. g) exit
                
                g = g - 1
                if (x(g) .lt. p1) then
                    x(k) = x(l)
                    x(l) = x(g)
                    x(g) = temp
                    ! increment
                    l = l + 1
                else
                    x(k) = x(g)
                    x(g) = temp
                end if
            end if
            k = k + 1
        end do
        
        if (l .gt. 2) then
            x(1) = x(l - 1)
            x(l - 1) = p1
            call quick_sort_(l-2, x(1:l-2))
        end if
        call quick_sort_(g-l, x(l:g-1))
        if (g .lt. n) then
            x(n) = x(g)
            x(g) = p2
            call quick_sort_(n-g, x(g+1:n))
        end if   
    end subroutine
end module sort