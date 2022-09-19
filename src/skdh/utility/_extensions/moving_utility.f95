! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.


subroutine fmoving_max(n, x, wlen, skip, res) bind(C, name="fmoving_max")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: res((n - wlen) / skip + 1)
    ! local
    integer(c_long) :: i, j

    if (skip == 1) then  ! full overlap
        res(1) = MAXVAL(x(1:wlen))
        j = 2_c_long
        do i=1 + skip, n - wlen + 1, skip
            res(j) = MAX(res(j - 1), x(i + wlen - 1))
            j = j + 1
        end do
    else if (skip < wlen) then  ! overlapping windows
        res(1) = MAXVAL(x(1:wlen))  ! get the first value
        j = 2_c_long
        do i=1 + skip, n - wlen + 1, skip
            ! i1 = (i - skip) + wlen
            ! i2 = i + wlen - 1
            res(j) = MAX(res(j - 1), MAXVAL(x(i - skip + wlen:i + wlen - 1)))
            j = j + 1
        end do
    else
        ! no overlapping, so no benefits gained here
        j = 1_c_long
        do i=1, n - wlen + 1, skip
            res(j) = MAXVAL(x(i:i + wlen - 1))
            j = j + 1_c_long
        end do
    end if
end subroutine

subroutine fmoving_min(n, x, wlen, skip, res) bind(C, name="fmoving_min")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: res((n - wlen) / skip + 1)
    ! local
    integer(c_long) :: i, j

    if (skip == 1) then  ! full overlap
        res(1) = MINVAL(x(1:wlen))
        j = 2_c_long
        do i=1 + skip, n - wlen + 1, skip
            res(j) = MIN(res(j - 1), x(i + wlen - 1))
            j = j + 1
        end do
    else if (skip < wlen) then  ! overlapping windows
        res(1) = MINVAL(x(1:wlen))  ! get the first value
        j = 2_c_long
        do i=1 + skip, n - wlen + 1, skip
            ! i1 = (i - skip) + wlen
            ! i2 = i + wlen - 1
            res(j) = MIN(res(j - 1), MINVAL(x(i - skip + wlen:i + wlen - 1)))
            j = j + 1
        end do
    else
        ! no overlapping, so no benefits gained here
        j = 1_c_long
        do i=1, n - wlen + 1, skip
            res(j) = MINVAL(x(i:i + wlen - 1))
            j = j + 1_c_long
        end do
    end if
end subroutine
