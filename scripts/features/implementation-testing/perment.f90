! -*- f95 -*-

subroutine embed(n, x, order, delay, res)
    implicit none
    integer(8), intent(in) :: n, order, delay
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: res(n - ( order - 1) * delay, order)
!f2py intent(hide) :: n
    integer(8) :: i, m
    
    m = n - ( order - 1) * delay
    
    do i=0, order-1
        res(:, i+1) = x(i * delay + 1:i * delay + 1 + m)
    end do
end subroutine


!subroutine permutationEntropy(m, n, p, x, order, delay, normalize, pe)
!    implicit none
!    integer(8), intent(in) :: m, n, p, order, delay
!    real(8), intent(in) :: x(m, n, p)
!    logical, intent(in) :: normalize
!    real(8), intent(out) :: pe(n, p)
!f2py intent(hide) :: m, n, p


recursive subroutine quicksort(a, first, last)
    implicit none
    real*8  a(*), x, t
    integer first, last
    integer i, j

    x = a( (first+last) / 2 )
    i = first
    j = last
    do
     do while (a(i) < x)
        i=i+1
     end do
     do while (x < a(j))
        j=j-1
     end do
     if (i >= j) exit
     t = a(i);  a(i) = a(j);  a(j) = t
     i=i+1
     j=j-1
    end do
    if (first < i-1) call quicksort(a, first, i-1)
    if (j+1 < last)  call quicksort(a, j+1, last)
end subroutine quicksort

recursive subroutine qsort(n, x, idx)
    implicit none
    integer(8), intent(in) :: n
    real(8), intent(inout) :: x(n)
    integer(8), intent(inout) :: idx(n)
!f2py intent(hide) :: n
    integer(8) :: i, j, itmp, mrk
    real(8) :: pivot, rnd, rtmp
    
    if (n > 1) then
        call random_number(rnd)
        ! random pivot, not best but avoids worst case
        pivot = x(int(rnd * real(n - 1)) + 1)
        i = 0
        j = n + 1
        
        do while (i < j)
            j = j - 1
            do while (x(j) > pivot)
                j = j - 1
            end do
            i = i + 1
            do while (x(j) < pivot)
                i = i + 1
            end do
            if (i < j) then
                rtmp = x(i); itmp = idx(i)
                x(i) = x(j); x(j) = rtmp
                idx(i) = idx(j); idx(j) = itmp
            end if
        end do
        
        if (i == j) then
            mrk = i + 1
        else
            mrk = i
        end if
        
        call qsort(x(:(mrk - 1)))
                

            IF (I < J) THEN
              RTEMP = A(I)
              A(I) = A(J)
              A(J) = RTEMP
              ITEMP = IDX(I)
              IDX(I) = IDX(J)
              IDX(J) = ITEMP
            END IF
          END DO

          IF (I == J) THEN
            MARK = I + 1
          ELSE
            MARK = I
          END IF

          CALL QSORT(A(:(MARK - 1)), IDX(:(MARK - 1)), MARK - 1)
          CALL QSORT(A(MARK:), IDX(MARK:), N - MARK + 1)

        END IF
      END
    