module mov_extrema
    use, intrinsic :: iso_c_binding
    use stackmod
    implicit none

    type queue
        type(stack) :: dqstack
        type(stack) :: dqstackext
        type(stack) :: eqstack
        type(stack) :: eqstackext
    end type queue

contains
    function new_queue(n_items) result(q)
        integer(c_long), intent(in) :: n_items
        type(queue) :: q

        q%dqstack = new_stack(n_items)
        q%dqstackext = new_stack(n_items)
        q%eqstack = new_stack(n_items)
        q%eqstackext = new_stack(n_items)
    end function new_queue

    subroutine free_queue(q)
        type(queue) :: q

        call free_stack(q%dqstack)
        call free_stack(q%dqstackext)
        call free_stack(q%eqstack)
        call free_stack(q%eqstackext)
    end subroutine free_queue

    function is_empty_queue(q)
        type(queue), intent(in) :: q
        logical :: is_empty_queue

        is_empty_queue = is_empty_stack(q%dqstack) .and. is_empty_stack(q%eqstack)
    end function is_empty_queue

    subroutine enqueue_max(q, val)
        type(queue) :: q
        real(c_double), intent(in) :: val
        ! local
        real(c_double), pointer :: res
        logical :: tmp

        if (is_empty_queue(q)) then
            call push(q%dqstack, val)
            call push(q%dqstackext, val)
        elseif (.not. is_empty_stack(q%eqstack)) then
            call push(q%eqstack, val)
            ! get/update the current max
            tmp = peek(q%eqstackext, res)

            call push(q%eqstackext, max(res, val))
        else
            call push(q%eqstack, val)
            call push(q%eqstackext, val)
        end if
    end subroutine enqueue_max

    subroutine enqueue_min(q, val)
        type(queue) :: q
        real(c_double), intent(in) :: val
        ! local
        real(c_double), pointer :: res
        logical :: tmp

        if (is_empty_queue(q)) then
            call push(q%dqstack, val)
            call push(q%dqstackext, val)
        elseif (.not. is_empty_stack(q%eqstack)) then
            call push(q%eqstack, val)
            ! get/update the current max
            tmp = peek(q%eqstackext, res)

            call push(q%eqstackext, min(res, val))
        else
            call push(q%eqstack, val)
            call push(q%eqstackext, val)
        end if
    end subroutine enqueue_min

    subroutine move_enqueue_to_dequeue_max(q)
        type(queue) :: q
        real(c_double) :: cmax, data, tmp
        real(c_double), pointer :: pr

        ! set to max to low value so it will get overwritten
        cmax = - huge(cmax)
        do while (peek(q%eqstack, pr))
            data = pop(q%eqstack)
            tmp = pop(q%eqstackext)
            ! update current max if data is larger
            cmax = max(cmax, data)
            call push(q%dqstack, data)
            call push(q%dqstackext, cmax)
        end do
    end subroutine

    subroutine move_enqueue_to_dequeue_min(q)
        type(queue) :: q
        real(c_double) :: cmin, data, tmp
        real(c_double), pointer :: pr

        ! set to max to high value so it will get overwritten
        cmin = huge(cmin)
        do while (peek(q%eqstack, pr))
            data = pop(q%eqstack)
            tmp = pop(q%eqstackext)
            ! update current min if data is smaller
            cmin = min(cmin, data)
            call push(q%dqstack, data)
            call push(q%dqstackext, cmin)
        end do
    end subroutine

    function dequeue_max(q) result(res)
        type(queue) :: q
        real(c_double) :: res
        ! local
        real(c_double) :: tmp

        ! NO EMPTY STACK ERRORING
        res = pop(q%dqstack)
        tmp = pop(q%dqstackext)

        if (is_empty_stack(q%dqstack)) then
            call move_enqueue_to_dequeue_max(q)
        end if
    end function

    function dequeue_min(q) result(res)
        type(queue) :: q
        real(c_double) :: res
        ! local
        real(c_double) :: tmp

        ! NO EMPTY STACK ERRORING
        res = pop(q%dqstack)
        tmp = pop(q%dqstackext)

        if (is_empty_stack(q%dqstack)) then
            call move_enqueue_to_dequeue_min(q)
        end if
    end function

    function get_max(q)
        type(queue) :: q
        real(c_double) :: get_max
        ! local
        real(c_double), pointer :: dq_max, eq_max
        logical :: tmp

        tmp = peek(q%dqstackext, dq_max)
        tmp = peek(q%eqstackext, eq_max)

        get_max = max(dq_max, eq_max)
    end function

    function get_min(q)
        type(queue) :: q
        real(c_double) :: get_min
        ! local
        real(c_double), pointer :: dq_min, eq_min
        logical :: tmp

        tmp = peek(q%dqstackext, dq_min)
        tmp = peek(q%eqstackext, eq_min)

        get_min = min(dq_min, eq_min)
    end function

    subroutine moving_max_f(n, x, wlen, skip, res)
        integer(c_long), intent(in) :: n, wlen, skip
        real(c_double), intent(in) :: x(0:n-1)
        real(c_double), intent(out) :: res(0:(n - wlen) / skip)
        ! local
        type(queue) :: q
        real(c_double) :: dq_res
        integer(c_long) :: i, j, k, ii

        q = new_queue(wlen)
        k = 0

        do i=0, wlen - 1
            call enqueue_max(q, x(i))
        end do

        ! get the max for the first window
        res(k) = get_max(q)
        k = k + 1

        ! keep track of the last element (+1) inserted into the stack
        ii = wlen
        do i=skip, n - wlen, skip
            do j=max(ii, i), i + wlen - 1
                dq_res = dequeue_max(q)
                call enqueue_max(q, x(j))
            end do
            ii = i + wlen  ! update to latest taken element

            res(k) = get_max(q)
            k = k + 1
        end do

        call free_queue(q)
    end subroutine
end module mov_extrema
