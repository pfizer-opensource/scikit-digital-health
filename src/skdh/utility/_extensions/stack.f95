module stackmod
    use, intrinsic :: iso_c_binding
    implicit none

    type stack
        integer(c_int) :: maxsize  ! max capacity of the stack
        integer(c_int) :: top  ! keep track of where in the stack we are
        real(c_double), pointer, dimension(:) :: items
    end type stack

contains
    function new_stack(n_items)
        integer(c_long), intent(in) :: n_items
        type(stack) :: new_stack

        new_stack%maxsize = n_items
        new_stack%top = -1;
        allocate(new_stack%items(0:n_items - 1))
    end function

    subroutine free_stack(stk)
        type(stack) :: stk

        deallocate(stk%items)
    end subroutine

    function size_stack(stk)
        type(stack), intent(in) :: stk
        integer(c_int) :: size_stack

        size_stack = stk%top + 1;
    end function

    function is_empty_stack(stk)
        type(stack), intent(in) :: stk
        logical :: is_empty_stack

        is_empty_stack = stk%top == -1
    end function

    function is_full_stack(stk)
        type(stack), intent(in) :: stk
        logical :: is_full_stack

        is_full_stack = stk%top == stk%maxsize - 1
    end function is_full_stack

    subroutine push(stk, val)
        type(stack) :: stk
        real(c_double), intent(in) :: val

        ! NO ERROR CHECKING FOR FULL STACK
        stk%top = stk%top + 1
        stk%items(stk%top) = val
    end subroutine

    function pop(stk)
        type(stack) :: stk
        real(c_double) :: pop

        pop = stk%items(stk%top)
        stk%top = stk%top - 1
    end function

    function peek(stk, res)
        type(stack), intent(in) :: stk
        real(c_double), pointer, intent(out) :: res
        logical :: peek

        peek = .not. is_empty_stack(stk)
        if (.not. peek) then  ! if empty (not not emtpy)
            return
        end if
        res => stk%items(stk%top)
    end function peek

end module stackmod