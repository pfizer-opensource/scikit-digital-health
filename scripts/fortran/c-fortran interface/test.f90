! -*- f90 -*-

interface
    subroutine test_c(a) bind(c)
        use iso_c_binding
        integer(c_int), intent(out) :: a
    end subroutine test_c
end interface

subroutine test_f()
    use iso_c_binding
    integer(c_int) :: b

    b = 125
    print *, b

    call test_c(b)

    print *, b
end subroutine test_f
