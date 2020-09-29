! -*- f90 -*-

subroutine test()
    use tmod
    implicit none
    integer(8) :: x, y
    
    x = 5_8
    print *, x
    
    call fn1(x, y)
    print *, x, y
    
end subroutine test