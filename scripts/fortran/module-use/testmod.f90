! -*- f90 -*-

module tmod
    integer(8), private :: nfct = 25_8
    
    type :: ttype
        integer(8) :: nfct = 25_8
    end type ttype
    
    type(ttype), private :: tst
contains
    subroutine fn1(x, y)
        implicit none
        integer(8), intent(in) :: x
        integer(8), intent(out) :: y
        
        y = x + tst%nfct
    end subroutine
end module tmod