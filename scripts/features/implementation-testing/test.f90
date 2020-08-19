! -*- f90 -*-

module testmod
    type :: fctdata
        real(8), pointer :: tw(:)
    end type fctdata
    
    type :: rplan
        real(8), pointer :: mem(:)
        type(fctdata) :: fct(2)
    end type rplan
    
    ! module level init
    type(rplan) :: plan
    
contains
    subroutine test()
        implicit none
        integer(8) :: i
        
        ! allocate mem variable
        allocate(plan%mem(8))
        plan%mem = 0._8  ! make sure its 0

        call comp_twiddle()
        
        do i=1, 8
            print "(f4.2, f6.2)", plan%fct(1).tw(i), plan%fct(2).tw(i)
        end do
    end subroutine
    
    subroutine comp_twiddle()
        implicit none
        
        integer(8) :: length, i, j
        real(8), pointer :: twid(:), ptr(:)
        length = 8_8
        
        allocate(twid(2 * length))
        do i=1, 2 * length
            twid[i] = 0.1 * real(i)
        end do
        
        ptr => plan%mem
        
        do k=1, 2
            plan%fct(i).tw => ptr