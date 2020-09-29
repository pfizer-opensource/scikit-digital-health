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
    type(rplan), private :: plan
    
contains
    subroutine test()
        implicit none
        integer(8) :: i
        
        ! allocate mem variable
        allocate(plan%mem(2*8))
        plan%mem = 0._8  ! make sure its 0

        call comp_twiddle()

        do i=1, 8
            print "(f4.1, f6.1)", plan%fct(1)%tw(i), plan%fct(2)%tw(i)
        end do
    end subroutine
    
    subroutine comp_twiddle()
        implicit none
        
        integer(8) :: length, i, j, k, iptr
        real(8), target :: twid(2*8)
        length = 8_8
        iptr = 1_8
        
        do i=1, 2 * length
            twid(i) = 0.1 * real(i-1)
        end do
        
        do k=1, 2
            !allocate(plan%fct(k)%tw(length))
            plan%fct(k)%tw => plan%mem(iptr:iptr+length-1)
            
            do i=2, length, 4
                plan%fct(k)%tw(i) = twid(i)
                plan%fct(k)%tw(i+1) = twid(i+1)
                
                !plan%mem(i+length+iptr+1) = twid(i)
                !plan%mem(i+length+iptr+2) = twid(i+1)
            end do
            iptr = iptr + 5
        end do
        
    end subroutine
end module testmod
                