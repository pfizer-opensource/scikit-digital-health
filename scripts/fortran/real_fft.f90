! -*- f90 -*-

module real_fft
    type :: rfftp_fctdata
        integer(8) :: fct
        real(8), pointer :: tw(:)=>null()
    end type rfftp_fctdata

    type :: rfftp_plan
        integer(8) :: length = -1
        integer(8) :: nfct
        integer(8) :: twsize
        real(8), pointer :: mem(:)=>null()
        type(rfftp_fctdata) :: fct(25)
    end type rfftp_plan
    
    ! parameters
    integer(8), parameter, private :: NFCT_ = 25
    
    ! variables
    type(rfftp_plan), private :: plan

contains

    subroutine destroy_plan()
        implicit none
        integer :: i
        
        if (associated(plan%mem)) nullify(plan%mem)
        do i=1, NFCT_
            if (associated(plan%fct(i)%tw)) nullify(plan%fct(i)%tw)
        end do
    end subroutine
    
    subroutine execute_real_forward(n, x, fct, ret, ier)
        implicit none
        integer(8), intent(in) :: n
        real(8), intent(in) :: x(n), fct
        real(8), intent(out) :: ret(n+2)
        integer(8), intent(out) :: ier
!f2py   intent(hide) :: n
        
        ! ensure proper power of 2 size
        if (iand(n, n-1) .NE. 0) then
            ier = -1_8
            return
        end if
        
        if ((plan%length .NE. n) .OR. (plan%length == -1_8)) then
            call make_rfftp_plan(n, ier)
        end if
        if (ier .NE. 0) return
        
        ret = 0_8
        
        ret = 0._8
        ret(2:n+1) = x
        call rfftp_forward(n, ret(2:), fct, ier)
        if (ier .NE. 0) return
        
        ret(1) = ret(2)
        ret(2) = 0._8
        
        ier = 0_8
    
    end subroutine
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    subroutine rfftp_forward(m, x, fct, ier)
        implicit none
        integer(8), intent(in) :: m
        real(8), intent(in), target :: x(m)
        real(8), intent(in) :: fct
        integer(8), intent(out) :: ier
        ! local
        integer(8) :: n, l1, nf, k1, k, ip, ido, iswap
        real(8), target :: ch(m)
        real(8), pointer :: p1(:)=>null(), p2(:)=>null()
        
        if (plan%length == 1_8) then
            ier = -1_8
            return
        end if
        
        n = plan%length
        l1 = n
        nf = plan%nfct
        
        iswap = 1_8  ! keep track of swapping
        p1 => x
        p2 => ch
        
        do k1=1, nf
            k = nf - k1 + 1
            ip = plan%fct(k)%fct
            ido = n / l1
            l1 = l1 / ip
            
            if (ip .EQ. 4) then
                call radf4(ido, l1, p1, p2, plan%fct(k)%tw)
            else if (ip .EQ. 2) then
                call radf2(ido, l1, p1, p2, plan%fct(k)%tw)
            else
                ier = -1
                return
            end if
            
            ! swap which arrays the pointers point to
            nullify(p1)
            nullify(p2)
            if (iand(iswap, 1) .NE. 0) then
                p1 => ch
                p2 => x
            else
                p1 => x
                p2 => ch
            end if
            iswap = iswap + 1_8
            
        end do
        
        ! normalize
        call copy_and_norm(x, p1, n, fct)
        nullify(p1)
        nullify(p2)
        ier = 0_8
    end subroutine
    
    
    subroutine radf2(ido, l1, cc, ch, wa)
        implicit none
        integer(8), intent(in) :: ido, l1
        real(8), dimension(:) :: cc, ch, wa
        ! local
        integer(8), parameter :: cdim=2_8
        integer(8) :: k, i, ic
        real(8) :: tr2, ti2
        
        do k=0, l1-1
            ch(ido*cdim*k+1) = cc(ido*k+1) + cc(ido*(k+l1)+1)
            ch(ido+ido*(1+cdim*k)) = cc(ido*k+1) - cc(ido*(k+l1)+1)
        end do
        if (iand(ido, 1) == 0) then
            do k=0, l1-1
                ch(ido*(1+cdim*k)+1) = -cc(ido+ido*(k+l1))
                ch(ido+ido*(cdim*k)) = cc(ido+ido*k)
            end do
        end if
        if (ido <= 2) return
        do k=0, l1-1
            do i=2, ido-1, 2
                ic = ido - i
                
                tr2 = wa(i-1) * cc(i+ido*(k+l1)) + wa(i) * cc(i+ido*(k+l1)+1)
                ti2 = wa(i-1) * cc(i+ido*(k+l1)+1) - wa(i) * cc(i+ido*(k+l1))
                
                ch(i+ido*cdim*k) = cc(i+ido*k) + tr2
                ch(ic+ido*(1+cdim*k)) = cc(i+ido*k) - tr2
                ch(i+ido*cdim*k+1) = ti2 + cc(i+ido*k+1)
                ch(ic+ido*(1+cdim*k)+1) = ti2 - cc(i+ido*k+1)
            end do
        end do
    end subroutine
    
    subroutine radf4(ido, l1, cc, ch, wa)
        implicit none
        integer(8), intent(in) :: ido, l1
        real(8), dimension(:) :: cc, ch, wa
        ! local
        integer(8), parameter :: cdim=4_8
        real(8), parameter :: hsqt2=0.70710678118654752440
        integer(8) :: k, i, ic
        real(8) :: ci2, ci3, ci4, cr2, cr3, cr4
        real(8) :: ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4
        
        do k=0, l1-1
            tr1 = cc(ido*(k+l1*3)+1) + cc(ido*(k+l1)+1)
            ch(ido*(2+cdim*k)+1) = cc(ido*(k+l1*3)+1) - cc(ido*(k+l1)+1)
            
            tr2 = cc(ido*k+1) + cc(ido*(k+l1*2)+1)
            ch(ido+ido*(1+cdim*k)) = cc(ido*k+1) - cc(ido*(k+l1*2)+1)
            
            ch(ido*(cdim*k)+1) = tr2 + tr1
            ch(ido+ido*(3+cdim*k)) = tr2 - tr1
        end do
        if (iand(ido, 1) == 0) then
            do k=0, l1-1
                ti1 = -hsqt2 * (cc(ido+ido*(k+l1)) + cc(ido+ido*(k+l1*3)))
                tr1 = hsqt2 * (cc(ido+ido*(k+l1)) - cc(ido+ido*(k+l1*3)))

                ch(ido+ido*(cdim*k)) = cc(ido+ido*k) + tr1
                ch(ido+ido*(2+cdim*k)) = cc(ido+ido*k) - tr1
                ch(ido*(3+cdim*k)+1) = ti1 + cc(ido+ido*(k+l1*2))
                ch(ido*(1+cdim*k)+1) = ti1 - cc(ido+ido*(k+l1*2))
            end do
        end if
        if (ido <= 2) return
        do k=0, l1-1
            do i=2, ido-1, 2
                ic = ido-i
                
                cr2 = wa(i-1) * cc(i+ido*(k+l1)) + wa(i) * cc(i+ido*(k+l1)+1)
                ci2 = wa(i-1) * cc(i+ido*(k+l1)+1) - wa(i) * cc(i+ido*(k+l1))
                
                cr3 = wa((i-2)+ido) * cc(i+ido*(k+l1*2)) + wa(i-1+ido) * cc(i+ido*(k+l1*2)+1)
                ci3 = wa((i-2)+ido) * cc(i+ido*(k+l1*2)+1) - wa(i-1+ido) * cc(i+ido*(k+l1*2))
                
                cr4 = wa(i-1+2*(ido-1)) * cc(i+ido*(k+l1*3)) + wa(i+2*(ido-1)) * cc(i+ido*(k+l1*3)+1)
                ci4 = wa((i-1)+2*(ido-1)) * cc(i+ido*(k+l1*3)+1) - wa(i+2*(ido-1)) * cc(i+ido*(k+l1*3))
                
                
                tr1 = cr4 + cr2
                tr4 = cr4 - cr2
                
                ti1 = ci2 + ci4
                ti4 = ci2 - ci4
                
                tr2 = cc(i+ido*k) + cr3
                tr3 = cc(i+ido*k) - cr3
                
                ti2 = cc(i+ido*k+1) + ci3
                ti3 = cc(i+ido*k+1) - ci3
                
                ch(i+ido*cdim*k) = tr2 + tr1
                ch(ic+ido*(3+cdim*k)) = tr2 - tr1
                
                ch(i+ido*cdim*k+1) = ti1 + ti2
                ch(ic+ido*(3+cdim*k)+1) = ti1 - ti2
                
                ch(i+ido*(2+cdim*k)) = tr3 + ti4
                ch(ic+ido*(1+cdim*k)) = tr3 - ti4
                
                ch(i+ido*(2+cdim*k)+1) = tr4 + ti3
                ch(ic+ido*(1+cdim*k)+1) = tr4 - ti3
            end do
        end do
    end subroutine
    
    subroutine copy_and_norm(c, p1, n, fct)
        implicit none
        real(8), target :: c(n)
        real(8), pointer :: p1(:)
        integer(8) :: n
        real(8) :: fct
        
        if (.NOT. associated(p1, c)) then
            if (fct .NE. 1._8) then
                c = fct * p1
            else
                c = p1
            end if
        else
            if (fct .NE. 1._8) then
                c = c * fct
            end if
        end if
    end subroutine
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    subroutine make_rfftp_plan( length, ier)
        implicit none
        integer(8), intent(in) :: length
        integer(8), intent(out) :: ier
        ! local
        integer(8) :: i, tws
        
        if ( length == 0 ) then
            ier = -1_8
            return
        end if
        
        plan%length = length
        plan%nfct = 0_8
        
        do i=1, NFCT_
            plan%fct(i)%fct = 0_8
        end do
        
        if ( length == 1) return
        
        call rfftp_factorize(ier)
        ! error catching
        
        call rfftp_twsize(tws)
        plan%twsize = tws
        
        allocate(plan%mem(tws))
        plan%mem = 0._8
        
        call rfftp_comp_twiddle( length, ier)
    end subroutine
    
    
    subroutine rfftp_comp_twiddle( length, ier)
        implicit none
        integer(8), intent(in) :: length
        integer(8), intent(out) :: ier
        ! local
        real(8), target :: twid(2 * length)
        integer(8) :: l1, i, j, k, ip, ido, iptr, twsize
        
        twsize = plan%twsize
        
        call sincos_2pibyn_half( length, twid)
        l1 = 1_8
        
        ! keep track of plan%mem assignment
        iptr = 1_8
        
        do k=1, plan%nfct
            ip = plan%fct(k)%fct
            ido = length / (l1 * ip)
            
            if (k < plan%nfct)  then ! last factor doesn't need twiddles
                plan%fct(k)%tw => plan%mem(iptr:)
                iptr = iptr + (ip - 1) * (ido - 1)
                
                do j=1, ip-1
                    do i=1, (ido-1)/2
                        plan%fct(k)%tw((j-1)*(ido-1)+2*i-1) = twid(2*j*l1*i+1)
                        plan%fct(k)%tw((j-1)*(ido-1)+2*i) = twid(2*j*l1*i+2)
                    end do
                end do
            end if
            ! remove if ip > 5 due to factors always being 2 or 4
            l1 = l1 * ip
        end do
        ier = 0_8
    end subroutine
    
    subroutine sincos_2pibyn_half(n, res)
        implicit none
        integer(8), intent(in) :: n
        real(8), intent(out) :: res(2 * n)
        
        if ( iand(n, 3_8) == 0) then
            call calc_first_octant(n, res)
            call fill_first_quadrant(n, res)
            call fill_first_half(n, res)
        else if ( iand(n, 1_8) == 0) then
            call calc_first_quadrant(n, res)
            call fill_first_half(n, res)
        else
            call calc_first_half(n, res)
        end if
    end subroutine
    
    subroutine calc_first_quadrant(n, res)
        implicit none
        integer(8), intent(in) :: n
        real(8), intent(inout) :: res(0:2*n-1)
        !local
        real(8) :: p(0:2*n-1)
        integer(8) :: ndone, i, idx1, idx2
        
        p = res + real(n, 8)
        
        call calc_first_octant(ishft(n, 1), p)
        
        ndone = ishft(n+2, -2)
        idx1 = 0_8
        idx2 = 2_8 * ndone - 2_8
        
        do i=0_8, ndone - 2, 2
            res(idx1) = p(2 * i)
            res(idx1+1) = p(2 * i + 1)
            res(idx2) = p(2*i+3)
            res(idx2+1) = p(2*i+2)
            idx1 = idx1 + 2
            idx2 = idx2 - 2
        end do
        
        if (i .NE. ndone) then
            res(idx1) = p(2 * i)
            res(idx1+1) = p(2 * i + 1)
        end if
    end subroutine
    
    subroutine calc_first_half(n, res)
        implicit none
        integer(8), intent(in) :: n
        real(8), intent(inout) :: res(0:2*n-1)
        ! local
        integer(8) :: ndone, i4, in, i, xm
        real(8) :: p(0:2*n-1)
        
        ndone = ishft(n+1, -1)
        p = res + n - 1._8
        
        call calc_first_octant(ishft(n, 2), p)
        
        i4 = 0_8
        in = n
        i = 1_8
        
        do while (i4 <= in - i4)  ! octant 0
            res(2*i) = p(2*i4)
            res(2*i+1) = p(2*i4+1)
            
            i = i + 1
            i4 = i4 + 4
        end do
        do while (i4-in <= 0)  ! octant 1
            xm = in - i4
            res(2*i) = p(2*xm+1)
            res(2*i+1) = p(2*xm)
            
            i = i + 1
            i4 = i4 + 4
        end do
        do while (i4 <= 3 * in-i4)  ! octant 2
            xm = i4 - in
            res(2*i) = -p(2*xm+1)
            res(2*i+1) = p(2*xm)
            
            i = i + 1
            i4 = i4 + 4
        end do
        do while (i < ndone)
            xm = 2 * in - i4
            res(2*i) = -p(2*xm)
            res(2*i+1) = p(2*xm+1)
        end do
    end subroutine
    
    subroutine fill_first_quadrant(n, res)
        implicit none
        integer(8), intent(in) :: n
        real(8), intent(inout) :: res(0:2 * n - 1)  ! adjust limits
        ! local
        real(8), parameter :: hsqt2 = 0.707106781186547524400844362104849
        integer(8) :: quart, i, j
        
        quart = ishft(n, -2)
        
        if (iand(n, 7) == 0) then
            res(quart) = hsqt2
            res(quart + 1) = hsqt2
        end if
        
        j = 2 * quart - 2
        do i=2, quart-1, 2
            res(j) = res(i + 1)
            res(j+1) = res(i)
            j = j - 2
        end do
    end subroutine
        
    
    subroutine fill_first_half(n, res)
        implicit none
        integer(8), intent(in) :: n
        real(8), intent(inout) :: res(0:2 * n - 1)  ! adjust limits
        ! local
        integer(8) :: half, i, j
        
        half = ishft(n, -1)
        
        if (iand(n, 3) == 0) then
            do i=0, half-1, 2
                res(i+half) = -res(i+1)
                res(i+half+1) = res(i)
            end do
        else
            j = 2 * half - 2
            do i=2, half-1, 2
                res(j) = -res(i)
                res(j+1) = res(i+1)
                j = j - 2
            end do
        end if
    end subroutine

    
    subroutine calc_first_octant(den, res)
        implicit none
        integer(8), intent(in) :: den
        real(8), intent(inout) :: res(0:2*den-1)
        !local
        integer(8) :: n, l1, i, start, iend
        real(8) :: cs(2), csx(2)
        
        n = ishft(den + 4, -3)
        
        if (n == 0) return
        res(0) = 1._8
        res(1) = 0._8
        if (n == 1) return
        
        l1 = int(sqrt(real(n, 8)))
        
        do i=1, l1-1
            call my_sincosm1pi((2._8 * i) / den, res(2 * i:2*i+1))
        end do
        
        start = l1
        
        do while (start < n)
            call my_sincosm1pi((2._8 * start) / den, cs)
            res(2*start) = cs(1) + 1._8
            res(2 * start + 1) = cs(2)
            
            iend = l1
            
            if ((start + iend) > n) iend = n - start
            
            do i=1, iend-1
                csx = res(2*i:2*i+1)
                res(2*(start+i)) = ((cs(1) * csx(1) - cs(2)*csx(2) + cs(1)) + csx(1)) + 1._8
                res(2*(start+i) + 1) = (cs(1) * csx(2) + cs(2) * csx(1)) + cs(2) + csx(2)
            end do
            
            start = start + l1
        end do
        
        do i=1, l1-1
            res(2 * i) = res(2 * i) + 1._8
        end do
    end subroutine
    
    
    subroutine my_sincosm1pi(a, res)
        implicit none
        real(8) :: a
        real(8), intent(out) :: res(2)
        ! local
        real(8) :: s, r
        
        s = a * a
        ! approximate cos(pi * x) for x in [-0.25, 0.25]
        r = -1.0369917389758117d-4
        r = (r * s) +  1.9294935641298806d-3
        r = (r * s) -  2.5806887942825395d-2
        r = (r * s) +  2.3533063028328211d-1
        r = (r * s) -  1.3352627688538006d+0
        r = (r * s) +  4.0587121264167623d+0
        r = (r * s) -  4.9348022005446790d+0
        res(1) = r * s  ! cosine
        ! approximate sin(pi * x) for x in [-0.25, 0.25]
        r =            4.6151442520157035d-4
        r = (r * s) -  7.3700183130883555d-3
        r = (r * s) +  8.2145868949323936d-2
        r = (r * s) -  5.9926452893214921d-1
        r = (r * s) +  2.5501640398732688d+0
        r = (r * s) -  5.1677127800499516d+0
        s = s * a
        r = r * s
        res(2) = (a * 3.1415926535897931e+0) + r  ! sine
    end subroutine
            
    
    subroutine rfftp_twsize(tws)
        implicit none
        integer(8), intent(out) :: tws
        ! local
        integer(8) :: l1, k, ip, ido
        
        tws = 0_8
        l1 = 1_8
        
        do k=1, plan%nfct
            ip = plan%fct(k)%fct
            ido = plan%length / (l1 * ip)
            
            tws = tws + (ip - 1) * (ido - 1)
            
            if (ip > 5) tws = tws + 2 * ip
            l1 = l1 * ip
        end do
    end subroutine
        
    
    subroutine rfftp_factorize(ier)
        implicit none
        integer(8), intent(out) :: ier
        ! local
        integer(8) :: length, nfct, tmp, maxl, divisor
        
        length = plan%length
        nfct = 0_8
        
        do while (mod( length, 4) == 0)
            if (nfct >= NFCT_) then
                ier = -1_8
                return
            end if
            nfct = nfct + 1
            plan%fct(nfct)%fct = 4_8
            length = ishft( length, -2)
        end do
        
        if (mod( length, 2) == 0) then
            if (nfct >= NFCT_) then
                ier = -1_8
                return
            end if
            
            length = ishft( length, -1)
            nfct = nfct + 1
            plan%fct(nfct)%fct = 2_8
            
            tmp = plan%fct(1)%fct
            plan%fct(1)%fct = plan%fct(nfct)%fct
            plan%fct(nfct)%fct = tmp
        end if
        
        maxl = int(sqrt(real( length)), 8) + 1
        divisor = 3_8
        
        do while (( length > 1) .AND. (divisor < maxl))
            if (mod( length, divisor) == 0) then
                do while (mod( length, divisor) == 0)
                    if (nfct >= NFCT_) then
                        ier = -1_8
                        return
                    end if
                    
                    nfct = nfct + 1
                    plan%fct(nfct)%fct = divisor
                    length = length / divisor
                end do
                
                maxl = int(sqrt(real( length)), 8) + 1
            end if
            
            divisor = divisor + 2
        end do
        
        if ( length > 1) then
            nfct = nfct + 1
            plan%fct(nfct)%fct = length
        end if
        plan%nfct = nfct
        
        ier = 0_8
    end subroutine
                        
            
        

end module real_fft

            
    
    