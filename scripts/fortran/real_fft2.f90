! -*- f90 -*-


module real_fft
  use, intrinsic :: iso_fortran_env, only : real64, int64

  integer(int64), parameter, private :: NFCT_ = 25_int64

  type :: rfftp_fctdata
    integer(int64) :: fct
    real(real64), pointer :: tw(:)=>null()
  end type rfftp_fctdata

  type :: rfftp_plan
    integer(int64) :: length = -1_int64
    integer(int64) :: nfct, twsize
    real(real64), pointer :: mem(:)=>null()
    type(rfftp_fctdata) :: fct(NFCT_)
  end type rfftp_plan

  ! the plan
  type(rfftp_plan), private :: plan

contains
  subroutine destroy_play()
    integer :: i

    if (associated(plan%mem)) then
      deallocate(plan%mem)
      nullify(plan%mem)
    end if
    do i=1, NFCT_
      if (associated(plan%fct(i)%tw)) nullify(plan%fct(i)%tw)
    end do
  end subroutine

  subroutine compute_real_forward(n, x, fct, ret, ier)
    integer(int64), intent(in) :: n
    real(real64), intent(in) :: x(n), fct
    real(real64), intent(out) :: ret(n+2)
    integer(int64), intent(out) :: ier
!f2py intent(hide) :: n
    ier = 0_int64  ! initialize for sure at 0

    ! ensure proper power of 2 size
    if (iand(n, n-1) .NE. 0) then
      ier = -2_int64
      return
    end if

    ! check if the plan needs to be computed
    if ((plan%length .NE. n) .OR. (plan%length == -1_int64)) then
      call make_rfftp_plan(n, ier)
    end if
    if (ier .NE. 0_int64) return

    ret = 0._real64
    ret(2:n+1) = x
    call rfftp_forward(n, ret(2:), fct, ier)
    if (ier .NE. 0) return

    ret(1) = ret(2)
    ret(2) = 0._real64
  end subroutine






  subroutine make_rfftp_plan(length, ier)
    integer(int64), intent(in) :: length
    integer(int64), intent(inout) :: ier
    ! local
    integer(int64) :: i

    if (length == 0_int64) then
      ier = -1_int64
      return
    end if

    plan%length = length
    plan%nfct = 0_int64

    do i=1_int64, NFCT_
      plan%fct(i)%fct = 0_int64
    end do

    if (length == 1_int64) return

    call rfftp_factorize(ier)
    if (ier .NE. 0_int64) return

    call rfftp_twsize(plan%twsize, ier)
    if (ier .NE. 0_int64) return

    allocate(plan%mem(plan%twsize))
    plan%mem = 0._real64

    call rfftp_comp_twiddle(length, ier)
    if (ier .NE. 0_int64) return

  end subroutine

  subroutine rfftp_factorize(ier)
    integer(int64), intent(inout) :: ier
    ! local
    integer(int64) :: length, nfct, tmp, maxl, divisor

    length = plan%length
    nfct = 0_int64

    do while (mod(length, 4_int64) == 0_int64)
      if (nfct >= NFCT_) then
        ier = -999_int64
        return
      end if
      nfct = nfct + 1_int64
      plan%fct(nfct)%fct = 4_int64
      length = ishft(length, -2)  ! divide by 4
    end do

    if (mod(length, 2_int64) == 0_int64) then
      if (nfct >= NFCT_) then
        ier = -999_int64
        return
      end if

      length = ishft(length, -1)  ! divide by 2
      nfct = nfct + 1_int64
      plan%fct(nfct)%fct = 2_int64

      ! factor of 2 should be at the front
      tmp = plan%fct(1)%fct
      plan%fct(1)%fct = plan%fct(nfct)%fct
      plan%fct(nfct)%fct = tmp
    end if

    ! since length must be a power of 2, no need to look for other divisors
    if (length > 1) then
      if (iand(length, length-1_int64) /= 0_int64) then
        ier = -2_int64
      end if
      nfct = nfct + 1_int64
      plan%fct(nfct)%fct = length
    end if
    plan%nfct = nfct
  end subroutine








  subroutine rfftp_forward(npts, x, fct, ier)
    integer(int64), intent(in) :: npts
    real(real64), intent(inout), target :: x(npts)
    real(real64), intent(in) :: fct
    integer(int64), intent(inout) :: ier
    ! local
    integer(int64) :: l1, nf, k1, k, ip, ido, iswap
    real(real64), target :: ch(npts)
    real(real64), pointer :: p1(:)=>null(), p2(:)=>null()

    if (plan%length <= 1_int64) then
      ier = -4_int64
      return
    end if

    l1 = npts
    nf = plan%nfct

    iswap = 1_int64  ! keep track of swapping
    p1 => x
    p2 => ch

    do k1=1_int64, nf
      k = nf - k1 + 1
      ip = plan%fct(k)%fct
      ido = n / l1
      l1 = l1 / ip

      if (ip == 4_int64) then
        call radf4(ido, l1, p1, p2, plan%fct(k)%tw)
      else if (ip == 2_int64) then
        call radf2(ido, l1, p1, p2, plan%fct(k)%tw)
      else
        ier = -5_int64
        return
      end if

      ! swap which arrays the pointers point to
      nullify(p1)
      nullify(p2)
      if (iand(iswap, 1_int64) .NE. 0) then
        p1 => ch
        p2 => x
      else
        p1 => x
        p2 => ch
      end if
      iswap = iswap + i_int64
    end do
    
    ! normalize
    if (.NOT. associated(p1, x)) then
      if (fct .NE. 1._real64) then
        x = fct * p1
      else
        x = p1
      end if
    else
      if (fct .NE. 1._real64) then
        x = x * fct
      end if
    end if

    nullify(p1)
    nullify(p2)
  end subroutine


  subroutine radf2(ido, l1, cc, ch, wa)
    integer(int64), intent(in) :: ido, l1
    real(real64), dimension(:) :: cc, ch, wa
    ! local
    integer(int64), parameter :: cdim = 2_int64
    integer(int64) :: k, i, ic
    real(real64) :: tr2, ti2

    do k=0_int64, l1-1_int64
      ch(ido*cdim*k+1) = cc(ido*k+1) + cc(ido*(k+l1)+1)
      ch(ido+ido*(1+cdim*k)) = cc(ido*k+1) - cc(ido*(k+l1)+1)
    end do
    if (iand(ido, 1_int64) == 0_int64) then
      do k=0_int64, l1-1_int64
        ch(ido*(1+cdim*k)+1) = -cc(ido+ido*(k+l1))
        ch(ido+ido*(cdim*k)) = cc(ido+ido*k)
      end do
    end if
    if (ido <= 2_int64) return
    do k=0_int64, l1-1_int64
      do i=2_int64, ido-1, 2_int64
        ic = ido-i

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
    integer(int64), intent(in) :: ido, l1
    real(real64), dimension(:) :: cc, ch, wa
    ! local
    integer(int64), parameter :: cdim=4_int64
    real(real64), parameter :: hsqt2=0.70710678118654752440_real64
    integer(int64) :: k, i, ic
    real(real64) :: ci2, ci3, ci4, cr2, cr3, cr4
    real(real64) :: ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4
    
    do k=0_int64, l1-1_int64
      tr1 = cc(ido*(k+l1*3)+1) + cc(ido*(k+l1)+1)
      ch(ido*(2+cdim*k)+1) = cc(ido*(k+l1*3)+1) - cc(ido*(k+l1)+1)
      
      tr2 = cc(ido*k+1) + cc(ido*(k+l1*2)+1)
      ch(ido+ido*(1+cdim*k)) = cc(ido*k+1) - cc(ido*(k+l1*2)+1)
      
      ch(ido*(cdim*k)+1) = tr2 + tr1
      ch(ido+ido*(3+cdim*k)) = tr2 - tr1
    end do
    if (iand(ido, 1_int64) == 0_int64) then
      do k=0_int64, l1-1_int64
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
end module real_fft