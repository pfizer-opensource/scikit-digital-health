! -*- f90 -*-

module test
  use, intrinsic :: iso_fortran_env, only : real64, int64
  implicit none
contains


  subroutine fn1(n, a, b)
    integer(int64) :: n
    real(real64) :: a(n), b(n)
!f2py intent(hide) :: n
    integer(int64) :: i

    do i=1, n-2, 2
      call PM(B_(i), B_(i+1), A_(i), A_(i+1))
    end do
  end subroutine
end module test