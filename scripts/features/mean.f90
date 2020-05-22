! -*- f90 -*-
module mean2D
  use ISO_FORTRAN_ENV
  implicit none
  ! allocatable variables
  real(real64), allocatable :: mean(:)
contains
  subroutine compute(N, L, x)
    integer(8), intent(in) :: N, L
    
