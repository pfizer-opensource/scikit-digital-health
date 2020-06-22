! -*- f90 -*-
  
subroutine wfft2(n, x, k, L, step, spec_pow)
  ! n : length of x
  ! x : signal to analyze
  ! k : number of points in the FFT
  ! L : window length in samples, less than k
  ! step : step size in samples
  ! F : output windowed FFT
  implicit none
  integer(8), intent(in) :: n, L, step, k
  real(8), intent(in) :: x(n)
  ! complex(8), intent(out) :: F((n-L)/step + 1, k)
  complex(8), intent(out) :: spec_pow(k)
!f2py intent(hide) :: n
  ! local parameters
  real(8),  parameter :: PI  = 4 * atan(1.0_8)
  integer(8), parameter :: nmin = 32_8
  ! local variables
  complex(8) :: eterm(nmin, nmin)
  integer(8) :: i, j

  do i=1, nmin
    eterm(i, :) = (/( exp(complex(0, -2 * PI * i * j)), j=1, nmin)/)
  end do

  


end
  





