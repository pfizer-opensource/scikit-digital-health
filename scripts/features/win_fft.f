! -*- f90 -*-
      include "fftpack/dcfftf.f"
      include "fftpack/dcffti.f"

! subroutine cfft(x, f, k, n)
!   use, intrinsic :: iso_fortran_env, only: ll=>int64, dp=>real64
!   implicit none
!   integer(ll), intent(in) :: k, n
!   real(dp) :: x(n)
!   complex(8) :: f(k)
! Cf2py   intent(out) :: f
! Cf2py   intent(hide) :: n
!   ! local variables
!   complex(8) :: ws(4 * k + 15)

!   f = cmplx(0.0, kind=8)
!   f(:n) = cmplx(x, kind=8)

!   call dcffti(int(k, 4), ws)
!   call dcfftf(int(k, 4), f, ws)
! end

      subroutine win_fft(n, x, k, L, step, F)
        ! n : length of x
        ! x : signal to analyze
        ! k : number of points in the FFT
        ! L : window length in samples, less than k
        ! step : step size in samples
        ! F : output windowed FFT
        implicit none
        integer(8), intent(in) :: n, L, step, k
        real(8), intent(in) :: x(n)
        complex(8), intent(out) :: F((n-L)/step + 1, k)
        !f2py intent(hide) :: n
        ! local variables
        complex(8) :: ws(4 * k + 15)
        integer(8) :: i, j

        F = complex(0.0_8, 0.0_8)
        call dcffti(int(k, 4), ws)

        do i=1, n-L, step
          j = int(i / step + 1)

          F(j, :L) = cmplx(x(i:i+L-1), kind=8)

          call dcfftf(int(k, 4), F(j, :), ws)
        end do
      end


