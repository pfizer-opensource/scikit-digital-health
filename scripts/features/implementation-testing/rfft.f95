! -*- f95 -*-
include "dfft5.f90"



subroutine rfft(n, x, nfft, res)
    implicit none
    integer(8), intent(in) :: n, nfft
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: res(nfft)
!f2py intent(hide) :: n
    real(8) :: wsave(nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav
    real(8) :: wrk(nfft)
    integer(4) :: lenwrk
    integer(4) :: ier
    
    lensav = nfft + int(log(real(2 * nfft))) + 4
    lenwrk = nfft
    
    call dfft1i(nfft, wsave, lensav, ier)

    res = 0._8
    res(:n) = x
    
    call dfft1f(nfft, 1_4, res, nfft, wsave, lensav, wrk, lenwrk, ier)
end subroutine