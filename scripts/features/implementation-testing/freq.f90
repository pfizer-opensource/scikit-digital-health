! -*- f90 -*-
include "fftpack3.f90"

!subroutine cfft1f ( n, inc, c, lenc, wsave, lensav, work, lenwrk, ier )
!*****************************************************************************80
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array C, of two consecutive elements within the sequence to be transformed.
!
!    Input/output, complex ( kind = 4 ) C(LENC) containing the sequence to 
!    be transformed.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least INC*(N-1) + 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to CFFT1I before the first call to routine CFFT1F 
!    or CFFT1B for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to CFFT1F and CFFT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENC   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!

!subroutine cfft1i ( n, wsave, lensav, ier )
!*****************************************************************************80
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product 
!    of small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors 
!    of N and  also containing certain trigonometric values which will be used 
!    in routines CFFT1B or CFFT1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.
subroutine fft(n, x, nfft, res)
    implicit none
    integer(4), intent(in) :: n, nfft
    real(4), intent(in) :: x(n)
    complex(4), intent(out) :: res(2 * nfft)
!f2py intent(hide) :: n
    real(4) :: wsave(2 * 2 * nfft + int(log(real(2 * nfft))) + 4)
    integer(4) :: lensav
    real(4) :: wrk(2 * 2 * nfft)
    integer(4) :: lenwrk
    integer(4) :: ier
    
    lensav = 2 * 2 * nfft + int(log(real(2 * nfft))) + 4
    lenwrk = 2 * 2 * nfft
    
    call cfft1i(2 * nfft, wsave, lensav, ier)

    res = complex(0._4, 0._4)
    res(:n) = cmplx(x, kind=4)
    
    call cfft1f(2 * nfft, 1_4, res, 2 * nfft, wsave, lensav, wrk, lenwrk, ier)
end subroutine