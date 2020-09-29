!*****************************************************************************80
!
!! CFFT1I: initialization for CFFT1B and CFFT1F.
!
!  Discussion:
!
!    CFFT1I initializes array WSAVE for use in its companion routines 
!    CFFT1B and CFFT1F.  Routine CFFT1I must be called before the first 
!    call to CFFT1B or CFFT1F, and after whenever the value of integer 
!    N changes.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    22 March 2005
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
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
subroutine cfft1i ( n, wsave, lensav, ier )
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'CFFT1I', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call r4_mcfti1 ( n, wsave, wsave(iw1), wsave(iw1+1) )

  return
end


!*****************************************************************************80
!
!! XERFFT is an error handler for the FFTPACK routines.
!
!  Discussion:
!
!    XERFFT is an error handler for FFTPACK version 5.0 routines.
!    It is called by an FFTPACK 5.0 routine if an input parameter has an
!    invalid value.  A message is printed and execution stops.
!
!    Installers may consider modifying the stop statement in order to
!    call system-specific exception-handling facilities.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, character ( len = * ) SRNAME, the name of the calling routine.
!
!    Input, integer ( kind = 4 ) INFO, an error code.  When a single invalid 
!    parameter in the parameter list of the calling routine has been detected, 
!    INFO is the position of that parameter.  In the case when an illegal 
!    combination of LOT, JUMP, N, and INC has been detected, the calling 
!    subprogram calls XERFFT with INFO = -1.
!
subroutine xerfft ( srname, info )
  implicit none

  integer ( kind = 4 ) info
  character ( len = * ) srname

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'XERFFT - Fatal error!'

  if ( 1 <= info ) then
    write ( *, '(a,a,a,i3,a)') '  On entry to ', trim ( srname ), &
      ' parameter number ', info, ' had an illegal value.'
  else if ( info == -1 ) then
    write( *, '(a,a,a,a)') '  On entry to ', trim ( srname ), &
      ' parameters LOT, JUMP, N and INC are inconsistent.'
  else if ( info == -2 ) then
    write( *, '(a,a,a,a)') '  On entry to ', trim ( srname ), &
      ' parameter L is greater than LDIM.'
  else if ( info == -3 ) then
    write( *, '(a,a,a,a)') '  On entry to ', trim ( srname ), &
      ' parameter M is greater than MDIM.'
  else if ( info == -5 ) then
    write( *, '(a,a,a,a)') '  Within ', trim ( srname ), &
      ' input error returned by lower level routine.'
  else if ( info == -6 ) then
    write( *, '(a,a,a,a)') '  On entry to ', trim ( srname ), &
      ' parameter LDIM is less than 2*(L/2+1).'
  end if

  stop
end


!*****************************************************************************80
!
!! R4_MCFTI1 sets up factors and tables, real single precision arithmetic.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine r4_mcfti1 ( n, wa, fnf, fac )
  implicit none

  real ( kind = 4 ) fac(*)
  real ( kind = 4 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nf
  real ( kind = 4 ) wa(*)
!
!  Get the factorization of N.
!
  call r4_factor ( n, nf, fac )
  fnf = real ( nf, kind = 4 )
  iw = 1
  l1 = 1
!
!  Set up the trigonometric tables.
!
  do k1 = 1, nf
    ip = int ( fac(k1) )
    l2 = l1 * ip
    ido = n / l2
    call r4_tables ( ido, ip, wa(iw) )
    iw = iw + ( ip - 1 ) * ( ido + ido )
    l1 = l2
  end do

  return
end


!*****************************************************************************80
!
!! R4_FACTOR factors of an integer for real single precision computations.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 August 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number for which factorization and 
!    other information is needed.
!
!    Output, integer ( kind = 4 ) NF, the number of factors.
!
!    Output, real ( kind = 4 ) FAC(*), a list of factors of N.
!
subroutine r4_factor ( n, nf, fac )
  implicit none

  real ( kind = 4 ) fac(*)
  integer ( kind = 4 ) j
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  integer ( kind = 4 ) nq
  integer ( kind = 4 ) nr
  integer ( kind = 4 ) ntry

  nl = n
  nf = 0
  j = 0

  do while ( 1 < nl )

    j = j + 1

    if ( j == 1 ) then
      ntry = 4
    else if ( j == 2 ) then
      ntry = 2
    else if ( j == 3 ) then
      ntry = 3
    else if ( j == 4 ) then
      ntry = 5
    else
      ntry = ntry + 2
    end if

    do

      nq = nl / ntry
      nr = nl - ntry * nq

      if ( nr /= 0 ) then
        exit
      end if

      nf = nf + 1
      fac(nf) = real ( ntry, kind = 4 )
      nl = nq

    end do

  end do

  return
end


!*****************************************************************************80
!
!! R4_TABLES computes trigonometric tables, real single precision arithmetic.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 August 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine r4_tables ( ido, ip, wa )
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip

  real ( kind = 4 ) arg1
  real ( kind = 4 ) arg2
  real ( kind = 4 ) arg3
  real ( kind = 4 ) arg4
  real ( kind = 4 ) argz
  integer ( kind = 4 ) i
  integer ( kind = 4 ) j
  real ( kind = 4 ) tpi
  real ( kind = 4 ) wa(ido,ip-1,2)

  tpi = 8.0E+00 * atan ( 1.0E+00 )
  argz = tpi / real ( ip, kind = 4 )
  arg1 = tpi / real ( ido * ip, kind = 4 )

  do j = 2, ip

    arg2 = real ( j - 1, kind = 4 ) * arg1

    do i = 1, ido
      arg3 = real ( i - 1, kind = 4 ) * arg2
      wa(i,j-1,1) = cos ( arg3 )
      wa(i,j-1,2) = sin ( arg3 )
    end do

    if ( 5 < ip ) then
      arg4 = real ( j - 1, kind = 4 ) * argz
      wa(1,j-1,1) = cos ( arg4 )
      wa(1,j-1,2) = sin ( arg4 )
    end if

  end do

  return
end


!*****************************************************************************80
!
!! CFFT1F: complex single precision forward fast Fourier transform, 1D.
!
!  Discussion:
!
!    CFFT1F computes the one-dimensional Fourier transform of a single 
!    periodic sequence within a complex array.  This transform is referred 
!    to as the forward transform or Fourier analysis, transforming the 
!    sequence from physical to spectral space.
!
!    This transform is normalized since a call to CFFT1F followed
!    by a call to CFFT1B (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    22 March 2005
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
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
subroutine cfft1f ( n, inc, c, lenc, wsave, lensav, work, lenwrk, ier )
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 4 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lenc < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'CFFT1F', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'CFFT1F', 8 )
    return
  end if

  if ( lenwrk < 2 * n ) then
    ier = 3
    call xerfft ( 'CFFT1F', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call c1fm1f ( n, inc, c, work, wsave, wsave(iw1), wsave(iw1+1) )

  return
end


!*****************************************************************************80
!
!! C1FM1F is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine c1fm1f ( n, inc, c, ch, wa, fnf, fac )
  implicit none

  complex ( kind = 4 ) c(*)
  real ( kind = 4 ) ch(*)
  real ( kind = 4 ) fac(*)
  real ( kind = 4 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) inc2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) lid
  integer ( kind = 4 ) n
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nbr
  integer ( kind = 4 ) nf
  real ( kind = 4 ) wa(*)

  inc2 = inc + inc
  nf = int ( fnf )
  na = 0
  l1 = 1
  iw = 1

  do k1 = 1, nf

     ip = int ( fac(k1) )
     l2 = ip * l1
     ido = n / l2
     lid = l1 * ido
     nbr = 1 + na + 2 * min ( ip - 2, 4 )

     if ( nbr == 1 ) then
       call c1f2kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 2 ) then
       call c1f2kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 3 ) then
       call c1f3kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 4 ) then
       call c1f3kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 5 ) then
       call c1f4kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 6 ) then
       call c1f4kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 7 ) then
       call c1f5kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 8 ) then
       call c1f5kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 9 ) then
       call c1fgkf ( ido, ip, l1, lid, na, c, c, inc2, ch, ch, 1, wa(iw) )
     else if ( nbr == 10 ) then
       call c1fgkf ( ido, ip, l1, lid, na, ch, ch, 2, c, c, inc2, wa(iw) )
     end if

     l1 = l2
     iw = iw + ( ip - 1 ) * ( ido + ido )

     if ( ip <= 5 ) then
       na = 1 - na
     end if

  end do

  return
end


!*****************************************************************************80
!
!! C1F2KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine c1f2kf ( ido, l1, na, cc, in1, ch, in2, wa )
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,l1,ido,2)
  real ( kind = 4 ) ch(in2,l1,2,ido)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,1,2)

  if ( 1 < ido ) then

    do k = 1, l1
      ch(1,k,1,1) = cc(1,k,1,1) + cc(1,k,1,2)
      ch(1,k,2,1) = cc(1,k,1,1) - cc(1,k,1,2)
      ch(2,k,1,1) = cc(2,k,1,1) + cc(2,k,1,2)
      ch(2,k,2,1) = cc(2,k,1,1) - cc(2,k,1,2)
    end do

    do i = 2, ido
      do k = 1, l1
        ch(1,k,1,i) = cc(1,k,i,1) + cc(1,k,i,2)
        tr2         = cc(1,k,i,1) - cc(1,k,i,2)
        ch(2,k,1,i) = cc(2,k,i,1) + cc(2,k,i,2)
        ti2         = cc(2,k,i,1) - cc(2,k,i,2)
        ch(2,k,2,i) = wa(i,1,1) * ti2 - wa(i,1,2) * tr2
        ch(1,k,2,i) = wa(i,1,1) * tr2 + wa(i,1,2) * ti2
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 2 * l1, kind = 4 )

    do k = 1, l1
      ch(1,k,1,1) = sn * ( cc(1,k,1,1) + cc(1,k,1,2) )
      ch(1,k,2,1) = sn * ( cc(1,k,1,1) - cc(1,k,1,2) )
      ch(2,k,1,1) = sn * ( cc(2,k,1,1) + cc(2,k,1,2) )
      ch(2,k,2,1) = sn * ( cc(2,k,1,1) - cc(2,k,1,2) )
    end do

  else

    sn = 1.0E+00 / real ( 2 * l1, kind = 4 )

    do k = 1, l1

      chold1      = sn * ( cc(1,k,1,1) + cc(1,k,1,2) )
      cc(1,k,1,2) = sn * ( cc(1,k,1,1) - cc(1,k,1,2) )
      cc(1,k,1,1) = chold1

      chold2      = sn * ( cc(2,k,1,1) + cc(2,k,1,2) )
      cc(2,k,1,2) = sn * ( cc(2,k,1,1) - cc(2,k,1,2) )
      cc(2,k,1,1) = chold2

    end do

  end if

  return
end

!*****************************************************************************80
!
!! C1F3KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine c1f3kf ( ido, l1, na, cc, in1, ch, in2, wa )
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,l1,ido,3)
  real ( kind = 4 ) ch(in2,l1,3,ido)
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) di2
  real ( kind = 4 ) di3
  real ( kind = 4 ) dr2
  real ( kind = 4 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ), parameter :: taui = -0.866025403784439E+00
  real ( kind = 4 ), parameter :: taur = -0.5E+00
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,2,2)

  if ( 1 < ido ) then

    do k = 1, l1

      tr2 = cc(1,k,1,2)+cc(1,k,1,3)
      cr2 = cc(1,k,1,1)+taur*tr2
      ch(1,k,1,1) = cc(1,k,1,1)+tr2
      ti2 = cc(2,k,1,2)+cc(2,k,1,3)
      ci2 = cc(2,k,1,1)+taur*ti2
      ch(2,k,1,1) = cc(2,k,1,1)+ti2
      cr3 = taui*(cc(1,k,1,2)-cc(1,k,1,3))
      ci3 = taui*(cc(2,k,1,2)-cc(2,k,1,3))

      ch(1,k,2,1) = cr2 - ci3
      ch(1,k,3,1) = cr2 + ci3
      ch(2,k,2,1) = ci2 + cr3
      ch(2,k,3,1) = ci2 - cr3

    end do

    do i = 2, ido
      do k = 1, l1

        tr2 = cc(1,k,i,2)+cc(1,k,i,3)
        cr2 = cc(1,k,i,1)+taur*tr2
        ch(1,k,1,i) = cc(1,k,i,1)+tr2
        ti2 = cc(2,k,i,2)+cc(2,k,i,3)
        ci2 = cc(2,k,i,1)+taur*ti2
        ch(2,k,1,i) = cc(2,k,i,1)+ti2
        cr3 = taui*(cc(1,k,i,2)-cc(1,k,i,3))
        ci3 = taui*(cc(2,k,i,2)-cc(2,k,i,3))

        dr2 = cr2 - ci3
        dr3 = cr2 + ci3
        di2 = ci2 + cr3
        di3 = ci2 - cr3

        ch(2,k,2,i) = wa(i,1,1) * di2 - wa(i,1,2) * dr2
        ch(1,k,2,i) = wa(i,1,1) * dr2 + wa(i,1,2) * di2
        ch(2,k,3,i) = wa(i,2,1) * di3 - wa(i,2,2) * dr3
        ch(1,k,3,i) = wa(i,2,1) * dr3 + wa(i,2,2) * di3

       end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 3 * l1, kind = 4 )

    do k = 1, l1
      tr2 = cc(1,k,1,2)+cc(1,k,1,3)
      cr2 = cc(1,k,1,1)+taur*tr2
      ch(1,k,1,1) = sn*(cc(1,k,1,1)+tr2)
      ti2 = cc(2,k,1,2)+cc(2,k,1,3)
      ci2 = cc(2,k,1,1)+taur*ti2
      ch(2,k,1,1) = sn*(cc(2,k,1,1)+ti2)
      cr3 = taui*(cc(1,k,1,2)-cc(1,k,1,3))
      ci3 = taui*(cc(2,k,1,2)-cc(2,k,1,3))

      ch(1,k,2,1) = sn*(cr2-ci3)
      ch(1,k,3,1) = sn*(cr2+ci3)
      ch(2,k,2,1) = sn*(ci2+cr3)
      ch(2,k,3,1) = sn*(ci2-cr3)

    end do

  else

    sn = 1.0E+00 / real ( 3 * l1, kind = 4 )

    do k = 1, l1

      tr2 = cc(1,k,1,2)+cc(1,k,1,3)
      cr2 = cc(1,k,1,1)+taur*tr2
      cc(1,k,1,1) = sn*(cc(1,k,1,1)+tr2)
      ti2 = cc(2,k,1,2)+cc(2,k,1,3)
      ci2 = cc(2,k,1,1)+taur*ti2
      cc(2,k,1,1) = sn*(cc(2,k,1,1)+ti2)
      cr3 = taui*(cc(1,k,1,2)-cc(1,k,1,3))
      ci3 = taui*(cc(2,k,1,2)-cc(2,k,1,3))

      cc(1,k,1,2) = sn*(cr2-ci3)
      cc(1,k,1,3) = sn*(cr2+ci3)
      cc(2,k,1,2) = sn*(ci2+cr3)
      cc(2,k,1,3) = sn*(ci2-cr3)

    end do

  end if

  return
end


!*****************************************************************************80
!
!! C1F4KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine c1f4kf ( ido, l1, na, cc, in1, ch, in2, wa )
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,l1,ido,4)
  real ( kind = 4 ) ch(in2,l1,4,ido)
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) ci4
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ) ti1
  real ( kind = 4 ) ti2
  real ( kind = 4 ) ti3
  real ( kind = 4 ) ti4
  real ( kind = 4 ) tr1
  real ( kind = 4 ) tr2
  real ( kind = 4 ) tr3
  real ( kind = 4 ) tr4
  real ( kind = 4 ) wa(ido,3,2)

  if ( 1 < ido ) then

    do k = 1, l1

      ti1 = cc(2,k,1,1)-cc(2,k,1,3)
      ti2 = cc(2,k,1,1)+cc(2,k,1,3)
      tr4 = cc(2,k,1,2)-cc(2,k,1,4)
      ti3 = cc(2,k,1,2)+cc(2,k,1,4)
      tr1 = cc(1,k,1,1)-cc(1,k,1,3)
      tr2 = cc(1,k,1,1)+cc(1,k,1,3)
      ti4 = cc(1,k,1,4)-cc(1,k,1,2)
      tr3 = cc(1,k,1,2)+cc(1,k,1,4)

      ch(1,k,1,1) = tr2 + tr3
      ch(1,k,3,1) = tr2 - tr3
      ch(2,k,1,1) = ti2 + ti3
      ch(2,k,3,1) = ti2 - ti3
      ch(1,k,2,1) = tr1 + tr4
      ch(1,k,4,1) = tr1 - tr4
      ch(2,k,2,1) = ti1 + ti4
      ch(2,k,4,1) = ti1 - ti4

    end do

    do i = 2, ido
      do k = 1, l1
        ti1 = cc(2,k,i,1)-cc(2,k,i,3)
        ti2 = cc(2,k,i,1)+cc(2,k,i,3)
        ti3 = cc(2,k,i,2)+cc(2,k,i,4)
        tr4 = cc(2,k,i,2)-cc(2,k,i,4)
        tr1 = cc(1,k,i,1)-cc(1,k,i,3)
        tr2 = cc(1,k,i,1)+cc(1,k,i,3)
        ti4 = cc(1,k,i,4)-cc(1,k,i,2)
        tr3 = cc(1,k,i,2)+cc(1,k,i,4)
        ch(1,k,1,i) = tr2+tr3
        cr3 = tr2-tr3
        ch(2,k,1,i) = ti2+ti3
        ci3 = ti2-ti3
        cr2 = tr1+tr4
        cr4 = tr1-tr4
        ci2 = ti1+ti4
        ci4 = ti1-ti4
        ch(1,k,2,i) = wa(i,1,1)*cr2+wa(i,1,2)*ci2
        ch(2,k,2,i) = wa(i,1,1)*ci2-wa(i,1,2)*cr2
        ch(1,k,3,i) = wa(i,2,1)*cr3+wa(i,2,2)*ci3
        ch(2,k,3,i) = wa(i,2,1)*ci3-wa(i,2,2)*cr3
        ch(1,k,4,i) = wa(i,3,1)*cr4+wa(i,3,2)*ci4
        ch(2,k,4,i) = wa(i,3,1)*ci4-wa(i,3,2)*cr4
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 4 * l1, kind = 4 )

    do k = 1, l1
      ti1 = cc(2,k,1,1)-cc(2,k,1,3)
      ti2 = cc(2,k,1,1)+cc(2,k,1,3)
      tr4 = cc(2,k,1,2)-cc(2,k,1,4)
      ti3 = cc(2,k,1,2)+cc(2,k,1,4)
      tr1 = cc(1,k,1,1)-cc(1,k,1,3)
      tr2 = cc(1,k,1,1)+cc(1,k,1,3)
      ti4 = cc(1,k,1,4)-cc(1,k,1,2)
      tr3 = cc(1,k,1,2)+cc(1,k,1,4)
      ch(1,k,1,1) = sn*(tr2+tr3)
      ch(1,k,3,1) = sn*(tr2-tr3)
      ch(2,k,1,1) = sn*(ti2+ti3)
      ch(2,k,3,1) = sn*(ti2-ti3)
      ch(1,k,2,1) = sn*(tr1+tr4)
      ch(1,k,4,1) = sn*(tr1-tr4)
      ch(2,k,2,1) = sn*(ti1+ti4)
      ch(2,k,4,1) = sn*(ti1-ti4)
    end do

  else

    sn = 1.0E+00 / real ( 4 * l1, kind = 4 )

    do k = 1, l1
      ti1 = cc(2,k,1,1)-cc(2,k,1,3)
      ti2 = cc(2,k,1,1)+cc(2,k,1,3)
      tr4 = cc(2,k,1,2)-cc(2,k,1,4)
      ti3 = cc(2,k,1,2)+cc(2,k,1,4)
      tr1 = cc(1,k,1,1)-cc(1,k,1,3)
      tr2 = cc(1,k,1,1)+cc(1,k,1,3)
      ti4 = cc(1,k,1,4)-cc(1,k,1,2)
      tr3 = cc(1,k,1,2)+cc(1,k,1,4)
      cc(1,k,1,1) = sn*(tr2+tr3)
      cc(1,k,1,3) = sn*(tr2-tr3)
      cc(2,k,1,1) = sn*(ti2+ti3)
      cc(2,k,1,3) = sn*(ti2-ti3)
      cc(1,k,1,2) = sn*(tr1+tr4)
      cc(1,k,1,4) = sn*(tr1-tr4)
      cc(2,k,1,2) = sn*(ti1+ti4)
      cc(2,k,1,4) = sn*(ti1-ti4)
    end do

  end if

  return
end


!*****************************************************************************80
!
!! C1F5KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine c1f5kf ( ido, l1, na, cc, in1, ch, in2, wa )
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,l1,ido,5)
  real ( kind = 4 ) ch(in2,l1,5,ido)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) ci4
  real ( kind = 4 ) ci5
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) cr4
  real ( kind = 4 ) cr5
  real ( kind = 4 ) di2
  real ( kind = 4 ) di3
  real ( kind = 4 ) di4
  real ( kind = 4 ) di5
  real ( kind = 4 ) dr2
  real ( kind = 4 ) dr3
  real ( kind = 4 ) dr4
  real ( kind = 4 ) dr5
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ) ti2
  real ( kind = 4 ) ti3
  real ( kind = 4 ) ti4
  real ( kind = 4 ) ti5
  real ( kind = 4 ), parameter :: ti11 = -0.9510565162951536E+00
  real ( kind = 4 ), parameter :: ti12 = -0.5877852522924731E+00
  real ( kind = 4 ) tr2
  real ( kind = 4 ) tr3
  real ( kind = 4 ) tr4
  real ( kind = 4 ) tr5
  real ( kind = 4 ), parameter :: tr11 =  0.3090169943749474E+00
  real ( kind = 4 ), parameter :: tr12 = -0.8090169943749474E+00
  real ( kind = 4 ) wa(ido,4,2)

  if ( 1 < ido ) then

    do k = 1, l1

      ti5 = cc(2,k,1,2)-cc(2,k,1,5)
      ti2 = cc(2,k,1,2)+cc(2,k,1,5)
      ti4 = cc(2,k,1,3)-cc(2,k,1,4)
      ti3 = cc(2,k,1,3)+cc(2,k,1,4)
      tr5 = cc(1,k,1,2)-cc(1,k,1,5)
      tr2 = cc(1,k,1,2)+cc(1,k,1,5)
      tr4 = cc(1,k,1,3)-cc(1,k,1,4)
      tr3 = cc(1,k,1,3)+cc(1,k,1,4)

      ch(1,k,1,1) = cc(1,k,1,1)+tr2+tr3
      ch(2,k,1,1) = cc(2,k,1,1)+ti2+ti3
      cr2 = cc(1,k,1,1)+tr11*tr2+tr12*tr3
      ci2 = cc(2,k,1,1)+tr11*ti2+tr12*ti3
      cr3 = cc(1,k,1,1)+tr12*tr2+tr11*tr3
      ci3 = cc(2,k,1,1)+tr12*ti2+tr11*ti3
      cr5 = ti11*tr5+ti12*tr4
      ci5 = ti11*ti5+ti12*ti4
      cr4 = ti12*tr5-ti11*tr4
      ci4 = ti12*ti5-ti11*ti4
      ch(1,k,2,1) = cr2-ci5
      ch(1,k,5,1) = cr2+ci5
      ch(2,k,2,1) = ci2+cr5
      ch(2,k,3,1) = ci3+cr4
      ch(1,k,3,1) = cr3-ci4
      ch(1,k,4,1) = cr3+ci4
      ch(2,k,4,1) = ci3-cr4
      ch(2,k,5,1) = ci2-cr5
    end do

    do i = 2, ido
      do k = 1, l1

        ti5 = cc(2,k,i,2)-cc(2,k,i,5)
        ti2 = cc(2,k,i,2)+cc(2,k,i,5)
        ti4 = cc(2,k,i,3)-cc(2,k,i,4)
        ti3 = cc(2,k,i,3)+cc(2,k,i,4)
        tr5 = cc(1,k,i,2)-cc(1,k,i,5)
        tr2 = cc(1,k,i,2)+cc(1,k,i,5)
        tr4 = cc(1,k,i,3)-cc(1,k,i,4)
        tr3 = cc(1,k,i,3)+cc(1,k,i,4)

        ch(1,k,1,i) = cc(1,k,i,1)+tr2+tr3
        ch(2,k,1,i) = cc(2,k,i,1)+ti2+ti3
        cr2 = cc(1,k,i,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,k,i,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,k,i,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,k,i,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        dr3 = cr3-ci4
        dr4 = cr3+ci4
        di3 = ci3+cr4
        di4 = ci3-cr4
        dr5 = cr2+ci5
        dr2 = cr2-ci5
        di5 = ci2-cr5
        di2 = ci2+cr5
        ch(1,k,2,i) = wa(i,1,1)*dr2+wa(i,1,2)*di2
        ch(2,k,2,i) = wa(i,1,1)*di2-wa(i,1,2)*dr2
        ch(1,k,3,i) = wa(i,2,1)*dr3+wa(i,2,2)*di3
        ch(2,k,3,i) = wa(i,2,1)*di3-wa(i,2,2)*dr3
        ch(1,k,4,i) = wa(i,3,1)*dr4+wa(i,3,2)*di4
        ch(2,k,4,i) = wa(i,3,1)*di4-wa(i,3,2)*dr4
        ch(1,k,5,i) = wa(i,4,1)*dr5+wa(i,4,2)*di5
        ch(2,k,5,i) = wa(i,4,1)*di5-wa(i,4,2)*dr5
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 5 * l1, kind = 4 )

    do k = 1, l1

      ti5 = cc(2,k,1,2)-cc(2,k,1,5)
      ti2 = cc(2,k,1,2)+cc(2,k,1,5)
      ti4 = cc(2,k,1,3)-cc(2,k,1,4)
      ti3 = cc(2,k,1,3)+cc(2,k,1,4)
      tr5 = cc(1,k,1,2)-cc(1,k,1,5)
      tr2 = cc(1,k,1,2)+cc(1,k,1,5)
      tr4 = cc(1,k,1,3)-cc(1,k,1,4)
      tr3 = cc(1,k,1,3)+cc(1,k,1,4)

      ch(1,k,1,1) = sn*(cc(1,k,1,1)+tr2+tr3)
      ch(2,k,1,1) = sn*(cc(2,k,1,1)+ti2+ti3)

      cr2 = cc(1,k,1,1)+tr11*tr2+tr12*tr3
      ci2 = cc(2,k,1,1)+tr11*ti2+tr12*ti3
      cr3 = cc(1,k,1,1)+tr12*tr2+tr11*tr3
      ci3 = cc(2,k,1,1)+tr12*ti2+tr11*ti3
      cr5 = ti11*tr5+ti12*tr4
      ci5 = ti11*ti5+ti12*ti4
      cr4 = ti12*tr5-ti11*tr4
      ci4 = ti12*ti5-ti11*ti4

      ch(1,k,2,1) = sn*(cr2-ci5)
      ch(1,k,5,1) = sn*(cr2+ci5)
      ch(2,k,2,1) = sn*(ci2+cr5)
      ch(2,k,3,1) = sn*(ci3+cr4)
      ch(1,k,3,1) = sn*(cr3-ci4)
      ch(1,k,4,1) = sn*(cr3+ci4)
      ch(2,k,4,1) = sn*(ci3-cr4)
      ch(2,k,5,1) = sn*(ci2-cr5)

    end do

  else

    sn = 1.0E+00 / real ( 5 * l1, kind = 4 )

    do k = 1, l1

      ti5 = cc(2,k,1,2)-cc(2,k,1,5)
      ti2 = cc(2,k,1,2)+cc(2,k,1,5)
      ti4 = cc(2,k,1,3)-cc(2,k,1,4)
      ti3 = cc(2,k,1,3)+cc(2,k,1,4)
      tr5 = cc(1,k,1,2)-cc(1,k,1,5)
      tr2 = cc(1,k,1,2)+cc(1,k,1,5)
      tr4 = cc(1,k,1,3)-cc(1,k,1,4)
      tr3 = cc(1,k,1,3)+cc(1,k,1,4)

      chold1 = sn*(cc(1,k,1,1)+tr2+tr3)
      chold2 = sn*(cc(2,k,1,1)+ti2+ti3)

      cr2 = cc(1,k,1,1)+tr11*tr2+tr12*tr3
      ci2 = cc(2,k,1,1)+tr11*ti2+tr12*ti3
      cr3 = cc(1,k,1,1)+tr12*tr2+tr11*tr3
      ci3 = cc(2,k,1,1)+tr12*ti2+tr11*ti3

      cc(1,k,1,1) = chold1
      cc(2,k,1,1) = chold2

      cr5 = ti11*tr5+ti12*tr4
      ci5 = ti11*ti5+ti12*ti4
      cr4 = ti12*tr5-ti11*tr4
      ci4 = ti12*ti5-ti11*ti4

      cc(1,k,1,2) = sn*(cr2-ci5)
      cc(1,k,1,5) = sn*(cr2+ci5)
      cc(2,k,1,2) = sn*(ci2+cr5)
      cc(2,k,1,3) = sn*(ci3+cr4)
      cc(1,k,1,3) = sn*(cr3-ci4)
      cc(1,k,1,4) = sn*(cr3+ci4)
      cc(2,k,1,4) = sn*(ci3-cr4)
      cc(2,k,1,5) = sn*(ci2-cr5)

    end do

  end if

  return
end


!*****************************************************************************80
!
!! C1FGKF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Paul Swarztrauber
!    Richard Valent
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
subroutine c1fgkf ( ido, ip, l1, lid, na, cc, cc1, in1, ch, ch1, in2, wa )
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 4 ) cc(in1,l1,ip,ido)
  real ( kind = 4 ) cc1(in1,lid,ip)
  real ( kind = 4 ) ch(in2,l1,ido,ip)
  real ( kind = 4 ) ch1(in2,lid,ip)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idlj
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) j
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) ki
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ) wa(ido,ip-1,2)
  real ( kind = 4 ) wai
  real ( kind = 4 ) war

  ipp2 = ip+2
  ipph = (ip+1)/2

  do ki = 1, lid
    ch1(1,ki,1) = cc1(1,ki,1)
    ch1(2,ki,1) = cc1(2,ki,1)
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      ch1(1,ki,j) =  cc1(1,ki,j)+cc1(1,ki,jc)
      ch1(1,ki,jc) = cc1(1,ki,j)-cc1(1,ki,jc)
      ch1(2,ki,j) =  cc1(2,ki,j)+cc1(2,ki,jc)
      ch1(2,ki,jc) = cc1(2,ki,j)-cc1(2,ki,jc)
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      cc1(1,ki,1) = cc1(1,ki,1) + ch1(1,ki,j)
      cc1(2,ki,1) = cc1(2,ki,1) + ch1(2,ki,j)
    end do
  end do

  do l = 2, ipph

    lc = ipp2 - l

    do ki = 1, lid
      cc1(1,ki,l)  = ch1(1,ki,1) + wa(1,l-1,1) * ch1(1,ki,2)
      cc1(1,ki,lc) =             - wa(1,l-1,2) * ch1(1,ki,ip)
      cc1(2,ki,l)  = ch1(2,ki,1) + wa(1,l-1,1) * ch1(2,ki,2)
      cc1(2,ki,lc) =             - wa(1,l-1,2) * ch1(2,ki,ip)
    end do

    do j = 3, ipph

      jc = ipp2 - j
      idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
      war = wa(1,idlj,1)
      wai = -wa(1,idlj,2)

      do ki = 1, lid
        cc1(1,ki,l) = cc1(1,ki,l)+war*ch1(1,ki,j)
        cc1(1,ki,lc) = cc1(1,ki,lc)+wai*ch1(1,ki,jc)
        cc1(2,ki,l) = cc1(2,ki,l)+war*ch1(2,ki,j)
        cc1(2,ki,lc) = cc1(2,ki,lc)+wai*ch1(2,ki,jc)
      end do

    end do

  end do

  if ( 1 < ido ) then

    do ki = 1, lid
      ch1(1,ki,1) = cc1(1,ki,1)
      ch1(2,ki,1) = cc1(2,ki,1)
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        ch1(1,ki,j) = cc1(1,ki,j)-cc1(2,ki,jc)
        ch1(2,ki,j) = cc1(2,ki,j)+cc1(1,ki,jc)
        ch1(1,ki,jc) = cc1(1,ki,j)+cc1(2,ki,jc)
        ch1(2,ki,jc) = cc1(2,ki,j)-cc1(1,ki,jc)
      end do
    end do

    do i = 1, ido
      do k = 1, l1
        cc(1,k,1,i) = ch(1,k,i,1)
        cc(2,k,1,i) = ch(2,k,i,1)
      end do
    end do

    do j = 2, ip
      do k = 1, l1
        cc(1,k,j,1) = ch(1,k,1,j)
        cc(2,k,j,1) = ch(2,k,1,j)
      end do
    end do

    do j = 2, ip
      do i = 2, ido
        do k = 1, l1
          cc(1,k,j,i) = wa(i,j-1,1)*ch(1,k,i,j) + wa(i,j-1,2)*ch(2,k,i,j)
          cc(2,k,j,i) = wa(i,j-1,1)*ch(2,k,i,j) - wa(i,j-1,2)*ch(1,k,i,j)
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( ip * l1, kind = 4 )

    do ki = 1, lid
      ch1(1,ki,1) = sn * cc1(1,ki,1)
      ch1(2,ki,1) = sn * cc1(2,ki,1)
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        ch1(1,ki,j) =  sn * ( cc1(1,ki,j) - cc1(2,ki,jc) )
        ch1(2,ki,j) =  sn * ( cc1(2,ki,j) + cc1(1,ki,jc) )
        ch1(1,ki,jc) = sn * ( cc1(1,ki,j) + cc1(2,ki,jc) )
        ch1(2,ki,jc) = sn * ( cc1(2,ki,j) - cc1(1,ki,jc) )
      end do
    end do

  else

    sn = 1.0E+00 / real ( ip * l1, kind = 4 )

    do ki = 1, lid
      cc1(1,ki,1) = sn * cc1(1,ki,1)
      cc1(2,ki,1) = sn * cc1(2,ki,1)
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        chold1 = sn*(cc1(1,ki,j)-cc1(2,ki,jc))
        chold2 = sn*(cc1(1,ki,j)+cc1(2,ki,jc))
        cc1(1,ki,j) = chold1
        cc1(2,ki,jc) = sn*(cc1(2,ki,j)-cc1(1,ki,jc))
        cc1(2,ki,j) = sn*(cc1(2,ki,j)+cc1(1,ki,jc))
        cc1(1,ki,jc) = chold2
      end do
    end do

  end if

  return
end