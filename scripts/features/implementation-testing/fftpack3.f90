subroutine c1f2kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! C1F2KB is an FFTPACK5 auxiliary routine.
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
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,1,2)

  if ( 1 < ido .or. na == 1 ) then

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

        ch(2,k,2,i) = wa(i,1,1) * ti2 + wa(i,1,2) * tr2
        ch(1,k,2,i) = wa(i,1,1) * tr2 - wa(i,1,2) * ti2

      end do
    end do

  else

    do k = 1, l1

      chold1      = cc(1,k,1,1) + cc(1,k,1,2)
      cc(1,k,1,2) = cc(1,k,1,1) - cc(1,k,1,2)
      cc(1,k,1,1) = chold1

      chold2      = cc(2,k,1,1) + cc(2,k,1,2)
      cc(2,k,1,2) = cc(2,k,1,1) - cc(2,k,1,2)
      cc(2,k,1,1) = chold2

    end do

  end if

  return
end
subroutine c1f2kf ( ido, l1, na, cc, in1, ch, in2, wa )

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
subroutine c1f3kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! C1F3KB is an FFTPACK5 auxiliary routine.
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
  real ( kind = 4 ), parameter :: taui =  0.866025403784439E+00
  real ( kind = 4 ), parameter :: taur = -0.5E+00
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,2,2)

  if ( 1 < ido .or. na == 1 ) then

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

        ch(2,k,2,i) = wa(i,1,1) * di2 + wa(i,1,2) * dr2
        ch(1,k,2,i) = wa(i,1,1) * dr2 - wa(i,1,2) * di2
        ch(2,k,3,i) = wa(i,2,1) * di3 + wa(i,2,2) * dr3
        ch(1,k,3,i) = wa(i,2,1) * dr3 - wa(i,2,2) * di3

      end do
    end do

  else

    do k = 1, l1

      tr2 = cc(1,k,1,2)+cc(1,k,1,3)
      cr2 = cc(1,k,1,1)+taur*tr2
      cc(1,k,1,1) = cc(1,k,1,1)+tr2
      ti2 = cc(2,k,1,2)+cc(2,k,1,3)
      ci2 = cc(2,k,1,1)+taur*ti2
      cc(2,k,1,1) = cc(2,k,1,1)+ti2
      cr3 = taui*(cc(1,k,1,2)-cc(1,k,1,3))
      ci3 = taui*(cc(2,k,1,2)-cc(2,k,1,3))

      cc(1,k,1,2) = cr2 - ci3
      cc(1,k,1,3) = cr2 + ci3
      cc(2,k,1,2) = ci2 + cr3
      cc(2,k,1,3) = ci2 - cr3

    end do

  end if

  return
end
subroutine c1f3kf ( ido, l1, na, cc, in1, ch, in2, wa )

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
subroutine c1f4kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! C1F4KB is an FFTPACK5 auxiliary routine.
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
  real ( kind = 4 ) ti1
  real ( kind = 4 ) ti2
  real ( kind = 4 ) ti3
  real ( kind = 4 ) ti4
  real ( kind = 4 ) tr1
  real ( kind = 4 ) tr2
  real ( kind = 4 ) tr3
  real ( kind = 4 ) tr4
  real ( kind = 4 ) wa(ido,3,2)

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      ti1 = cc(2,k,1,1)-cc(2,k,1,3)
      ti2 = cc(2,k,1,1)+cc(2,k,1,3)
      tr4 = cc(2,k,1,4)-cc(2,k,1,2)
      ti3 = cc(2,k,1,2)+cc(2,k,1,4)
      tr1 = cc(1,k,1,1)-cc(1,k,1,3)
      tr2 = cc(1,k,1,1)+cc(1,k,1,3)
      ti4 = cc(1,k,1,2)-cc(1,k,1,4)
      tr3 = cc(1,k,1,2)+cc(1,k,1,4)
      ch(1,k,1,1) = tr2+tr3
      ch(1,k,3,1) = tr2-tr3
      ch(2,k,1,1) = ti2+ti3
      ch(2,k,3,1) = ti2-ti3
      ch(1,k,2,1) = tr1+tr4
      ch(1,k,4,1) = tr1-tr4
      ch(2,k,2,1) = ti1+ti4
      ch(2,k,4,1) = ti1-ti4
    end do

    do i = 2, ido
      do k = 1, l1

        ti1 = cc(2,k,i,1)-cc(2,k,i,3)
        ti2 = cc(2,k,i,1)+cc(2,k,i,3)
        ti3 = cc(2,k,i,2)+cc(2,k,i,4)
        tr4 = cc(2,k,i,4)-cc(2,k,i,2)
        tr1 = cc(1,k,i,1)-cc(1,k,i,3)
        tr2 = cc(1,k,i,1)+cc(1,k,i,3)
        ti4 = cc(1,k,i,2)-cc(1,k,i,4)
        tr3 = cc(1,k,i,2)+cc(1,k,i,4)
        ch(1,k,1,i) = tr2+tr3
        cr3 = tr2-tr3
        ch(2,k,1,i) = ti2+ti3
        ci3 = ti2-ti3
        cr2 = tr1+tr4
        cr4 = tr1-tr4
        ci2 = ti1+ti4
        ci4 = ti1-ti4

        ch(1,k,2,i) = wa(i,1,1)*cr2-wa(i,1,2)*ci2
        ch(2,k,2,i) = wa(i,1,1)*ci2+wa(i,1,2)*cr2
        ch(1,k,3,i) = wa(i,2,1)*cr3-wa(i,2,2)*ci3
        ch(2,k,3,i) = wa(i,2,1)*ci3+wa(i,2,2)*cr3
        ch(1,k,4,i) = wa(i,3,1)*cr4-wa(i,3,2)*ci4
        ch(2,k,4,i) = wa(i,3,1)*ci4+wa(i,3,2)*cr4

       end do
    end do

  else

    do k = 1, l1
       ti1 = cc(2,k,1,1)-cc(2,k,1,3)
       ti2 = cc(2,k,1,1)+cc(2,k,1,3)
       tr4 = cc(2,k,1,4)-cc(2,k,1,2)
       ti3 = cc(2,k,1,2)+cc(2,k,1,4)
       tr1 = cc(1,k,1,1)-cc(1,k,1,3)
       tr2 = cc(1,k,1,1)+cc(1,k,1,3)
       ti4 = cc(1,k,1,2)-cc(1,k,1,4)
       tr3 = cc(1,k,1,2)+cc(1,k,1,4)
       cc(1,k,1,1) = tr2+tr3
       cc(1,k,1,3) = tr2-tr3
       cc(2,k,1,1) = ti2+ti3
       cc(2,k,1,3) = ti2-ti3
       cc(1,k,1,2) = tr1+tr4
       cc(1,k,1,4) = tr1-tr4
       cc(2,k,1,2) = ti1+ti4
       cc(2,k,1,4) = ti1-ti4
    end do

  end if

  return
end
subroutine c1f4kf ( ido, l1, na, cc, in1, ch, in2, wa )

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
subroutine c1f5kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! C1F5KB is an FFTPACK5 auxiliary routine.
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
  real ( kind = 4 ) ti2
  real ( kind = 4 ) ti3
  real ( kind = 4 ) ti4
  real ( kind = 4 ) ti5
  real ( kind = 4 ), parameter :: ti11 =  0.9510565162951536E+00
  real ( kind = 4 ), parameter :: ti12 =  0.5877852522924731E+00
  real ( kind = 4 ) tr2
  real ( kind = 4 ) tr3
  real ( kind = 4 ) tr4
  real ( kind = 4 ) tr5
  real ( kind = 4 ), parameter :: tr11 =  0.3090169943749474E+00
  real ( kind = 4 ), parameter :: tr12 = -0.8090169943749474E+00
  real ( kind = 4 ) wa(ido,4,2)

  if ( 1 < ido .or. na == 1 ) then

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
        ch(1,k,2,i) = wa(i,1,1)*dr2-wa(i,1,2)*di2
        ch(2,k,2,i) = wa(i,1,1)*di2+wa(i,1,2)*dr2
        ch(1,k,3,i) = wa(i,2,1)*dr3-wa(i,2,2)*di3
        ch(2,k,3,i) = wa(i,2,1)*di3+wa(i,2,2)*dr3
        ch(1,k,4,i) = wa(i,3,1)*dr4-wa(i,3,2)*di4
        ch(2,k,4,i) = wa(i,3,1)*di4+wa(i,3,2)*dr4
        ch(1,k,5,i) = wa(i,4,1)*dr5-wa(i,4,2)*di5
        ch(2,k,5,i) = wa(i,4,1)*di5+wa(i,4,2)*dr5
      end do
    end do

  else

    do k = 1, l1
      ti5 = cc(2,k,1,2)-cc(2,k,1,5)
      ti2 = cc(2,k,1,2)+cc(2,k,1,5)
      ti4 = cc(2,k,1,3)-cc(2,k,1,4)
      ti3 = cc(2,k,1,3)+cc(2,k,1,4)
      tr5 = cc(1,k,1,2)-cc(1,k,1,5)
      tr2 = cc(1,k,1,2)+cc(1,k,1,5)
      tr4 = cc(1,k,1,3)-cc(1,k,1,4)
      tr3 = cc(1,k,1,3)+cc(1,k,1,4)
      chold1 = cc(1,k,1,1)+tr2+tr3
      chold2 = cc(2,k,1,1)+ti2+ti3
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
      cc(1,k,1,2) = cr2-ci5
      cc(1,k,1,5) = cr2+ci5
      cc(2,k,1,2) = ci2+cr5
      cc(2,k,1,3) = ci3+cr4
      cc(1,k,1,3) = cr3-ci4
      cc(1,k,1,4) = cr3+ci4
      cc(2,k,1,4) = ci3-cr4
      cc(2,k,1,5) = ci2-cr5
    end do

  end if

  return
end
subroutine c1f5kf ( ido, l1, na, cc, in1, ch, in2, wa )

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
subroutine c1fgkb ( ido, ip, l1, lid, na, cc, cc1, in1, ch, ch1, in2, wa )

!*****************************************************************************80
!
!! C1FGKB is an FFTPACK5 auxiliary routine.
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
  real ( kind = 4 ) wa(ido,ip-1,2)
  real ( kind = 4 ) wai
  real ( kind = 4 ) war

  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2
  do ki = 1, lid
    ch1(1,ki,1) = cc1(1,ki,1)
    ch1(2,ki,1) = cc1(2,ki,1)
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      ch1(1,ki,j) =  cc1(1,ki,j) + cc1(1,ki,jc)
      ch1(1,ki,jc) = cc1(1,ki,j) - cc1(1,ki,jc)
      ch1(2,ki,j) =  cc1(2,ki,j) + cc1(2,ki,jc)
      ch1(2,ki,jc) = cc1(2,ki,j) - cc1(2,ki,jc)
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      cc1(1,ki,1) = cc1(1,ki,1)+ch1(1,ki,j)
      cc1(2,ki,1) = cc1(2,ki,1)+ch1(2,ki,j)
    end do
  end do

  do l = 2, ipph

     lc = ipp2 - l
     do ki = 1, lid
       cc1(1,ki,l) = ch1(1,ki,1)+wa(1,l-1,1)*ch1(1,ki,2)
       cc1(1,ki,lc) = wa(1,l-1,2)*ch1(1,ki,ip)
       cc1(2,ki,l) = ch1(2,ki,1)+wa(1,l-1,1)*ch1(2,ki,2)
       cc1(2,ki,lc) = wa(1,l-1,2)*ch1(2,ki,ip)
     end do

     do j = 3, ipph
       jc = ipp2 - j
       idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
       war = wa(1,idlj,1)
       wai = wa(1,idlj,2)
       do ki = 1, lid
         cc1(1,ki,l) = cc1(1,ki,l)+war*ch1(1,ki,j)
         cc1(1,ki,lc) = cc1(1,ki,lc)+wai*ch1(1,ki,jc)
         cc1(2,ki,l) = cc1(2,ki,l)+war*ch1(2,ki,j)
         cc1(2,ki,lc) = cc1(2,ki,lc)+wai*ch1(2,ki,jc)
       end do
     end do

  end do

  if ( 1 < ido .or. na == 1 ) then

    do ki = 1, lid
      ch1(1,ki,1) = cc1(1,ki,1)
      ch1(2,ki,1) = cc1(2,ki,1)
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        ch1(1,ki,j) = cc1(1,ki,j)-cc1(2,ki,jc)
        ch1(1,ki,jc) = cc1(1,ki,j)+cc1(2,ki,jc)
        ch1(2,ki,jc) = cc1(2,ki,j)-cc1(1,ki,jc)
        ch1(2,ki,j) = cc1(2,ki,j)+cc1(1,ki,jc)
      end do
    end do

    if ( ido == 1 ) then
      return
    end if

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
          cc(1,k,j,i) = wa(i,j-1,1)*ch(1,k,i,j) &
                       -wa(i,j-1,2)*ch(2,k,i,j)
          cc(2,k,j,i) = wa(i,j-1,1)*ch(2,k,i,j) &
                       +wa(i,j-1,2)*ch(1,k,i,j)
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        chold1 = cc1(1,ki,j)-cc1(2,ki,jc)
        chold2 = cc1(1,ki,j)+cc1(2,ki,jc)
        cc1(1,ki,j) = chold1
        cc1(2,ki,jc) = cc1(2,ki,j)-cc1(1,ki,jc)
        cc1(2,ki,j) = cc1(2,ki,j)+cc1(1,ki,jc)
        cc1(1,ki,jc) = chold2
      end do
    end do

  end if

  return
end
subroutine c1fgkf ( ido, ip, l1, lid, na, cc, cc1, in1, ch, ch1, in2, wa )

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
subroutine c1fm1b ( n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! C1FM1B is an FFTPACK5 auxiliary routine.
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
      call c1f2kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 2 ) then
      call c1f2kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 3 ) then
      call c1f3kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 4 ) then
      call c1f3kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 5 ) then
      call c1f4kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 6 ) then
      call c1f4kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 7 ) then
      call c1f5kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 8 ) then
      call c1f5kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 9 ) then
      call c1fgkb ( ido, ip, l1, lid, na, c, c, inc2, ch, ch, 2, wa(iw) )
    else if ( nbr == 10 ) then
      call c1fgkb ( ido, ip, l1, lid, na, ch, ch, 2, c, c, inc2, wa(iw) )
    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ( ido + ido )

    if ( ip <= 5 ) then
      na = 1 - na
    end if

  end do

  return
end
subroutine c1fm1f ( n, inc, c, ch, wa, fnf, fac )

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
subroutine cfft1b ( n, inc, c, lenc, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! CFFT1B: complex single precision backward fast Fourier transform, 1D.
!
!  Discussion:
!
!    CFFT1B computes the one-dimensional Fourier transform of a single 
!    periodic sequence within a complex array.  This transform is referred 
!    to as the backward transform or Fourier synthesis, transforming the
!    sequence from spectral to physical space.
!
!    This transform is normalized since a call to CFFT1B followed
!    by a call to CFFT1F (or vice-versa) reproduces the original
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
!    Input/output, complex ( kind = 4 ) C(LENC) containing the sequence to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least INC*(N-1) + 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to CFFT1I before the first call to routine CFFT1F 
!    or CFFT1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to CFFT1F and CFFT1B with the same N.
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
!    1, input parameter LENC not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
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
    call xerfft ( 'CFFT1B', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'CFFT1B', 8 )
    return
  end if

  if ( lenwrk < 2 * n ) then
    ier = 3
    call xerfft ( 'CFFT1B', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call c1fm1b ( n, inc, c, work, wsave, wsave(iw1), wsave(iw1+1) )

  return
end
subroutine cfft1f ( n, inc, c, lenc, wsave, lensav, work, lenwrk, ier )

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
subroutine cfft1i ( n, wsave, lensav, ier )

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
subroutine cfft2b ( ldim, l, m, c, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! CFFT2B: complex single precision backward fast Fourier transform, 2D.
!
!  Discussion:
!
!    CFFT2B computes the two-dimensional discrete Fourier transform of a 
!    complex periodic array.  This transform is known as the backward 
!    transform or Fourier synthesis, transforming from spectral to 
!    physical space.  Routine CFFT2B is normalized, in that a call to 
!    CFFT2B followed by a call to CFFT2F (or vice-versa) reproduces the 
!    original array within roundoff error. 
!
!    On 10 May 2010, this code was modified by changing the value
!    of an index into the WSAVE array.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    10 May 2010
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
!    Input, integer ( kind = 4 ) LDIM, the first dimension of C. 
!
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension of the two-dimensional complex array C.  The value 
!    of L must be less than or equal to that of LDIM.  The transform is
!    most efficient when L is a product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed in
!    the second dimension of the two-dimensional complex array C.  The transform
!    is most efficient when M is a product of small primes. 
!
!    Input/output, complex ( kind = 4 ) C(LDIM,M), on intput, the array of 
!    two dimensions containing the (L,M) subarray to be transformed.  On 
!    output, the transformed data.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV). WSAVE's contents must be 
!    initialized with a call to CFFT2I before the first call to routine CFFT2F
!    or CFFT2B with transform lengths L and M.  WSAVE's contents may be
!    re-used for subsequent calls to CFFT2F and CFFT2B with the same 
!    transform lengths L and M. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array. 
!    LENSAV must be at least 2*(L+M) + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8. 
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*L*M. 
!
!    Output, integer ( kind = 4 ) IER, the error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    5, input parameter LDIM < L;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) ldim
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 4 ) c(ldim,m)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) l
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( ldim < l ) then
    ier = 5
    call xerfft ( 'CFFT2B', -2 )
    return
  end if

  if ( lensav < 2 * l + int ( log ( real ( l, kind = 4 ) ) ) + &
                2 * m + int ( log ( real ( m, kind = 4 ) ) ) + 8 ) then
    ier = 2
    call xerfft ( 'CFFT2B', 6 )
    return
  end if

  if ( lenwrk < 2 * l * m ) then
    ier = 3
    call xerfft ( 'CFFT2B', 8 )
    return
  end if
!
!  Transform the X lines of the C array.
!
!  The value of IW was modified on 10 May 2010.
!
  iw = 2 * l + int ( log ( real ( l, kind = 4 ) ) ) + 5

  call cfftmb ( l, 1, m, ldim, c, (l-1)+ldim*(m-1) +1, &
    wsave(iw), 2*m + int(log( real ( m, kind = 4 ))) + 4, work, 2*l*m, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'CFFT2B', -5 )
    return
  end if
!
!  Transform the Y lines of the C array.
!
  iw = 1
  call cfftmb ( m, ldim, l, 1, c, (m-1)*ldim + l, wsave(iw), &
    2*l + int(log( real ( l, kind = 4 ))) + 4, work, 2*m*l, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'CFFT2B', -5 )
    return
  end if

  return
end
subroutine cfft2f ( ldim, l, m, c, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! CFFT2F: complex single precision forward fast Fourier transform, 2D.
!
!  Discussion:
!
!    CFFT2F computes the two-dimensional discrete Fourier transform of 
!    a complex periodic array. This transform is known as the forward 
!    transform or Fourier analysis, transforming from physical to 
!    spectral space. Routine CFFT2F is normalized, in that a call to 
!    CFFT2F followed by a call to CFFT2B (or vice-versa) reproduces the 
!    original array within roundoff error. 
!
!    On 10 May 2010, this code was modified by changing the value
!    of an index into the WSAVE array.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    10 May 2010
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
!    Input, integer ( kind = 4 ) LDIM, the first dimension of the array C. 
!
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension of the two-dimensional complex array C.  The value 
!    of L must be less than or equal to that of LDIM.  The transform is most 
!    efficient when L is a product of small primes. 
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension of the two-dimensional complex array C.  The 
!    transform is most efficient when M is a product of small primes. 
!
!    Input/output, complex ( kind = 4 ) C(LDIM,M), on input, the array of two 
!    dimensions containing the (L,M) subarray to be transformed.  On output, the
!    transformed data.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV). WSAVE's contents must be 
!    initialized with a call to CFFT2I before the first call to routine CFFT2F 
!    or CFFT2B with transform lengths L and M.  WSAVE's contents may be re-used 
!    for subsequent calls to CFFT2F and CFFT2B having those same 
!    transform lengths. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*(L+M) + INT(LOG(REAL(L)))
!    + INT(LOG(REAL(M))) + 8. 
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*L*M.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit; 
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    5, input parameter LDIM < L;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) ldim
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 4 ) c(ldim,m)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) l
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( ldim < l ) then
    ier = 5
    call xerfft ( 'CFFT2F', -2 )
    return
  end if

  if ( lensav < 2 * l + int ( log ( real ( l, kind = 4 ) ) ) + &
                2 * m + int ( log ( real ( m, kind = 4 ) ) ) + 8 ) then
    ier = 2
    call xerfft ( 'CFFT2F', 6 )
    return
  end if

  if ( lenwrk < 2 * l * m ) then
    ier = 3
    call xerfft ( 'CFFT2F', 8 )
    return
  end if
!
!  Transform the X lines of the C array.
!
!  The value of IW was modified on 10 May 2010.
!
  iw = 2 * l + int ( log ( real ( l, kind = 4 ) ) ) + 5

  call cfftmf ( l, 1, m, ldim, c, (l-1) + ldim*(m-1) +1, wsave(iw), &
    2*m + int(log( real ( m, kind = 4 ))) + 4, work, 2*l*m, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'CFFT2F', -5 )
    return
  end if
!
!  Transform the Y lines of the C array.
!
  iw = 1

  call cfftmf ( m, ldim, l, 1, c, (m-1)*ldim + l, wsave(iw), &
    2*l + int(log( real ( l, kind = 4 ))) + 4, work, 2*m*l, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'CFFT2F', -5 )
    return
  end if

  return
end
subroutine cfft2i ( l, m, wsave, lensav, ier )

!*****************************************************************************80
!
!! CFFT2I: initialization for CFFT2B and CFFT2F.
!
!  Discussion:
!
!    CFFT2I initializes real array WSAVE for use in its companion 
!    routines CFFT2F and CFFT2B for computing two-dimensional fast 
!    Fourier transforms of complex data.  Prime factorizations of L and M,
!    together with tabulations of the trigonometric functions, are 
!    computed and stored in array WSAVE.
!
!    On 10 May 2010, this code was modified by changing the value
!    of an index into the WSAVE array.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    10 May 2010
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
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension.  The transform is most efficient when L is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension.  The transform is most efficient when M is a 
!    product of small primes. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*(L+M) + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), contains the prime factors of L 
!    and M, and also certain trigonometric values which will be used in 
!    routines CFFT2B or CFFT2F. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) l
  integer ( kind = 4 ) m
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * l + int ( log ( real ( l, kind = 4 ) ) ) + &
                2 * m + int ( log ( real ( m, kind = 4 ) ) ) + 8 ) then
    ier = 2
    call xerfft ( 'CFFT2I', 4 )
    return
  end if

  call cfftmi ( l, wsave(1), 2*l + int(log( real ( l, kind = 4 ))) + 4, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'CFFT2I', -5 )
    return
  end if
!
!  On 10 May 2010, the value of IW was modified.
!
  iw = 2 * l + int ( log ( real ( l, kind = 4 ) ) ) + 5

  call cfftmi ( m, wsave(iw), 2*m + int(log( real ( m, kind = 4 ))) + 4, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'CFFT2I', -5 )
    return
  end if

  return
end
subroutine cfftmb ( lot, jump, n, inc, c, lenc, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! CFFTMB: complex single precision backward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    CFFTMB computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a complex array.  This transform is referred
!    to as the backward transform or Fourier synthesis, transforming the
!    sequences from spectral to physical space.  This transform is 
!    normalized since a call to CFFTMF followed by a call to CFFTMB (or
!    vice-versa) reproduces the original array within roundoff error. 
!
!    The parameters INC, JUMP, N and LOT are consistent if equality 
!    I1*INC + J1*JUMP = I2*INC + J2*JUMP for I1,I2 < N and J1,J2 < LOT 
!    implies I1=I2 and J1=J2.  For multiple FFTs to execute correctly, 
!    input variables INC, JUMP, N and LOT must be consistent, otherwise 
!    at least one array element mistakenly is transformed more than once.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    24 March 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array C. 
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in 
!    array C, of the first elements of two consecutive sequences to be 
!    transformed. 
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array C, of two consecutive elements within the same sequence to be 
!    transformed. 
!
!    Input/output, complex ( kind = 4 ) C(LENC), an array containing LOT 
!    sequences, each having length N, to be transformed.  C can have any 
!    number of dimensions, but the total number of locations must be at least 
!    LENC.  On output, C contains the transformed sequences.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least (LOT-1)*JUMP + INC*(N-1) + 1. 
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to CFFTMI before the first call to routine CFFTMF 
!    or CFFTMB for a given transform length N. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit
!    1, input parameter LENC not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC, JUMP, N, LOT are not consistent. 
!
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 4 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  logical              xercon

  ier = 0

  if ( lenc < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'CFFTMB', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'CFFTMB', 8 )
    return
  end if

  if ( lenwrk < 2 * lot * n ) then
    ier = 3
    call xerfft ( 'CFFTMB', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'CFFTMB', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call cmfm1b ( lot, jump, n, inc, c, work, wsave, wsave(iw1), &
    wsave(iw1+1) )

  return
end
subroutine cfftmf ( lot, jump, n, inc, c, lenc, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! CFFTMF: complex single precision forward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    CFFTMF computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a complex array. This transform is referred 
!    to as the forward transform or Fourier analysis, transforming the 
!    sequences from physical to spectral space. This transform is 
!    normalized since a call to CFFTMF followed by a call to CFFTMB 
!    (or vice-versa) reproduces the original array within roundoff error. 
!
!    The parameters integers INC, JUMP, N and LOT are consistent if equality
!    I1*INC + J1*JUMP = I2*INC + J2*JUMP for I1,I2 < N and J1,J2 < LOT 
!    implies I1=I2 and J1=J2. For multiple FFTs to execute correctly, 
!    input variables INC, JUMP, N and LOT must be consistent, otherwise 
!    at least one array element mistakenly is transformed more than once.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    24 March 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be 
!    transformed within array C. 
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, 
!    in array C, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes. 
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array C, of two consecutive elements within the same sequence to be 
!    transformed.
!
!    Input/output, complex ( kind = 4 ) C(LENC), array containing LOT sequences,
!    each having length N, to be transformed.  C can have any number of 
!    dimensions, but the total number of locations must be at least LENC.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array. 
!    LENC must be at least (LOT-1)*JUMP + INC*(N-1) + 1. 
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to CFFTMI before the first call to routine CFFTMF 
!    or CFFTMB for a given transform length N. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4. 
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*LOT*N. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0 successful exit;
!    1 input parameter LENC not big enough;
!    2 input parameter LENSAV not big enough;
!    3 input parameter LENWRK not big enough;
!    4 input parameters INC, JUMP, N, LOT are not consistent.
!
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 4 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  logical              xercon

  ier = 0

  if ( lenc < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'CFFTMF', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'CFFTMF', 8 )
    return
  end if

  if ( lenwrk < 2 * lot * n ) then
    ier = 3
    call xerfft ( 'CFFTMF', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'CFFTMF', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call cmfm1f ( lot, jump, n, inc, c, work, wsave, wsave(iw1), &
    wsave(iw1+1) )

  return
end
subroutine cfftmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! CFFTMI: initialization for CFFTMB and CFFTMF.
!
!  Discussion:
!
!    CFFTMI initializes array WSAVE for use in its companion routines 
!    CFFTMB and CFFTMF.  CFFTMI must be called before the first call 
!    to CFFTMB or CFFTMF, and after whenever the value of integer N changes. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    24 March 2005
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4. 
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used in
!    routines CFFTMB or CFFTMF.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit; 
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cfftmi ', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1
  call r4_mcfti1 ( n, wsave, wsave(iw1), wsave(iw1+1) )

  return
end
subroutine cmf2kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF2KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,2)
  real ( kind = 4 ) ch(2,in2,l1,2,ido)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,1,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1) + cc(1,m1,k,1,2)
        ch(1,m2,k,2,1) = cc(1,m1,k,1,1) - cc(1,m1,k,1,2)
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1) + cc(2,m1,k,1,2)
        ch(2,m2,k,2,1) = cc(2,m1,k,1,1) - cc(2,m1,k,1,2)
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+cc(1,m1,k,i,2)
          tr2 = cc(1,m1,k,i,1)-cc(1,m1,k,i,2)
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+cc(2,m1,k,i,2)
          ti2 = cc(2,m1,k,i,1)-cc(2,m1,k,i,2)

          ch(2,m2,k,2,i) = wa(i,1,1) * ti2 + wa(i,1,2) * tr2
          ch(1,m2,k,2,i) = wa(i,1,1) * tr2 - wa(i,1,2) * ti2

        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1

        chold1         = cc(1,m1,k,1,1) + cc(1,m1,k,1,2)
        cc(1,m1,k,1,2) = cc(1,m1,k,1,1) - cc(1,m1,k,1,2)
        cc(1,m1,k,1,1) = chold1

        chold2         = cc(2,m1,k,1,1) + cc(2,m1,k,1,2)
        cc(2,m1,k,1,2) = cc(2,m1,k,1,1) - cc(2,m1,k,1,2)
        cc(2,m1,k,1,1) = chold2

      end do
    end do

  end if

  return
end
subroutine cmf2kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF2KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,2)
  real ( kind = 4 ) ch(2,in2,l1,2,ido)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,1,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+cc(1,m1,k,1,2)
        ch(1,m2,k,2,1) = cc(1,m1,k,1,1)-cc(1,m1,k,1,2)
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+cc(2,m1,k,1,2)
        ch(2,m2,k,2,1) = cc(2,m1,k,1,1)-cc(2,m1,k,1,2)
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+cc(1,m1,k,i,2)
          tr2 = cc(1,m1,k,i,1)-cc(1,m1,k,i,2)
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+cc(2,m1,k,i,2)
          ti2 = cc(2,m1,k,i,1)-cc(2,m1,k,i,2)
          ch(2,m2,k,2,i) = wa(i,1,1)*ti2-wa(i,1,2)*tr2
          ch(1,m2,k,2,i) = wa(i,1,1)*tr2+wa(i,1,2)*ti2
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 2 * l1, kind = 4 )

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(1,m2,k,1,1) = sn * ( cc(1,m1,k,1,1) + cc(1,m1,k,1,2) )
        ch(1,m2,k,2,1) = sn * ( cc(1,m1,k,1,1) - cc(1,m1,k,1,2) )
        ch(2,m2,k,1,1) = sn * ( cc(2,m1,k,1,1) + cc(2,m1,k,1,2) )
        ch(2,m2,k,2,1) = sn * ( cc(2,m1,k,1,1) - cc(2,m1,k,1,2) )
      end do
    end do

  else

    sn = 1.0E+00 / real ( 2 * l1, kind = 4 )

    do k = 1, l1
      do m1 = 1, m1d, im1

        chold1         = sn * ( cc(1,m1,k,1,1) + cc(1,m1,k,1,2) )
        cc(1,m1,k,1,2) = sn * ( cc(1,m1,k,1,1) - cc(1,m1,k,1,2) )
        cc(1,m1,k,1,1) = chold1

        chold2         = sn * ( cc(2,m1,k,1,1) + cc(2,m1,k,1,2) )
        cc(2,m1,k,1,2) = sn * ( cc(2,m1,k,1,1) - cc(2,m1,k,1,2) )
        cc(2,m1,k,1,1) = chold2

      end do
    end do

  end if

  return
end
subroutine cmf3kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF3KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,3)
  real ( kind = 4 ) ch(2,in2,l1,3,ido)
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) di2
  real ( kind = 4 ) di3
  real ( kind = 4 ) dr2
  real ( kind = 4 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ), parameter :: taui =  0.866025403784439E+00
  real ( kind = 4 ), parameter :: taur = -0.5E+00
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,2,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1

        m2 = m2 + im2

        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2

        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2

        cr3 = taui * (cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui * (cc(2,m1,k,1,2)-cc(2,m1,k,1,3))

        ch(1,m2,k,2,1) = cr2-ci3
        ch(1,m2,k,3,1) = cr2+ci3
        ch(2,m2,k,2,1) = ci2+cr3
        ch(2,m2,k,3,1) = ci2-cr3

      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,3)
          cr2 = cc(1,m1,k,i,1)+taur*tr2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,3)
          ci2 = cc(2,m1,k,i,1)+taur*ti2
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2
          cr3 = taui*(cc(1,m1,k,i,2)-cc(1,m1,k,i,3))
          ci3 = taui*(cc(2,m1,k,i,2)-cc(2,m1,k,i,3))
          dr2 = cr2-ci3
          dr3 = cr2+ci3
          di2 = ci2+cr3
          di3 = ci2-cr3
          ch(2,m2,k,2,i) = wa(i,1,1)*di2+wa(i,1,2)*dr2
          ch(1,m2,k,2,i) = wa(i,1,1)*dr2-wa(i,1,2)*di2
          ch(2,m2,k,3,i) = wa(i,2,1)*di3+wa(i,2,2)*dr3
          ch(1,m2,k,3,i) = wa(i,2,1)*dr3-wa(i,2,2)*di3
        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        cc(1,m1,k,1,1) = cc(1,m1,k,1,1)+tr2
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        cc(2,m1,k,1,1) = cc(2,m1,k,1,1)+ti2
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        cc(1,m1,k,1,2) = cr2-ci3
        cc(1,m1,k,1,3) = cr2+ci3
        cc(2,m1,k,1,2) = ci2+cr3
        cc(2,m1,k,1,3) = ci2-cr3
      end do
    end do

  end if

  return
end
subroutine cmf3kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF3KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,3)
  real ( kind = 4 ) ch(2,in2,l1,3,ido)
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) di2
  real ( kind = 4 ) di3
  real ( kind = 4 ) dr2
  real ( kind = 4 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ), parameter :: taui = -0.866025403784439E+00
  real ( kind = 4 ), parameter :: taur = -0.5E+00
  real ( kind = 4 ) ti2
  real ( kind = 4 ) tr2
  real ( kind = 4 ) wa(ido,2,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        ch(1,m2,k,2,1) = cr2-ci3
        ch(1,m2,k,3,1) = cr2+ci3
        ch(2,m2,k,2,1) = ci2+cr3
        ch(2,m2,k,3,1) = ci2-cr3
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,3)
          cr2 = cc(1,m1,k,i,1)+taur*tr2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,3)
          ci2 = cc(2,m1,k,i,1)+taur*ti2
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2
          cr3 = taui*(cc(1,m1,k,i,2)-cc(1,m1,k,i,3))
          ci3 = taui*(cc(2,m1,k,i,2)-cc(2,m1,k,i,3))
          dr2 = cr2-ci3
          dr3 = cr2+ci3
          di2 = ci2+cr3
          di3 = ci2-cr3
          ch(2,m2,k,2,i) = wa(i,1,1)*di2-wa(i,1,2)*dr2
          ch(1,m2,k,2,i) = wa(i,1,1)*dr2+wa(i,1,2)*di2
          ch(2,m2,k,3,i) = wa(i,2,1)*di3-wa(i,2,2)*dr3
          ch(1,m2,k,3,i) = wa(i,2,1)*dr3+wa(i,2,2)*di3
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 3 * l1, kind = 4 )

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        ch(1,m2,k,1,1) = sn*(cc(1,m1,k,1,1)+tr2)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        ch(2,m2,k,1,1) = sn*(cc(2,m1,k,1,1)+ti2)
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        ch(1,m2,k,2,1) = sn*(cr2-ci3)
        ch(1,m2,k,3,1) = sn*(cr2+ci3)
        ch(2,m2,k,2,1) = sn*(ci2+cr3)
        ch(2,m2,k,3,1) = sn*(ci2-cr3)
      end do
    end do

  else

    sn = 1.0E+00 / real ( 3 * l1, kind = 4 )

    do k = 1, l1
      do m1 = 1, m1d, im1
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        cc(1,m1,k,1,1) = sn*(cc(1,m1,k,1,1)+tr2)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        cc(2,m1,k,1,1) = sn*(cc(2,m1,k,1,1)+ti2)
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        cc(1,m1,k,1,2) = sn*(cr2-ci3)
        cc(1,m1,k,1,3) = sn*(cr2+ci3)
        cc(2,m1,k,1,2) = sn*(ci2+cr3)
        cc(2,m1,k,1,3) = sn*(ci2-cr3)
      end do
    end do

  end if

  return
end
subroutine cmf4kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF4KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,4)
  real ( kind = 4 ) ch(2,in2,l1,4,ido)
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) ci4
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) ti1
  real ( kind = 4 ) ti2
  real ( kind = 4 ) ti3
  real ( kind = 4 ) ti4
  real ( kind = 4 ) tr1
  real ( kind = 4 ) tr2
  real ( kind = 4 ) tr3
  real ( kind = 4 ) tr4
  real ( kind = 4 ) wa(ido,3,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,4)-cc(2,m1,k,1,2)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,2)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = tr2+tr3
        ch(1,m2,k,3,1) = tr2-tr3
        ch(2,m2,k,1,1) = ti2+ti3
        ch(2,m2,k,3,1) = ti2-ti3
        ch(1,m2,k,2,1) = tr1+tr4
        ch(1,m2,k,4,1) = tr1-tr4
        ch(2,m2,k,2,1) = ti1+ti4
        ch(2,m2,k,4,1) = ti1-ti4
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti1 = cc(2,m1,k,i,1)-cc(2,m1,k,i,3)
          ti2 = cc(2,m1,k,i,1)+cc(2,m1,k,i,3)
          ti3 = cc(2,m1,k,i,2)+cc(2,m1,k,i,4)
          tr4 = cc(2,m1,k,i,4)-cc(2,m1,k,i,2)
          tr1 = cc(1,m1,k,i,1)-cc(1,m1,k,i,3)
          tr2 = cc(1,m1,k,i,1)+cc(1,m1,k,i,3)
          ti4 = cc(1,m1,k,i,2)-cc(1,m1,k,i,4)
          tr3 = cc(1,m1,k,i,2)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = tr2+tr3
          cr3 = tr2-tr3
          ch(2,m2,k,1,i) = ti2+ti3
          ci3 = ti2-ti3
          cr2 = tr1+tr4
          cr4 = tr1-tr4
          ci2 = ti1+ti4
          ci4 = ti1-ti4
          ch(1,m2,k,2,i) = wa(i,1,1)*cr2-wa(i,1,2)*ci2
          ch(2,m2,k,2,i) = wa(i,1,1)*ci2+wa(i,1,2)*cr2
          ch(1,m2,k,3,i) = wa(i,2,1)*cr3-wa(i,2,2)*ci3
          ch(2,m2,k,3,i) = wa(i,2,1)*ci3+wa(i,2,2)*cr3
          ch(1,m2,k,4,i) = wa(i,3,1)*cr4-wa(i,3,2)*ci4
          ch(2,m2,k,4,i) = wa(i,3,1)*ci4+wa(i,3,2)*cr4
        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,4)-cc(2,m1,k,1,2)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,2)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        cc(1,m1,k,1,1) = tr2+tr3
        cc(1,m1,k,1,3) = tr2-tr3
        cc(2,m1,k,1,1) = ti2+ti3
        cc(2,m1,k,1,3) = ti2-ti3
        cc(1,m1,k,1,2) = tr1+tr4
        cc(1,m1,k,1,4) = tr1-tr4
        cc(2,m1,k,1,2) = ti1+ti4
        cc(2,m1,k,1,4) = ti1-ti4
      end do
    end do

  end if

  return
end
subroutine cmf4kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF4KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,4)
  real ( kind = 4 ) ch(2,in2,l1,4,ido)
  real ( kind = 4 ) ci2
  real ( kind = 4 ) ci3
  real ( kind = 4 ) ci4
  real ( kind = 4 ) cr2
  real ( kind = 4 ) cr3
  real ( kind = 4 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
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

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,2)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,4)-cc(1,m1,k,1,2)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = tr2+tr3
        ch(1,m2,k,3,1) = tr2-tr3
        ch(2,m2,k,1,1) = ti2+ti3
        ch(2,m2,k,3,1) = ti2-ti3
        ch(1,m2,k,2,1) = tr1+tr4
        ch(1,m2,k,4,1) = tr1-tr4
        ch(2,m2,k,2,1) = ti1+ti4
        ch(2,m2,k,4,1) = ti1-ti4
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti1 = cc(2,m1,k,i,1)-cc(2,m1,k,i,3)
          ti2 = cc(2,m1,k,i,1)+cc(2,m1,k,i,3)
          ti3 = cc(2,m1,k,i,2)+cc(2,m1,k,i,4)
          tr4 = cc(2,m1,k,i,2)-cc(2,m1,k,i,4)
          tr1 = cc(1,m1,k,i,1)-cc(1,m1,k,i,3)
          tr2 = cc(1,m1,k,i,1)+cc(1,m1,k,i,3)
          ti4 = cc(1,m1,k,i,4)-cc(1,m1,k,i,2)
          tr3 = cc(1,m1,k,i,2)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = tr2+tr3
          cr3 = tr2-tr3
          ch(2,m2,k,1,i) = ti2+ti3
          ci3 = ti2-ti3
          cr2 = tr1+tr4
          cr4 = tr1-tr4
          ci2 = ti1+ti4
          ci4 = ti1-ti4
          ch(1,m2,k,2,i) = wa(i,1,1)*cr2+wa(i,1,2)*ci2
          ch(2,m2,k,2,i) = wa(i,1,1)*ci2-wa(i,1,2)*cr2
          ch(1,m2,k,3,i) = wa(i,2,1)*cr3+wa(i,2,2)*ci3
          ch(2,m2,k,3,i) = wa(i,2,1)*ci3-wa(i,2,2)*cr3
          ch(1,m2,k,4,i) = wa(i,3,1)*cr4+wa(i,3,2)*ci4
          ch(2,m2,k,4,i) = wa(i,3,1)*ci4-wa(i,3,2)*cr4
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 4 * l1, kind = 4 )

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,2)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,4)-cc(1,m1,k,1,2)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = sn*(tr2+tr3)
        ch(1,m2,k,3,1) = sn*(tr2-tr3)
        ch(2,m2,k,1,1) = sn*(ti2+ti3)
        ch(2,m2,k,3,1) = sn*(ti2-ti3)
        ch(1,m2,k,2,1) = sn*(tr1+tr4)
        ch(1,m2,k,4,1) = sn*(tr1-tr4)
        ch(2,m2,k,2,1) = sn*(ti1+ti4)
        ch(2,m2,k,4,1) = sn*(ti1-ti4)
      end do
    end do

  else

    sn = 1.0E+00 / real ( 4 * l1, kind = 4 )

    do k = 1, l1
      do m1 = 1, m1d, im1
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,2)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,4)-cc(1,m1,k,1,2)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        cc(1,m1,k,1,1) = sn*(tr2+tr3)
        cc(1,m1,k,1,3) = sn*(tr2-tr3)
        cc(2,m1,k,1,1) = sn*(ti2+ti3)
        cc(2,m1,k,1,3) = sn*(ti2-ti3)
        cc(1,m1,k,1,2) = sn*(tr1+tr4)
        cc(1,m1,k,1,4) = sn*(tr1-tr4)
        cc(2,m1,k,1,2) = sn*(ti1+ti4)
        cc(2,m1,k,1,4) = sn*(ti1-ti4)
      end do
    end do

  end if

  return
end
subroutine cmf5kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF5KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,5)
  real ( kind = 4 ) ch(2,in2,l1,5,ido)
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
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) ti2
  real ( kind = 4 ) ti3
  real ( kind = 4 ) ti4
  real ( kind = 4 ) ti5
  real ( kind = 4 ), parameter :: ti11 =  0.9510565162951536E+00
  real ( kind = 4 ), parameter :: ti12 =  0.5877852522924731E+00
  real ( kind = 4 ) tr2
  real ( kind = 4 ) tr3
  real ( kind = 4 ) tr4
  real ( kind = 4 ) tr5
  real ( kind = 4 ), parameter :: tr11 =  0.3090169943749474E+00
  real ( kind = 4 ), parameter :: tr12 = -0.8090169943749474E+00
  real ( kind = 4 ) wa(ido,4,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2+tr3
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2+ti3
        cr2 = cc(1,m1,k,1,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,m1,k,1,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,m1,k,1,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,m1,k,1,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        ch(1,m2,k,2,1) = cr2-ci5
        ch(1,m2,k,5,1) = cr2+ci5
        ch(2,m2,k,2,1) = ci2+cr5
        ch(2,m2,k,3,1) = ci3+cr4
        ch(1,m2,k,3,1) = cr3-ci4
        ch(1,m2,k,4,1) = cr3+ci4
        ch(2,m2,k,4,1) = ci3-cr4
        ch(2,m2,k,5,1) = ci2-cr5
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti5 = cc(2,m1,k,i,2)-cc(2,m1,k,i,5)
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,5)
          ti4 = cc(2,m1,k,i,3)-cc(2,m1,k,i,4)
          ti3 = cc(2,m1,k,i,3)+cc(2,m1,k,i,4)
          tr5 = cc(1,m1,k,i,2)-cc(1,m1,k,i,5)
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,5)
          tr4 = cc(1,m1,k,i,3)-cc(1,m1,k,i,4)
          tr3 = cc(1,m1,k,i,3)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2+tr3
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2+ti3
          cr2 = cc(1,m1,k,i,1)+tr11*tr2+tr12*tr3
          ci2 = cc(2,m1,k,i,1)+tr11*ti2+tr12*ti3
          cr3 = cc(1,m1,k,i,1)+tr12*tr2+tr11*tr3
          ci3 = cc(2,m1,k,i,1)+tr12*ti2+tr11*ti3
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
          ch(1,m2,k,2,i) = wa(i,1,1) * dr2 - wa(i,1,2) * di2
          ch(2,m2,k,2,i) = wa(i,1,1) * di2 + wa(i,1,2) * dr2
          ch(1,m2,k,3,i) = wa(i,2,1) * dr3 - wa(i,2,2) * di3
          ch(2,m2,k,3,i) = wa(i,2,1) * di3 + wa(i,2,2) * dr3
          ch(1,m2,k,4,i) = wa(i,3,1) * dr4 - wa(i,3,2) * di4
          ch(2,m2,k,4,i) = wa(i,3,1) * di4 + wa(i,3,2) * dr4
          ch(1,m2,k,5,i) = wa(i,4,1) * dr5 - wa(i,4,2) * di5
          ch(2,m2,k,5,i) = wa(i,4,1) * di5 + wa(i,4,2) * dr5
        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)

        chold1 = cc(1,m1,k,1,1) + tr2 + tr3
        chold2 = cc(2,m1,k,1,1) + ti2 + ti3

        cr2 = cc(1,m1,k,1,1) + tr11 * tr2 + tr12 * tr3
        ci2 = cc(2,m1,k,1,1) + tr11 * ti2 + tr12 * ti3
        cr3 = cc(1,m1,k,1,1) + tr12 * tr2 + tr11 * tr3
        ci3 = cc(2,m1,k,1,1) + tr12 * ti2 + tr11 * ti3

        cc(1,m1,k,1,1) = chold1
        cc(2,m1,k,1,1) = chold2

        cr5 = ti11*tr5 + ti12*tr4
        ci5 = ti11*ti5 + ti12*ti4
        cr4 = ti12*tr5 - ti11*tr4
        ci4 = ti12*ti5 - ti11*ti4
        cc(1,m1,k,1,2) = cr2-ci5
        cc(1,m1,k,1,5) = cr2+ci5
        cc(2,m1,k,1,2) = ci2+cr5
        cc(2,m1,k,1,3) = ci3+cr4
        cc(1,m1,k,1,3) = cr3-ci4
        cc(1,m1,k,1,4) = cr3+ci4
        cc(2,m1,k,1,4) = ci3-cr4
        cc(2,m1,k,1,5) = ci2-cr5
      end do
    end do

  end if

  return
end
subroutine cmf5kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! CMF5KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(2,in1,l1,ido,5)
  real ( kind = 4 ) ch(2,in2,l1,5,ido)
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
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
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

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2+tr3
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2+ti3
        cr2 = cc(1,m1,k,1,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,m1,k,1,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,m1,k,1,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,m1,k,1,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        ch(1,m2,k,2,1) = cr2-ci5
        ch(1,m2,k,5,1) = cr2+ci5
        ch(2,m2,k,2,1) = ci2+cr5
        ch(2,m2,k,3,1) = ci3+cr4
        ch(1,m2,k,3,1) = cr3-ci4
        ch(1,m2,k,4,1) = cr3+ci4
        ch(2,m2,k,4,1) = ci3-cr4
        ch(2,m2,k,5,1) = ci2-cr5
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti5 = cc(2,m1,k,i,2)-cc(2,m1,k,i,5)
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,5)
          ti4 = cc(2,m1,k,i,3)-cc(2,m1,k,i,4)
          ti3 = cc(2,m1,k,i,3)+cc(2,m1,k,i,4)
          tr5 = cc(1,m1,k,i,2)-cc(1,m1,k,i,5)
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,5)
          tr4 = cc(1,m1,k,i,3)-cc(1,m1,k,i,4)
          tr3 = cc(1,m1,k,i,3)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2+tr3
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2+ti3
          cr2 = cc(1,m1,k,i,1)+tr11*tr2+tr12*tr3
          ci2 = cc(2,m1,k,i,1)+tr11*ti2+tr12*ti3
          cr3 = cc(1,m1,k,i,1)+tr12*tr2+tr11*tr3
          ci3 = cc(2,m1,k,i,1)+tr12*ti2+tr11*ti3
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
          ch(1,m2,k,2,i) = wa(i,1,1)*dr2+wa(i,1,2)*di2
          ch(2,m2,k,2,i) = wa(i,1,1)*di2-wa(i,1,2)*dr2
          ch(1,m2,k,3,i) = wa(i,2,1)*dr3+wa(i,2,2)*di3
          ch(2,m2,k,3,i) = wa(i,2,1)*di3-wa(i,2,2)*dr3
          ch(1,m2,k,4,i) = wa(i,3,1)*dr4+wa(i,3,2)*di4
          ch(2,m2,k,4,i) = wa(i,3,1)*di4-wa(i,3,2)*dr4
          ch(1,m2,k,5,i) = wa(i,4,1)*dr5+wa(i,4,2)*di5
          ch(2,m2,k,5,i) = wa(i,4,1)*di5-wa(i,4,2)*dr5
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( 5 * l1, kind = 4 )
 
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = sn*(cc(1,m1,k,1,1)+tr2+tr3)
        ch(2,m2,k,1,1) = sn*(cc(2,m1,k,1,1)+ti2+ti3)
        cr2 = cc(1,m1,k,1,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,m1,k,1,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,m1,k,1,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,m1,k,1,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        ch(1,m2,k,2,1) = sn*(cr2-ci5)
        ch(1,m2,k,5,1) = sn*(cr2+ci5)
        ch(2,m2,k,2,1) = sn*(ci2+cr5)
        ch(2,m2,k,3,1) = sn*(ci3+cr4)
        ch(1,m2,k,3,1) = sn*(cr3-ci4)
        ch(1,m2,k,4,1) = sn*(cr3+ci4)
        ch(2,m2,k,4,1) = sn*(ci3-cr4)
        ch(2,m2,k,5,1) = sn*(ci2-cr5)
      end do
    end do

  else

    sn = 1.0E+00 / real ( 5 * l1, kind = 4 )

    do k = 1, l1
      do m1 = 1, m1d, im1

        ti5 = cc(2,m1,k,1,2) - cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2) + cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3) - cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3) + cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2) - cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2) + cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3) - cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3) + cc(1,m1,k,1,4)

        chold1 = sn * ( cc(1,m1,k,1,1) + tr2 + tr3 )
        chold2 = sn * ( cc(2,m1,k,1,1) + ti2 + ti3 )

        cr2 = cc(1,m1,k,1,1) + tr11 * tr2 + tr12 * tr3
        ci2 = cc(2,m1,k,1,1) + tr11 * ti2 + tr12 * ti3
        cr3 = cc(1,m1,k,1,1) + tr12 * tr2 + tr11 * tr3
        ci3 = cc(2,m1,k,1,1) + tr12 * ti2 + tr11 * ti3

        cc(1,m1,k,1,1) = chold1
        cc(2,m1,k,1,1) = chold2

        cr5 = ti11 * tr5 + ti12 * tr4
        ci5 = ti11 * ti5 + ti12 * ti4
        cr4 = ti12 * tr5 - ti11 * tr4
        ci4 = ti12 * ti5 - ti11 * ti4

        cc(1,m1,k,1,2) = sn * ( cr2 - ci5 )
        cc(1,m1,k,1,5) = sn * ( cr2 + ci5 )
        cc(2,m1,k,1,2) = sn * ( ci2 + cr5 )
        cc(2,m1,k,1,3) = sn * ( ci3 + cr4 )
        cc(1,m1,k,1,3) = sn * ( cr3 - ci4 )
        cc(1,m1,k,1,4) = sn * ( cr3 + ci4 )
        cc(2,m1,k,1,4) = sn * ( ci3 - cr4 )
        cc(2,m1,k,1,5) = sn * ( ci2 - cr5 )

      end do
    end do

  end if

  return
end
subroutine cmfgkb ( lot, ido, ip, l1, lid, na, cc, cc1, im1, in1, &
  ch, ch1, im2, in2, wa )

!*****************************************************************************80
!
!! CMFGKB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 4 ) cc(2,in1,l1,ip,ido)
  real ( kind = 4 ) cc1(2,in1,lid,ip)
  real ( kind = 4 ) ch(2,in2,l1,ido,ip)
  real ( kind = 4 ) ch1(2,in2,lid,ip)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idlj
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) j
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) ki
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) wa(ido,ip-1,2)
  real ( kind = 4 ) wai
  real ( kind = 4 ) war

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  do ki = 1, lid
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
      ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,j) =  cc1(1,m1,ki,j) + cc1(1,m1,ki,jc)
        ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) - cc1(1,m1,ki,jc)
        ch1(2,m2,ki,j) =  cc1(2,m1,ki,j) + cc1(2,m1,ki,jc)
        ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(2,m1,ki,jc)
      end do
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,1) = cc1(1,m1,ki,1) + ch1(1,m2,ki,j)
        cc1(2,m1,ki,1) = cc1(2,m1,ki,1) + ch1(2,m2,ki,j)
      end do
    end do
  end do

  do l = 2, ipph

    lc = ipp2 - l
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2

        cc1(1,m1,ki,l)  = ch1(1,m2,ki,1) + wa(1,l-1,1) * ch1(1,m2,ki,2)
        cc1(1,m1,ki,lc) =                  wa(1,l-1,2) * ch1(1,m2,ki,ip)
        cc1(2,m1,ki,l)  = ch1(2,m2,ki,1) + wa(1,l-1,1) * ch1(2,m2,ki,2)
        cc1(2,m1,ki,lc) =                  wa(1,l-1,2) * ch1(2,m2,ki,ip)

      end do
    end do

    do j = 3, ipph
      jc = ipp2 - j
      idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
      war = wa(1,idlj,1)
      wai = wa(1,idlj,2)
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc1(1,m1,ki,l)  = cc1(1,m1,ki,l)  + war * ch1(1,m2,ki,j)
          cc1(1,m1,ki,lc) = cc1(1,m1,ki,lc) + wai * ch1(1,m2,ki,jc)
          cc1(2,m1,ki,l)  = cc1(2,m1,ki,l)  + war * ch1(2,m2,ki,j)
          cc1(2,m1,ki,lc) = cc1(2,m1,ki,lc) + wai * ch1(2,m2,ki,jc)
        end do
      end do
    end do

  end do

  if( 1 < ido .or. na == 1 ) then

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
        ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch1(1,m2,ki,j)  = cc1(1,m1,ki,j) - cc1(2,m1,ki,jc)
          ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) + cc1(2,m1,ki,jc)
          ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(1,m1,ki,jc)
          ch1(2,m2,ki,j)  = cc1(2,m1,ki,j) + cc1(1,m1,ki,jc)
        end do
      end do
    end do

    if ( ido == 1 ) then
      return
    end if

    do i = 1, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,1,i) = ch(1,m2,k,i,1)
          cc(2,m1,k,1,i) = ch(2,m2,k,i,1)
        end do
      end do
    end do

    do j = 2, ip
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,j,1) = ch(1,m2,k,1,j)
          cc(2,m1,k,j,1) = ch(2,m2,k,1,j)
        end do
      end do
    end do

    do j = 2, ip
      do i = 2, ido
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            cc(1,m1,k,j,i) = wa(i,j-1,1) * ch(1,m2,k,i,j) &
                           - wa(i,j-1,2) * ch(2,m2,k,i,j)
            cc(2,m1,k,j,i) = wa(i,j-1,1) * ch(2,m2,k,i,j) &
                           + wa(i,j-1,2) * ch(1,m2,k,i,j)
          end do
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        do m1 = 1, m1d, im1

          chold1         = cc1(1,m1,ki,j) - cc1(2,m1,ki,jc)
          chold2         = cc1(1,m1,ki,j) + cc1(2,m1,ki,jc)
          cc1(1,m1,ki,j) = chold1

          cc1(2,m1,ki,jc) = cc1(2,m1,ki,j) - cc1(1,m1,ki,jc)
          cc1(2,m1,ki,j)  = cc1(2,m1,ki,j) + cc1(1,m1,ki,jc)
          cc1(1,m1,ki,jc) = chold2

        end do
      end do
    end do

  end if

  return
end
subroutine cmfgkf ( lot, ido, ip, l1, lid, na, cc, cc1, im1, in1, &
  ch, ch1, im2, in2, wa )

!*****************************************************************************80
!
!! CMFGKF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 4 ) cc(2,in1,l1,ip,ido)
  real ( kind = 4 ) cc1(2,in1,lid,ip)
  real ( kind = 4 ) ch(2,in2,l1,ido,ip)
  real ( kind = 4 ) ch1(2,in2,lid,ip)
  real ( kind = 4 ) chold1
  real ( kind = 4 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idlj
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) j 
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) ki
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 4 ) sn
  real ( kind = 4 ) wa(ido,ip-1,2)
  real ( kind = 4 ) wai
  real ( kind = 4 ) war

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  do ki = 1, lid
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
      ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,j) =  cc1(1,m1,ki,j) + cc1(1,m1,ki,jc)
        ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) - cc1(1,m1,ki,jc)
        ch1(2,m2,ki,j) =  cc1(2,m1,ki,j) + cc1(2,m1,ki,jc)
        ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(2,m1,ki,jc)
      end do
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,1) = cc1(1,m1,ki,1) + ch1(1,m2,ki,j)
        cc1(2,m1,ki,1) = cc1(2,m1,ki,1) + ch1(2,m2,ki,j)
      end do
    end do
  end do

  do l = 2, ipph

    lc = ipp2 - l

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,l)  = ch1(1,m2,ki,1) + wa(1,l-1,1) * ch1(1,m2,ki,2)
        cc1(1,m1,ki,lc) =                - wa(1,l-1,2) * ch1(1,m2,ki,ip)
        cc1(2,m1,ki,l)  = ch1(2,m2,ki,1) + wa(1,l-1,1) * ch1(2,m2,ki,2)
        cc1(2,m1,ki,lc) =                - wa(1,l-1,2) * ch1(2,m2,ki,ip)
      end do
    end do

    do j = 3, ipph
      jc = ipp2 - j
      idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
      war = wa(1,idlj,1)
      wai = -wa(1,idlj,2)
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc1(1,m1,ki,l)  = cc1(1,m1,ki,l)  + war * ch1(1,m2,ki,j)
          cc1(1,m1,ki,lc) = cc1(1,m1,ki,lc) + wai * ch1(1,m2,ki,jc)
          cc1(2,m1,ki,l)  = cc1(2,m1,ki,l)  + war * ch1(2,m2,ki,j)
          cc1(2,m1,ki,lc) = cc1(2,m1,ki,lc) + wai * ch1(2,m2,ki,jc)
        end do
      end do
    end do

  end do

  if ( 1 < ido ) then

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
        ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch1(1,m2,ki,j)  = cc1(1,m1,ki,j) - cc1(2,m1,ki,jc)
          ch1(2,m2,ki,j)  = cc1(2,m1,ki,j) + cc1(1,m1,ki,jc)
          ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) + cc1(2,m1,ki,jc)
          ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(1,m1,ki,jc)
        end do
      end do
    end do

    do i = 1, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,1,i) = ch(1,m2,k,i,1)
          cc(2,m1,k,1,i) = ch(2,m2,k,i,1)
        end do
      end do
    end do

    do j = 2, ip
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,j,1) = ch(1,m2,k,1,j)
          cc(2,m1,k,j,1) = ch(2,m2,k,1,j)
        end do
      end do
    end do

    do j = 2, ip
      do i = 2, ido
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            cc(1,m1,k,j,i) = wa(i,j-1,1) * ch(1,m2,k,i,j) &
                           + wa(i,j-1,2) * ch(2,m2,k,i,j)
            cc(2,m1,k,j,i) = wa(i,j-1,1) * ch(2,m2,k,i,j) &
                           - wa(i,j-1,2) * ch(1,m2,k,i,j)
          end do
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0E+00 / real ( ip * l1, kind = 4 )

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,1) = sn * cc1(1,m1,ki,1)
        ch1(2,m2,ki,1) = sn * cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch1(1,m2,ki,j)  = sn * ( cc1(1,m1,ki,j) - cc1(2,m1,ki,jc) )
          ch1(2,m2,ki,j)  = sn * ( cc1(2,m1,ki,j) + cc1(1,m1,ki,jc) )
          ch1(1,m2,ki,jc) = sn * ( cc1(1,m1,ki,j) + cc1(2,m1,ki,jc) )
          ch1(2,m2,ki,jc) = sn * ( cc1(2,m1,ki,j) - cc1(1,m1,ki,jc) )
        end do
      end do
    end do

  else

    sn = 1.0E+00 / real ( ip * l1, kind = 4 )

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,1) = sn * cc1(1,m1,ki,1)
        cc1(2,m1,ki,1) = sn * cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        do m1 = 1, m1d, im1
          chold1 = sn * ( cc1(1,m1,ki,j) - cc1(2,m1,ki,jc) )
          chold2 = sn * ( cc1(1,m1,ki,j) + cc1(2,m1,ki,jc) )
          cc1(1,m1,ki,j) = chold1
          cc1(2,m1,ki,jc) = sn * ( cc1(2,m1,ki,j) - cc1(1,m1,ki,jc) )
          cc1(2,m1,ki,j)  = sn * ( cc1(2,m1,ki,j) + cc1(1,m1,ki,jc) )
          cc1(1,m1,ki,jc) = chold2
        end do
      end do
    end do

  end if

  return
end
subroutine cmfm1b ( lot, jump, n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! CMFM1B is an FFTPACK5 auxiliary routine.
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
  implicit none

  complex ( kind = 4 ) c(*)
  real ( kind = 4 ) ch(*)
  real ( kind = 4 ) fac(*)
  real ( kind = 4 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) lid
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nbr
  integer ( kind = 4 ) nf
  real ( kind = 4 ) wa(*)

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
      call cmf2kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 2 ) then
      call cmf2kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 3 ) then
      call cmf3kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 4 ) then
      call cmf3kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 5 ) then
      call cmf4kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 6 ) then
      call cmf4kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 7 ) then
      call cmf5kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 8 ) then
      call cmf5kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 9 ) then
      call cmfgkb ( lot, ido, ip, l1, lid, na, c, c, jump, inc, ch, ch, &
        1, lot, wa(iw) )
    else if ( nbr == 10 ) then
      call cmfgkb ( lot, ido, ip, l1, lid, na, ch, ch, 1, lot, c, c, &
        jump, inc, wa(iw) )
    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ( ido + ido )

    if ( ip <= 5 ) then
      na = 1 - na
    end if

  end do

  return
end
subroutine cmfm1f ( lot, jump, n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! CMFM1F is an FFTPACK5 auxiliary routine.
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
  implicit none

  complex ( kind = 4 ) c(*)
  real ( kind = 4 ) ch(*)
  real ( kind = 4 ) fac(*)
  real ( kind = 4 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) lid
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nbr
  integer ( kind = 4 ) nf
  real ( kind = 4 ) wa(*)

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
      call cmf2kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 2 ) then
      call cmf2kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 3 ) then
      call cmf3kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 4 ) then
      call cmf3kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 5 ) then
      call cmf4kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 6 ) then
      call cmf4kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 7 ) then
      call cmf5kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 8 ) then
      call cmf5kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 9 ) then
      call cmfgkf ( lot, ido, ip, l1, lid, na, c, c, jump, inc, ch, ch, &
        1, lot, wa(iw) )
    else if ( nbr == 10 ) then
      call cmfgkf ( lot, ido, ip, l1, lid, na, ch, ch, 1, lot, c, c, &
        jump, inc, wa(iw) )
    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ( ido + ido )

    if ( ip <= 5 ) then
      na = 1 - na
    end if

  end do

  return
end
subroutine cosq1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! COSQ1B: real single precision backward cosine quarter wave transform, 1D.
!
!  Discussion:
!
!    COSQ1B computes the one-dimensional Fourier transform of a sequence 
!    which is a cosine series with odd wave numbers.  This transform is 
!    referred to as the backward transform or Fourier synthesis, transforming
!    the sequence from spectral to physical space.
!
!    This transform is normalized since a call to COSQ1B followed
!    by a call to COSQ1F (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    31 March 2005
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
!    Input, integer ( kind = 4 ) N, the number of elements to be transformed 
!    in the sequence.  The transform is most efficient when N is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR); on input, containing the sequence 
!    to be transformed, and on output, containing the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQ1I before the first call to routine COSQ1F 
!    or COSQ1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to COSQ1F and COSQ1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 4 ) ssqrt2
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) x1

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'COSQ1B', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'COSQ1B', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'COSQ1B', 10 ) 
    return
  end if

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt2 = 1.0E+00 / sqrt ( 2.0E+00 )
    x1 = x(1,1) + x(1,2)
    x(1,2) = ssqrt2 * ( x(1,1) - x(1,2) )
    x(1,1) = x1
    return
  end if

  call cosqb1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'COSQ1B', -5 )
    return
  end if

  return
end
subroutine cosq1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! COSQ1F: real single precision forward cosine quarter wave transform, 1D.
!
!  Discussion:
!
!    COSQ1F computes the one-dimensional Fourier transform of a sequence 
!    which is a cosine series with odd wave numbers.  This transform is 
!    referred to as the forward transform or Fourier analysis, transforming 
!    the sequence from physical to spectral space.
!
!    This transform is normalized since a call to COSQ1F followed
!    by a call to COSQ1B (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    31 March 2005
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
!    Input, integer ( kind = 4 ) N, the number of elements to be transformed 
!    in the sequence.  The transform is most efficient when N is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR); on input, containing the sequence 
!    to be transformed, and on output, containing the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQ1I before the first call to routine COSQ1F 
!    or COSQ1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to COSQ1F and COSQ1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) n
  integer ( kind = 4 ) lenx
  real ( kind = 4 ) ssqrt2
  real ( kind = 4 ) tsqx
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'cosq1f', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cosq1f', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'cosq1f', 10 )
    return
  end if

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt2 = 1.0E+00 / sqrt ( 2.0E+00 )
    tsqx = ssqrt2 * x(1,2)
    x(1,2) = 0.5E+00 * x(1,1) - tsqx
    x(1,1) = 0.5E+00 * x(1,1) + tsqx
    return
  end if

  call cosqf1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosq1f', -5 )
    return
  end if

  return
end
subroutine cosq1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! COSQ1I: initialization for COSQ1B and COSQ1F.
!
!  Discussion:
!
!    COSQ1I initializes array WSAVE for use in its companion routines 
!    COSQ1F and COSQ1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    31 March 2005
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
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors of N
!    and also containing certain trigonometric values which will be used 
!    in routines COSQ1B or COSQ1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 4 ) dt
  real ( kind = 4 ) fk
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  real ( kind = 4 ) pih
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cosq1i', 3 )
    return
  end if

  pih = 2.0E+00 * atan ( 1.0E+00 )
  dt = pih / real ( n, kind = 4 )
  fk = 0.0E+00

  do k = 1, n
    fk = fk + 1.0E+00
    wsave(k) = cos ( fk * dt )
  end do

  lnsv = n + int ( log ( real ( n, kind = 4 ) ) ) + 4

  call rfft1i ( n, wsave(n+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosq1i', -5 )
    return
  end if

  return
end
subroutine cosqb1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! COSQB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np2
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xim1

  ier = 0
  ns2 = ( n + 1 ) / 2
  np2 = n + 2

  do i = 3, n, 2
    xim1 = x(1,i-1) + x(1,i)
    x(1,i) = 0.5E+00 * ( x(1,i-1) - x(1,i) )
    x(1,i-1) = 0.5E+00 * xim1
  end do

  x(1,1) = 0.5E+00 * x(1,1)
  modn = mod ( n, 2 )

  if ( modn == 0 ) then
    x(1,n) = 0.5E+00 * x(1,n)
  end if

  lenx = inc * ( n - 1 )  + 1
  lnsv = n + int ( log ( real ( n, kind = 4 ) ) ) + 4
  lnwk = n

  call rfft1b ( n, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosqb1', -5 )
    return
  end if

  do k = 2, ns2
    kc = np2 - k
    work(k)  = wsave(k-1) * x(1,kc) + wsave(kc-1) * x(1,k)
    work(kc) = wsave(k-1) * x(1,k)  - wsave(kc-1) * x(1,kc)
  end do

  if ( modn == 0 ) then
    x(1,ns2+1) = wsave(ns2) * ( x(1,ns2+1) + x(1,ns2+1) )
  end if

  do k = 2, ns2
    kc = np2 - k
    x(1,k)  = work(k) + work(kc)
    x(1,kc) = work(k) - work(kc)
  end do

  x(1,1) = x(1,1) + x(1,1)

  return
end
subroutine cosqf1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! COSQF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np2
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xim1

  ier = 0
  ns2 = ( n + 1 ) / 2
  np2 = n + 2

  do k = 2, ns2
    kc = np2 - k
    work(k)  = x(1,k) + x(1,kc)
    work(kc) = x(1,k) - x(1,kc)
  end do

  modn = mod ( n, 2 )

  if ( modn == 0 ) then
    work(ns2+1) = x(1,ns2+1) + x(1,ns2+1)
  end if

  do k = 2, ns2
    kc = np2 - k
    x(1,k)  = wsave(k-1) * work(kc) + wsave(kc-1) * work(k)
    x(1,kc) = wsave(k-1) * work(k)  - wsave(kc-1) * work(kc)
  end do

  if ( modn == 0 ) then
    x(1,ns2+1) = wsave(ns2) * work(ns2+1)
  end if

  lenx = inc * ( n - 1 ) + 1
  lnsv = n + int ( log ( real ( n, kind = 4 ) ) ) + 4
  lnwk = n

  call rfft1f ( n, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosqf1', -5 )
    return
  end if

  do i = 3, n, 2
    xim1   = 0.5E+00 * ( x(1,i-1) + x(1,i) )
    x(1,i) = 0.5E+00 * ( x(1,i-1) - x(1,i) )
    x(1,i-1) = xim1
  end do

  return
end
subroutine cosqmb ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! COSQMB: real single precision backward cosine quarter wave, multiple vectors.
!
!  Discussion:
!
!    COSQMB computes the one-dimensional Fourier transform of multiple
!    sequences, each of which is a cosine series with odd wave numbers.
!    This transform is referred to as the backward transform or Fourier
!    synthesis, transforming the sequences from spectral to physical space.
!
!    This transform is normalized since a call to COSQMB followed
!    by a call to COSQMF (or vice-versa) reproduces the original
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
!    01 April 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, 
!    in array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), array containing LOT sequences, 
!    each having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQMI before the first call to routine COSQMF 
!    or COSQMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to COSQMF and COSQMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  real ( kind = 4 ) ssqrt2
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) x1
  logical              xercon

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'cosqmb', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cosqmb', 8 )
    return
  end if

  if ( lenwrk < lot * n ) then
    ier = 3
    call xerfft ( 'cosqmb', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'cosqmb', -1 )
    return
  end if

  lj = ( lot - 1 ) * jump + 1

  if ( n < 2 ) then
    do m = 1, lj, jump
      x(m,1) = x(m,1)
    end do
    return
  end if

  if ( n  ==  2 ) then
    ssqrt2 = 1.0E+00 / sqrt ( 2.0E+00 )
    do m = 1, lj, jump
      x1 = x(m,1) + x(m,2)
      x(m,2) = ssqrt2 * ( x(m,1) - x(m,2) )
      x(m,1) = x1
    end do
    return
  end if

  call mcsqb1 ( lot, jump, n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosqmb', -5 )
    return
  end if

  return
end
subroutine cosqmf ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! COSQMF: real single precision forward cosine quarter wave, multiple vectors.
!
!  Discussion:
!
!    COSQMF computes the one-dimensional Fourier transform of multiple 
!    sequences within a real array, where each of the sequences is a 
!    cosine series with odd wave numbers.  This transform is referred to 
!    as the forward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to COSQMF followed
!    by a call to COSQMB (or vice-versa) reproduces the original
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
!    01 April 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), array containing LOT sequences, 
!    each having length N.  R can have any number of dimensions, but the total
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQMI before the first call to routine COSQMF 
!    or COSQMB for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to COSQMF and COSQMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  real ( kind = 4 ) ssqrt2
  real ( kind = 4 ) tsqx
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'cosqmf', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cosqmf', 8 )
    return
  end if

  if ( lenwrk < lot * n ) then
    ier = 3
    call xerfft ( 'cosqmf', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'cosqmf', -1 )
    return
  end if

  lj = ( lot - 1 ) * jump + 1

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then

    ssqrt2 = 1.0E+00 / sqrt ( 2.0E+00 )

    do m = 1, lj, jump
      tsqx = ssqrt2 * x(m,2)
      x(m,2) = 0.5E+00 * x(m,1) - tsqx
      x(m,1) = 0.5E+00 * x(m,1) + tsqx
    end do

    return
  end if

  call mcsqf1 ( lot, jump, n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosqmf', -5 )
    return
  end if

  return
end
subroutine cosqmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! COSQMI: initialization for COSQMB and COSQMF.
!
!  Discussion:
!
!    COSQMI initializes array WSAVE for use in its companion routines 
!    COSQMF and COSQMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    01 April 2005
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors of 
!    N and also containing certain trigonometric values which will be used 
!    in routines COSQMB or COSQMF.
!
!    Input, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 4 ) dt
  real ( kind = 4 ) fk
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  real ( kind = 4 ) pih
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cosqmi', 3 )
    return
  end if

  pih = 2.0E+00 * atan ( 1.0E+00 )

  dt = pih / real ( n, kind = 4 )

  fk = 0.0E+00
  do k = 1, n
    fk = fk + 1.0E+00
    wsave(k) = cos ( fk * dt )
  end do

  lnsv = n + int ( log ( real ( n, kind = 4 ) ) ) + 4

  call rfftmi ( n, wsave(n+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cosqmi', -5 )
    return
  end if

  return
end
subroutine cost1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! COST1B: real single precision backward cosine transform, 1D.
!
!  Discussion:
!
!    COST1B computes the one-dimensional Fourier transform of an even 
!    sequence within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the sequence 
!    from spectral to physical space.
!
!    This transform is normalized since a call to COST1B followed
!    by a call to COST1F (or vice-versa) reproduces the original array 
!    within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    28 March 2005
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
!    transformed.  The transform is most efficient when N-1 is a product of
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), containing the sequence to 
!     be transformed.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COST1I before the first call to routine COST1F 
!    or COST1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to COST1F and COST1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N-1.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'cost1b', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cost1b', 8 )
    return
  end if

  if ( lenwrk < n - 1 ) then
    ier = 3
    call xerfft ( 'cost1b', 10 )
    return
  end if

  call costb1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'cost1b', -5 )
    return
  end if

  return
end
subroutine cost1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! COST1F: real single precision forward cosine transform, 1D.
!
!  Discussion:
!
!    COST1F computes the one-dimensional Fourier transform of an even 
!    sequence within a real array.  This transform is referred to as the 
!    forward transform or Fourier analysis, transforming the sequence 
!    from  physical to spectral space.
!
!    This transform is normalized since a call to COST1F followed by a call 
!    to COST1B (or vice-versa) reproduces the original array within 
!    roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    28 March 2005
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
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), containing the sequence to 
!    be transformed.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COST1I before the first call to routine COST1F
!    or COST1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to COST1F and COST1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N-1.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'COST1F', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'COST1F', 8 )
    return
  end if

  if ( lenwrk < n - 1 ) then
    ier = 3
    call xerfft ( 'COST1F', 10 )
    return
  end if

  call costf1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'COST1F', -5 )
    return
  end if

  return
end
subroutine cost1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! COST1I: initialization for COST1B and COST1F.
!
!  Discussion:
!
!    COST1I initializes array WSAVE for use in its companion routines 
!    COST1F and COST1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    28 March 2005
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
!    transformed.  The transform is most efficient when N-1 is a product 
!    of small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, dimension of WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors of
!    N and also containing certain trigonometric values which will be used in
!    routines COST1B or COST1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 4 ) dt
  real ( kind = 4 ) fk
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) pi
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'COST1I', 3 )
    return
  end if

  if ( n <= 3 ) then
    return
  end if

  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2
  pi = 4.0E+00 * atan ( 1.0E+00 )
  dt = pi / real ( nm1, kind = 4 )
  fk = 0.0E+00
  do k = 2, ns2
    kc = np1 - k
    fk = fk + 1.0E+00
    wsave(k) = 2.0E+00 * sin ( fk * dt )
    wsave(kc) = 2.0E+00 * cos ( fk * dt )
  end do 

  lnsv = nm1 + int ( log ( real ( nm1, kind = 4 ) ) ) + 4

  call rfft1i ( nm1, wsave(n+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'COST1I', -5 )
    return
  end if

  return
end
subroutine costb1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! COSTB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  real ( kind = 4 ) fnm1s2
  real ( kind = 4 ) fnm1s4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) x1h
  real ( kind = 4 ) x1p3
  real ( kind = 4 ) x2
  real ( kind = 4 ) xi

  ier = 0
  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    x1h    = x(1,1) + x(1,2)
    x(1,2) = x(1,1) - x(1,2)
    x(1,1) = x1h
    return
  end if

  if ( n == 3 ) then
    x1p3 = x(1,1) + x(1,3)
    x2 = x(1,2)
    x(1,2) = x(1,1) - x(1,3)
    x(1,1) = x1p3 + x2
    x(1,3) = x1p3 - x2
    return
  end if

  x(1,1) = x(1,1) + x(1,1)
  x(1,n) = x(1,n) + x(1,n)
  dsum = x(1,1) - x(1,n)
  x(1,1) = x(1,1) + x(1,n)

  do k = 2, ns2
    kc = np1 - k
    t1 = x(1,k) + x(1,kc)
    t2 = x(1,k) - x(1,kc)
    dsum = dsum + wsave(kc) * t2
    t2 = wsave(k) * t2
    x(1,k) = t1 - t2
    x(1,kc) = t1 + t2
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    x(1,ns2+1) = x(1,ns2+1) + x(1,ns2+1)
  end if

  lenx = inc * ( nm1 - 1 )  + 1
  lnsv = nm1 + int ( log ( real ( nm1, kind = 4 ) ) ) + 4
  lnwk = nm1

  call rfft1f ( nm1, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'COSTB1', -5 )
    return
  end if

  fnm1s2 = real ( nm1, kind = 4 ) / 2.0E+00
  dsum = 0.5E+00 * dsum
  x(1,1) = fnm1s2 * x(1,1)

  if ( mod ( nm1, 2 ) == 0 ) then
    x(1,nm1) = x(1,nm1) + x(1,nm1)
  end if

  fnm1s4 = real ( nm1, kind = 4 ) / 4.0E+00

  do i = 3, n, 2
    xi = fnm1s4 * x(1,i)
    x(1,i) = fnm1s4 * x(1,i-1)
    x(1,i-1) = dsum
    dsum = dsum + xi
  end do

  if ( modn == 0 ) then
    x(1,n) = dsum
  end if

  return
end
subroutine costf1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! COSTF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) snm1
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) tx2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) x1h
  real ( kind = 4 ) x1p3
  real ( kind = 4 ) xi

  ier = 0
  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    x1h = x(1,1) + x(1,2)
    x(1,2) = 0.5E+00 * ( x(1,1) - x(1,2) )
    x(1,1) = 0.5E+00 * x1h
    return
  end if

  if ( n == 3 ) then
    x1p3 = x(1,1) + x(1,3)
    tx2 = x(1,2) + x(1,2)
    x(1,2) = 0.5E+00 * ( x(1,1) - x(1,3) )
    x(1,1) = 0.25E+00 * ( x1p3 + tx2 )
    x(1,3) = 0.25E+00 * ( x1p3 - tx2 )
    return
  end if

  dsum = x(1,1) - x(1,n)
  x(1,1) = x(1,1) + x(1,n)
  do k = 2, ns2
    kc = np1 - k
    t1 = x(1,k) + x(1,kc)
    t2 = x(1,k) - x(1,kc)
    dsum = dsum + wsave(kc) * t2
    t2 = wsave(k) * t2
    x(1,k) = t1 - t2
    x(1,kc) = t1 + t2
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    x(1,ns2+1) = x(1,ns2+1) + x(1,ns2+1)
  end if

  lenx = inc * ( nm1 - 1 )  + 1
  lnsv = nm1 + int ( log ( real ( nm1, kind = 4 ) ) ) + 4
  lnwk = nm1

  call rfft1f ( nm1, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'costf1', -5 )
    return
  end if

  snm1 = 1.0E+00 / real ( nm1, kind = 4 )
  dsum = snm1 * dsum

  if ( mod ( nm1, 2 ) == 0 ) then
    x(1,nm1) = x(1,nm1) + x(1,nm1)
  end if

  do i = 3, n, 2
    xi = 0.5E+00 * x(1,i)
    x(1,i) = 0.5E+00 * x(1,i-1)
    x(1,i-1) = dsum
    dsum = dsum + xi
  end do

  if ( modn == 0 ) then
    x(1,n) = dsum
  end if

  x(1,1) = 0.5E+00 * x(1,1)
  x(1,n) = 0.5E+00 * x(1,n)

  return
end
subroutine costmb ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! COSTMB: real single precision backward cosine transform, multiple vectors.
!
!  Discussion:
!
!    COSTMB computes the one-dimensional Fourier transform of multiple 
!    even sequences within a real array.  This transform is referred to 
!    as the backward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to COSTMB followed
!    by a call to COSTMF (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    29 March 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), array containing LOT sequences, 
!    each having length N.  On input, the data to be transformed; on output,
!    the transormed data.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSTMI before the first call to routine COSTMF 
!    or COSTMB for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to COSTMF and COSTMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(N+1).
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'costmb', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'costmb', 8 )
    return
  end if

  if ( lenwrk < lot * ( n + 1 ) ) then
    ier = 3
    call xerfft ( 'costmb', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'costmb', -1 )
    return
  end if

  iw1 = lot + lot + 1

  call mcstb1 ( lot, jump, n, inc, x, wsave, work, work(iw1), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'costmb', -5 )
  end if

  return
end
subroutine costmf ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! COSTMF: real single precision forward cosine transform, multiple vectors.
!
!  Discussion:
!
!    COSTMF computes the one-dimensional Fourier transform of multiple 
!    even sequences within a real array.  This transform is referred to 
!    as the forward transform or Fourier analysis, transforming the 
!    sequences from physical to spectral space.
!
!    This transform is normalized since a call to COSTMF followed
!    by a call to COSTMB (or vice-versa) reproduces the original
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
!    29 March 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, 
!    in array R, of the first elements of two consecutive sequences to 
!    be transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), array containing LOT sequences, 
!    each having length N.  On input, the data to be transformed; on output,
!    the transormed data.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the  R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSTMI before the first call to routine COSTMF
!    or COSTMB for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to COSTMF and COSTMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(N+1).
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'COSTMF', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'COSTMF', 8 )
    return
  end if

  if ( lenwrk < lot * ( n + 1 ) ) then
    ier = 3
    call xerfft ( 'COSTMF', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'COSTMF', -1 )
    return
  end if

  iw1 = lot + lot + 1

  call mcstf1 ( lot, jump, n, inc, x, wsave, work, work(iw1), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'COSTMF', -5 )
    return
  end if

  return
end
subroutine costmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! COSTMI: initialization for COSTMB and COSTMF.
!
!  Discussion:
!
!    COSTMI initializes array WSAVE for use in its companion routines 
!    COSTMF and COSTMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    29 March 2005
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors of N 
!    and also containing certain trigonometric values which will be used 
!    in routines COSTMB or COSTMF.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 4 ) dt
  real ( kind = 4 ) fk
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) pi
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'costmi', 3 )
    return
  end if

  if ( n <= 3 ) then
    return
  end if

  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2
  pi = 4.0E+00 * atan ( 1.0E+00 )
  dt = pi / real ( nm1, kind = 4 )

  fk = 0.0E+00
  do k = 2, ns2
    kc = np1 - k
    fk = fk + 1.0E+00
    wsave(k) = 2.0E+00 * sin ( fk * dt )
    wsave(kc) = 2.0E+00 * cos ( fk * dt )
  end do

  lnsv = nm1 + int ( log ( real ( nm1, kind = 4 ) ) ) + 4

  call rfftmi ( nm1, wsave(n+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'costmi', -5 )
    return
  end if

  return
end
subroutine d1f2kb ( ido, l1, cc, in1, ch, in2, wa1 )

!*****************************************************************************80
!
!! D1F2KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,ido,2,l1)
  real ( kind = 8 ) ch(in2,ido,l1,2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) wa1(ido)

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k) + cc(1,ido,2,k)
    ch(1,1,k,2) = cc(1,1,1,k) - cc(1,ido,2,k)
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2
    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i

        ch(1,i-1,k,1) = cc(1,i-1,1,k) + cc(1,ic-1,2,k)
        ch(1,i,k,1)   = cc(1,i,1,k)   - cc(1,ic,2,k)

        ch(1,i-1,k,2) = wa1(i-2) * ( cc(1,i-1,1,k) - cc(1,ic-1,2,k) ) &
                      - wa1(i-1) * ( cc(1,i,1,k)   + cc(1,ic,2,k) )
        ch(1,i,k,2)   = wa1(i-2) * ( cc(1,i,1,k)   + cc(1,ic,2,k) ) &
                      + wa1(i-1) * ( cc(1,i-1,1,k) - cc(1,ic-1,2,k) )

      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,ido,k,1) =     cc(1,ido,1,k) + cc(1,ido,1,k)
    ch(1,ido,k,2) = - ( cc(1,1,2,k)   + cc(1,1,2,k) )
  end do

  return
end
subroutine d1f2kf ( ido, l1, cc, in1, ch, in2, wa1 )

!*****************************************************************************80
!
!! D1F2KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) ch(in2,ido,2,l1)
  real ( kind = 8 ) cc(in1,ido,l1,2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) wa1(ido)

  do k = 1, l1
    ch(1,1,1,k)   = cc(1,1,k,1) + cc(1,1,k,2)
    ch(1,ido,2,k) = cc(1,1,k,1) - cc(1,1,k,2)
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        ch(1,i,1,k) = cc(1,i,k,1)   + ( wa1(i-2) * cc(1,i,k,2) &
                                      - wa1(i-1) * cc(1,i-1,k,2) )
        ch(1,ic,2,k) = -cc(1,i,k,1) + ( wa1(i-2) * cc(1,i,k,2) &
                                      - wa1(i-1) * cc(1,i-1,k,2) ) 
        ch(1,i-1,1,k) = cc(1,i-1,k,1)  + ( wa1(i-2) * cc(1,i-1,k,2) &
                                         + wa1(i-1) * cc(1,i,k,2))
        ch(1,ic-1,2,k) = cc(1,i-1,k,1) - ( wa1(i-2) * cc(1,i-1,k,2) &
                                         + wa1(i-1) * cc(1,i,k,2))
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,1,2,k) = -cc(1,ido,k,2)
    ch(1,ido,1,k) = cc(1,ido,k,1)
  end do

  return
end
subroutine d1f3kb ( ido, l1, cc, in1, ch, in2, wa1, wa2 )

!*****************************************************************************80
!
!! D1F3KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) arg
  real ( kind = 8 ) cc(in1,ido,3,l1)
  real ( kind = 8 ) ch(in2,ido,l1,3)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) taui
  real ( kind = 8 ) taur
  real ( kind = 8 ) wa1(ido)
  real ( kind = 8 ) wa2(ido)

  arg = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 3.0D+00
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k) + 2.0D+00 * cc(1,ido,2,k)
    ch(1,1,k,2) = cc(1,1,1,k) + 2.0D+00 * taur * cc(1,ido,2,k) &
                              - 2.0D+00 * taui * cc(1,1,3,k)
    ch(1,1,k,3) = cc(1,1,1,k) + 2.0D+00 * taur * cc(1,ido,2,k) &
                              + 2.0D+00 * taui * cc(1,1,3,k)
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,k,1) = cc(1,i-1,1,k)+(cc(1,i-1,3,k)+cc(1,ic-1,2,k))
      ch(1,i,k,1) = cc(1,i,1,k)+(cc(1,i,3,k)-cc(1,ic,2,k))
      ch(1,i-1,k,2) = wa1(i-2)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))- &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k)))) -wa1(i-1)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))+ &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))))
      ch(1,i,k,2) = wa1(i-2)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))+ &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) +wa1(i-1)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))- &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k))))
      ch(1,i-1,k,3) = wa2(i-2)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))+ &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k)))) -wa2(i-1)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))- &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))))
      ch(1,i,k,3) = wa2(i-2)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))- &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) +wa2(i-1)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))+ &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k))))
    end do
  end do

  return
end
subroutine d1f3kf ( ido, l1, cc, in1, ch, in2, wa1, wa2 )

!*****************************************************************************80
!
!! D1F3KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) arg
  real ( kind = 8 ) cc(in1,ido,l1,3)
  real ( kind = 8 ) ch(in2,ido,3,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) taui
  real ( kind = 8 ) taur
  real ( kind = 8 ) wa1(ido)
  real ( kind = 8 ) wa2(ido)

  arg = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 3.0D+00
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    ch(1,1,1,k) = cc(1,1,k,1)          + ( cc(1,1,k,2) + cc(1,1,k,3) )
    ch(1,1,3,k) =                 taui * ( cc(1,1,k,3) - cc(1,1,k,2) )
    ch(1,ido,2,k) = cc(1,1,k,1) + taur * ( cc(1,1,k,2) + cc(1,1,k,3) )
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,1,k) = cc(1,i-1,k,1)+((wa1(i-2)*cc(1,i-1,k,2)+ &
        wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3)))
      ch(1,i,1,k) = cc(1,i,k,1)+((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3)))
      ch(1,i-1,3,k) = (cc(1,i-1,k,1)+taur*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))))+(taui*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))))
      ch(1,ic-1,2,k) = (cc(1,i-1,k,1)+taur*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))))-(taui*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))))
      ch(1,i,3,k) = (cc(1,i,k,1)+taur*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))))+(taui*((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2))))
      ch(1,ic,2,k) = (taui*((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2))))-(cc(1,i,k,1)+taur*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))))
    end do
  end do

  return
end
subroutine d1f4kb ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3 )

!*****************************************************************************80
!
!! D1F4KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,ido,4,l1)
  real ( kind = 8 ) ch(in2,ido,l1,4)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) sqrt2
  real ( kind = 8 ) wa1(ido)
  real ( kind = 8 ) wa2(ido)
  real ( kind = 8 ) wa3(ido)

  sqrt2 = sqrt ( 2.0D+00 )

  do k = 1, l1
    ch(1,1,k,3) = ( cc(1,1,1,k)   + cc(1,ido,4,k) ) &
                - ( cc(1,ido,2,k) + cc(1,ido,2,k) )
    ch(1,1,k,1) = ( cc(1,1,1,k)   + cc(1,ido,4,k) ) &
                + ( cc(1,ido,2,k) + cc(1,ido,2,k) )
    ch(1,1,k,4) = ( cc(1,1,1,k)   - cc(1,ido,4,k) ) &
                + ( cc(1,1,3,k)   + cc(1,1,3,k) )
    ch(1,1,k,2) = ( cc(1,1,1,k)   - cc(1,ido,4,k) ) &
                - ( cc(1,1,3,k)   + cc(1,1,3,k) )
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        ch(1,i-1,k,1) = (cc(1,i-1,1,k)+cc(1,ic-1,4,k)) &
          +(cc(1,i-1,3,k)+cc(1,ic-1,2,k))
        ch(1,i,k,1) = (cc(1,i,1,k)-cc(1,ic,4,k)) &
          +(cc(1,i,3,k)-cc(1,ic,2,k))
        ch(1,i-1,k,2) = wa1(i-2)*((cc(1,i-1,1,k)-cc(1,ic-1,4,k)) &
          -(cc(1,i,3,k)+cc(1,ic,2,k)))-wa1(i-1) &
          *((cc(1,i,1,k)+cc(1,ic,4,k))+(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))
        ch(1,i,k,2) = wa1(i-2)*((cc(1,i,1,k)+cc(1,ic,4,k)) &
          +(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))+wa1(i-1) &
          *((cc(1,i-1,1,k)-cc(1,ic-1,4,k))-(cc(1,i,3,k)+cc(1,ic,2,k)))
        ch(1,i-1,k,3) = wa2(i-2)*((cc(1,i-1,1,k)+cc(1,ic-1,4,k)) &
          -(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))-wa2(i-1) &
          *((cc(1,i,1,k)-cc(1,ic,4,k))-(cc(1,i,3,k)-cc(1,ic,2,k)))
        ch(1,i,k,3) = wa2(i-2)*((cc(1,i,1,k)-cc(1,ic,4,k)) &
          -(cc(1,i,3,k)-cc(1,ic,2,k)))+wa2(i-1) &
          *((cc(1,i-1,1,k)+cc(1,ic-1,4,k))-(cc(1,i-1,3,k) &
          +cc(1,ic-1,2,k)))
        ch(1,i-1,k,4) = wa3(i-2)*((cc(1,i-1,1,k)-cc(1,ic-1,4,k)) &
          +(cc(1,i,3,k)+cc(1,ic,2,k)))-wa3(i-1) &
          *((cc(1,i,1,k)+cc(1,ic,4,k))-(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))
        ch(1,i,k,4) = wa3(i-2)*((cc(1,i,1,k)+cc(1,ic,4,k)) &
          -(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))+wa3(i-1) &
          *((cc(1,i-1,1,k)-cc(1,ic-1,4,k))+(cc(1,i,3,k)+cc(1,ic,2,k)))
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,ido,k,1) = ( cc(1,ido,1,k) + cc(1,ido,3,k) ) &
                  + ( cc(1,ido,1,k) + cc(1,ido,3,k))
    ch(1,ido,k,2) = sqrt2 * ( ( cc(1,ido,1,k) - cc(1,ido,3,k) ) &
                            - ( cc(1,1,2,k)   + cc(1,1,4,k) ) )
    ch(1,ido,k,3) = ( cc(1,1,4,k) - cc(1,1,2,k) ) &
                  + ( cc(1,1,4,k) - cc(1,1,2,k) )
    ch(1,ido,k,4) = -sqrt2 * ( ( cc(1,ido,1,k) - cc(1,ido,3,k) ) &
                             + ( cc(1,1,2,k) + cc(1,1,4,k) ) )
  end do

  return
end
subroutine d1f4kf ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3 )

!*****************************************************************************80
!
!! D1F4KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,ido,l1,4)
  real ( kind = 8 ) ch(in2,ido,4,l1)
  real ( kind = 8 ) hsqt2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) wa1(ido)
  real ( kind = 8 ) wa2(ido)
  real ( kind = 8 ) wa3(ido)

  hsqt2 = sqrt ( 2.0D+00 ) / 2.0D+00 

  do k = 1, l1
    ch(1,1,1,k)   = ( cc(1,1,k,2) + cc(1,1,k,4) ) &
                  + ( cc(1,1,k,1) + cc(1,1,k,3) )
    ch(1,ido,4,k) = ( cc(1,1,k,1) + cc(1,1,k,3) ) &
                  - ( cc(1,1,k,2) + cc(1,1,k,4) )
    ch(1,ido,2,k) = cc(1,1,k,1) - cc(1,1,k,3)
    ch(1,1,3,k)   = cc(1,1,k,4) - cc(1,1,k,2)
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        ch(1,i-1,1,k) = ((wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
          cc(1,i,k,2))+(wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
          cc(1,i,k,4)))+(cc(1,i-1,k,1)+(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))
        ch(1,ic-1,4,k) = (cc(1,i-1,k,1)+(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))-((wa1(i-2)*cc(1,i-1,k,2)+ &
          wa1(i-1)*cc(1,i,k,2))+(wa3(i-2)*cc(1,i-1,k,4)+ &
          wa3(i-1)*cc(1,i,k,4)))
        ch(1,i,1,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))+(cc(1,i,k,1)+(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
        ch(1,ic,4,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))-(cc(1,i,k,1)+(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
        ch(1,i-1,3,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))-(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))+(cc(1,i-1,k,1)-(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))
        ch(1,ic-1,2,k) = (cc(1,i-1,k,1)-(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))-((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))-(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))
        ch(1,i,3,k) = ((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
          cc(1,i,k,4))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
          cc(1,i,k,2)))+(cc(1,i,k,1)-(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
        ch(1,ic,2,k) = ((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
          cc(1,i,k,4))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
          cc(1,i,k,2)))-(cc(1,i,k,1)-(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
       end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,ido,1,k) = (hsqt2*(cc(1,ido,k,2)-cc(1,ido,k,4)))+ cc(1,ido,k,1)
    ch(1,ido,3,k) = cc(1,ido,k,1)-(hsqt2*(cc(1,ido,k,2)- cc(1,ido,k,4)))
    ch(1,1,2,k) = (-hsqt2*(cc(1,ido,k,2)+cc(1,ido,k,4)))- cc(1,ido,k,3)
    ch(1,1,4,k) = (-hsqt2*(cc(1,ido,k,2)+cc(1,ido,k,4)))+ cc(1,ido,k,3)
  end do

  return
end
subroutine d1f5kb ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3, wa4 )

!*****************************************************************************80
!
!! D1F5KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) arg
  real ( kind = 8 ) cc(in1,ido,5,l1)
  real ( kind = 8 ) ch(in2,ido,l1,5)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) ti11
  real ( kind = 8 ) ti12
  real ( kind = 8 ) tr11
  real ( kind = 8 ) tr12
  real ( kind = 8 ) wa1(ido)
  real ( kind = 8 ) wa2(ido)
  real ( kind = 8 ) wa3(ido)
  real ( kind = 8 ) wa4(ido)

  arg = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 5.0D+00
  tr11 = cos ( arg )
  ti11 = sin ( arg )
  tr12 = cos ( 2.0D+00 * arg )
  ti12 = sin ( 2.0D+00 * arg )

  do k = 1, l1

    ch(1,1,k,1) = cc(1,1,1,k) + 2.0D+00 * cc(1,ido,2,k) &
                              + 2.0D+00 * cc(1,ido,4,k)

    ch(1,1,k,2) = ( cc(1,1,1,k) &
      +   tr11 * 2.0D+00 * cc(1,ido,2,k) + tr12 * 2.0D+00 * cc(1,ido,4,k) ) &
      - ( ti11 * 2.0D+00 * cc(1,1,3,k)   + ti12 * 2.0D+00 * cc(1,1,5,k))

    ch(1,1,k,3) = ( cc(1,1,1,k) &
      +   tr12 * 2.0D+00 * cc(1,ido,2,k) + tr11 * 2.0D+00 * cc(1,ido,4,k) ) &
      - ( ti12 * 2.0D+00 * cc(1,1,3,k)   - ti11 * 2.0D+00 * cc(1,1,5,k))

    ch(1,1,k,4) = ( cc(1,1,1,k) &
      +   tr12 * 2.0D+00 * cc(1,ido,2,k) + tr11 * 2.0D+00 * cc(1,ido,4,k) ) &
      + ( ti12 * 2.0D+00 * cc(1,1,3,k)   - ti11 * 2.0D+00 * cc(1,1,5,k))

    ch(1,1,k,5) = ( cc(1,1,1,k) &
      +   tr11 * 2.0D+00 * cc(1,ido,2,k) + tr12 * 2.0D+00 * cc(1,ido,4,k) ) &
      + ( ti11 * 2.0D+00 * cc(1,1,3,k)   + ti12 * 2.0D+00 * cc(1,1,5,k) )

  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,k,1) = cc(1,i-1,1,k)+(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +(cc(1,i-1,5,k)+cc(1,ic-1,4,k))
      ch(1,i,k,1) = cc(1,i,1,k)+(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +(cc(1,i,5,k)-cc(1,ic,4,k))
      ch(1,i-1,k,2) = wa1(i-2)*((cc(1,i-1,1,k)+tr11* &
        (cc(1,i-1,3,k)+cc(1,ic-1,2,k))+tr12 &
        *(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa1(i-1)*((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))+(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,2) = wa1(i-2)*((cc(1,i,1,k)+tr11*(cc(1,i,3,k) &
        -cc(1,ic,2,k))+tr12*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti11*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))+ti12 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))+wa1(i-1) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k) &
        +cc(1,ic-1,2,k))+tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k))) &
        -(ti11*(cc(1,i,3,k)+cc(1,ic,2,k))+ti12 &
        *(cc(1,i,5,k)+cc(1,ic,4,k))))
      ch(1,i-1,k,3) = wa2(i-2) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa2(i-1) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
      cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,3) = wa2(i-2) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa2(i-1) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k))))
      ch(1,i-1,k,4) = wa3(i-2) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa3(i-1) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
      cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        -(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,4) = wa3(i-2) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        -(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa3(i-1) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k))))
      ch(1,i-1,k,5) = wa4(i-2) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa4(i-1) &
        *((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))-(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,5) = wa4(i-2) &
        *((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))-(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa4(i-1) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k))))
    end do
  end do

  return
end
subroutine d1f5kf ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3, wa4 )

!*****************************************************************************80
!
!! D1F5KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) arg
  real ( kind = 8 ) cc(in1,ido,l1,5)
  real ( kind = 8 ) ch(in2,ido,5,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 8 ) ti11
  real ( kind = 8 ) ti12
  real ( kind = 8 ) tr11
  real ( kind = 8 ) tr12
  real ( kind = 8 ) wa1(ido)
  real ( kind = 8 ) wa2(ido)
  real ( kind = 8 ) wa3(ido)
  real ( kind = 8 ) wa4(ido)

  arg = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 5.0D+00
  tr11 = cos ( arg )
  ti11 = sin ( arg )
  tr12 = cos ( 2.0D+00 * arg )
  ti12 = sin ( 2.0D+00 * arg )

  do k = 1, l1

    ch(1,1,1,k) = cc(1,1,k,1) + ( cc(1,1,k,5) + cc(1,1,k,2) ) &
                              + ( cc(1,1,k,4) + cc(1,1,k,3) )

    ch(1,ido,2,k) = cc(1,1,k,1) + tr11 * ( cc(1,1,k,5) + cc(1,1,k,2) ) &
                                + tr12 * ( cc(1,1,k,4) + cc(1,1,k,3) )

    ch(1,1,3,k) =                 ti11 * ( cc(1,1,k,5) - cc(1,1,k,2) ) &
                                + ti12 * ( cc(1,1,k,4) - cc(1,1,k,3) )

    ch(1,ido,4,k) = cc(1,1,k,1) + tr12 * ( cc(1,1,k,5) + cc(1,1,k,2) ) &
                                + tr11 * ( cc(1,1,k,4) + cc(1,1,k,3) )

    ch(1,1,5,k) =                 ti12 * ( cc(1,1,k,5) - cc(1,1,k,2) ) &
                                - ti11 * ( cc(1,1,k,4) - cc(1,1,k,3) )
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,1,k) = cc(1,i-1,k,1)+((wa1(i-2)*cc(1,i-1,k,2)+ &
        wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
        cc(1,i,k,5)))+((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))+(wa3(i-2)*cc(1,i-1,k,4)+ &
        wa3(i-1)*cc(1,i,k,4)))
      ch(1,i,1,k) = cc(1,i,k,1)+((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4)))
      ch(1,i-1,3,k) = cc(1,i-1,k,1)+tr11* &
        ( wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2) &
        +wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5))+tr12* &
        ( wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3) &
        +wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))+ti11* &
        ( wa1(i-2)*cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2) &
        -(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))+ti12* &
        ( wa2(i-2)*cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3) &
        -(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4)))
      ch(1,ic-1,2,k) = cc(1,i-1,k,1)+tr11* &
        ( wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2) &
        +wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5))+tr12* &
        ( wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3) &
        +wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))-(ti11* &
        ( wa1(i-2)*cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2) &
        -(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))+ti12* &
        ( wa2(i-2)*cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3) &
        -(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))
      ch(1,i,3,k) = (cc(1,i,k,1)+tr11*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr12*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))+(ti11*((wa4(i-2)*cc(1,i-1,k,5)+ &
        wa4(i-1)*cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))+ti12*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))
      ch(1,ic,2,k) = (ti11*((wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
        cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))+ti12*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))-(cc(1,i,k,1)+tr11*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr12*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))
      ch(1,i-1,5,k) = (cc(1,i-1,k,1)+tr12*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)* &
        cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5)))+tr11*((wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))+(wa3(i-2)* &
        cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))))+(ti12*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa4(i-2)* &
        cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))-ti11*((wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))-(wa3(i-2)* &
        cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))
      ch(1,ic-1,4,k) = (cc(1,i-1,k,1)+tr12*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)* &
        cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5)))+tr11*((wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))+(wa3(i-2)* &
        cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))))-(ti12*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa4(i-2)* &
        cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))-ti11*((wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))-(wa3(i-2)* &
        cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))
      ch(1,i,5,k) = (cc(1,i,k,1)+tr12*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr11*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))+(ti12*((wa4(i-2)*cc(1,i-1,k,5)+ &
        wa4(i-1)*cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))-ti11*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))
      ch(1,ic,4,k) = (ti12*((wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
        cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))-ti11*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))-(cc(1,i,k,1)+tr12*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr11*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))
     end do
  end do

  return
end
subroutine d1fgkb ( ido, ip, l1, idl1, cc, c1, c2, in1, ch, ch2, in2, wa )

!*****************************************************************************80
!
!! D1FGKB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1

  real ( kind = 8 ) ai1
  real ( kind = 8 ) ai2
  real ( kind = 8 ) ar1
  real ( kind = 8 ) ar1h
  real ( kind = 8 ) ar2
  real ( kind = 8 ) ar2h
  real ( kind = 8 ) arg
  real ( kind = 8 ) c1(in1,ido,l1,ip)
  real ( kind = 8 ) c2(in1,idl1,ip)
  real ( kind = 8 ) cc(in1,ido,ip,l1)
  real ( kind = 8 ) ch(in2,ido,l1,ip)
  real ( kind = 8 ) ch2(in2,idl1,ip)
  real ( kind = 8 ) dc2
  real ( kind = 8 ) dcp
  real ( kind = 8 ) ds2
  real ( kind = 8 ) dsp
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idij
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) ik
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) j2
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) nbd
  real ( kind = 8 ) tpi
  real ( kind = 8 ) wa(ido)

  tpi = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 )
  arg = tpi / real ( ip, kind = 8 )
  dcp = cos ( arg )
  dsp = sin ( arg )
  idp2 = ido + 2
  nbd = ( ido - 1 ) / 2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  if ( ido < l1 ) then
    do i = 1, ido
      do k = 1, l1
        ch(1,i,k,1) = cc(1,i,1,k)
      end do
    end do
  else
    do k = 1, l1
      do i = 1, ido
        ch(1,i,k,1) = cc(1,i,1,k)
      end do
    end do
  end if

  do j = 2, ipph
    jc = ipp2 - j
    j2 = j + j
    do k = 1, l1
      ch(1,1,k,j) = cc(1,ido,j2-2,k)+cc(1,ido,j2-2,k)
      ch(1,1,k,jc) = cc(1,1,j2-1,k)+cc(1,1,j2-1,k)
    end do
  end do

  if ( ido == 1 ) then

  else if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      do i = 3, ido, 2
        ic = idp2 - i
        do k = 1, l1
          ch(1,i-1,k,j) = cc(1,i-1,2*j-1,k)+cc(1,ic-1,2*j-2,k)
          ch(1,i-1,k,jc) = cc(1,i-1,2*j-1,k)-cc(1,ic-1,2*j-2,k)
          ch(1,i,k,j) = cc(1,i,2*j-1,k)-cc(1,ic,2*j-2,k)
          ch(1,i,k,jc) = cc(1,i,2*j-1,k)+cc(1,ic,2*j-2,k)
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do k = 1, l1
        do i = 3, ido, 2
          ic = idp2 - i
          ch(1,i-1,k,j) = cc(1,i-1,2*j-1,k)+cc(1,ic-1,2*j-2,k)
          ch(1,i-1,k,jc) = cc(1,i-1,2*j-1,k)-cc(1,ic-1,2*j-2,k)
          ch(1,i,k,j) = cc(1,i,2*j-1,k)-cc(1,ic,2*j-2,k)
          ch(1,i,k,jc) = cc(1,i,2*j-1,k)+cc(1,ic,2*j-2,k)
        end do
      end do
    end do

  end if

  ar1 = 1.0D+00
  ai1 = 0.0D+00

  do l = 2, ipph

    lc = ipp2 - l
    ar1h = dcp * ar1 - dsp * ai1
    ai1 = dcp * ai1 + dsp * ar1
    ar1 = ar1h

    do ik = 1, idl1
      c2(1,ik,l) = ch2(1,ik,1)+ar1*ch2(1,ik,2)
      c2(1,ik,lc) = ai1*ch2(1,ik,ip)
    end do

    dc2 = ar1
    ds2 = ai1
    ar2 = ar1
    ai2 = ai1

    do j = 3, ipph

      jc = ipp2 - j
      ar2h = dc2*ar2-ds2*ai2
      ai2 = dc2*ai2+ds2*ar2
      ar2 = ar2h

      do ik = 1, idl1
        c2(1,ik,l) = c2(1,ik,l)+ar2*ch2(1,ik,j)
        c2(1,ik,lc) = c2(1,ik,lc)+ai2*ch2(1,ik,jc)
      end do

    end do

  end do

  do j = 2, ipph
    do ik = 1, idl1
      ch2(1,ik,1) = ch2(1,ik,1)+ch2(1,ik,j)
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do k = 1, l1
      ch(1,1,k,j) = c1(1,1,k,j)-c1(1,1,k,jc)
      ch(1,1,k,jc) = c1(1,1,k,j)+c1(1,1,k,jc)
    end do
  end do

  if ( ido == 1 ) then

  else if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      do i = 3, ido, 2
        do k = 1, l1
          ch(1,i-1,k,j)  = c1(1,i-1,k,j) - c1(1,i,k,jc)
          ch(1,i-1,k,jc) = c1(1,i-1,k,j) + c1(1,i,k,jc)
          ch(1,i,k,j)    = c1(1,i,k,j)   + c1(1,i-1,k,jc)
          ch(1,i,k,jc)   = c1(1,i,k,j)   - c1(1,i-1,k,jc)
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do k = 1, l1
        do i = 3, ido, 2
          ch(1,i-1,k,j) = c1(1,i-1,k,j)-c1(1,i,k,jc)
          ch(1,i-1,k,jc) = c1(1,i-1,k,j)+c1(1,i,k,jc)
          ch(1,i,k,j) = c1(1,i,k,j)+c1(1,i-1,k,jc)
          ch(1,i,k,jc) = c1(1,i,k,j)-c1(1,i-1,k,jc)
        end do
      end do
    end do

  end if

  if ( ido == 1 ) then
    return
  end if

  do ik = 1, idl1
    c2(1,ik,1) = ch2(1,ik,1)
  end do

  do j = 2, ip
    do k = 1, l1
      c1(1,1,k,j) = ch(1,1,k,j)
    end do
  end do

  if ( l1 < nbd ) then

    is = -ido
    do j = 2, ip
       is = is + ido
       do k = 1, l1
         idij = is
         do i = 3, ido, 2
           idij = idij + 2
           c1(1,i-1,k,j) = wa(idij-1)*ch(1,i-1,k,j)-wa(idij)* ch(1,i,k,j)
           c1(1,i,k,j) = wa(idij-1)*ch(1,i,k,j)+wa(idij)* ch(1,i-1,k,j)
         end do
       end do
    end do

  else

    is = -ido

    do j = 2, ip
      is = is + ido
      idij = is
      do i = 3, ido, 2
        idij = idij + 2
        do k = 1, l1
           c1(1,i-1,k,j) = wa(idij-1) * ch(1,i-1,k,j) - wa(idij) * ch(1,i,k,j)
           c1(1,i,k,j)   = wa(idij-1) * ch(1,i,k,j)   + wa(idij) * ch(1,i-1,k,j)
        end do
      end do
    end do

  end if

  return
end
subroutine d1fgkf ( ido, ip, l1, idl1, cc, c1, c2, in1, ch, ch2, in2, wa )

!*****************************************************************************80
!
!! D1FGKF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1

  real ( kind = 8 ) ai1
  real ( kind = 8 ) ai2
  real ( kind = 8 ) ar1
  real ( kind = 8 ) ar1h
  real ( kind = 8 ) ar2
  real ( kind = 8 ) ar2h
  real ( kind = 8 ) arg
  real ( kind = 8 ) c1(in1,ido,l1,ip)
  real ( kind = 8 ) c2(in1,idl1,ip)
  real ( kind = 8 ) cc(in1,ido,ip,l1)
  real ( kind = 8 ) ch(in2,ido,l1,ip)
  real ( kind = 8 ) ch2(in2,idl1,ip)
  real ( kind = 8 ) dc2
  real ( kind = 8 ) dcp
  real ( kind = 8 ) ds2
  real ( kind = 8 ) dsp
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idij
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) ik
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) j2
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) nbd
  real ( kind = 8 ) tpi
  real ( kind = 8 ) wa(ido)

  tpi = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 )
  arg = tpi / real ( ip, kind = 8 )
  dcp = cos ( arg )
  dsp = sin ( arg )
  ipph = ( ip + 1 ) / 2
  ipp2 = ip + 2
  idp2 = ido + 2
  nbd = ( ido - 1 ) / 2

  if ( ido == 1 ) then

    do ik = 1, idl1
      c2(1,ik,1) = ch2(1,ik,1)
    end do
 
  else

    do ik = 1, idl1
      ch2(1,ik,1) = c2(1,ik,1)
    end do

    do j = 2, ip
      do k = 1, l1
        ch(1,1,k,j) = c1(1,1,k,j)
      end do
    end do

    if ( l1 < nbd ) then

      is = -ido

      do j = 2, ip
        is = is + ido
        do k = 1, l1
          idij = is
          do i = 3, ido, 2
            idij = idij + 2
            ch(1,i-1,k,j) = wa(idij-1)*c1(1,i-1,k,j)+wa(idij) *c1(1,i,k,j)
            ch(1,i,k,j) = wa(idij-1)*c1(1,i,k,j)-wa(idij) *c1(1,i-1,k,j)
          end do
        end do
      end do

    else

      is = -ido

      do j = 2, ip
        is = is + ido
        idij = is
        do i = 3, ido, 2
          idij = idij + 2
          do k = 1, l1
            ch(1,i-1,k,j) = wa(idij-1)*c1(1,i-1,k,j)+wa(idij) *c1(1,i,k,j)
            ch(1,i,k,j) = wa(idij-1)*c1(1,i,k,j)-wa(idij) *c1(1,i-1,k,j)
          end do
        end do
      end do

    end if

    if ( nbd < l1 ) then

      do j = 2, ipph
        jc = ipp2 - j
        do i = 3, ido, 2
          do k = 1, l1
            c1(1,i-1,k,j) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
            c1(1,i-1,k,jc) = ch(1,i,k,j)-ch(1,i,k,jc)
            c1(1,i,k,j) = ch(1,i,k,j)+ch(1,i,k,jc)
            c1(1,i,k,jc) = ch(1,i-1,k,jc)-ch(1,i-1,k,j)
          end do
        end do
      end do

    else

      do j = 2, ipph
        jc = ipp2 - j
        do k = 1, l1
          do i = 3, ido, 2
            c1(1,i-1,k,j) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
            c1(1,i-1,k,jc) = ch(1,i,k,j)-ch(1,i,k,jc)
            c1(1,i,k,j) = ch(1,i,k,j)+ch(1,i,k,jc)
            c1(1,i,k,jc) = ch(1,i-1,k,jc)-ch(1,i-1,k,j)
          end do
        end do
      end do

    end if

  end if

  do j = 2, ipph
    jc = ipp2 - j
    do k = 1, l1
      c1(1,1,k,j) = ch(1,1,k,j)+ch(1,1,k,jc)
      c1(1,1,k,jc) = ch(1,1,k,jc)-ch(1,1,k,j)
    end do
  end do

  ar1 = 1.0D+00
  ai1 = 0.0D+00

  do l = 2, ipph

    lc = ipp2 - l
    ar1h = dcp * ar1 - dsp * ai1
    ai1 = dcp * ai1 + dsp * ar1
    ar1 = ar1h

    do ik = 1, idl1
      ch2(1,ik,l) = c2(1,ik,1)+ar1*c2(1,ik,2)
      ch2(1,ik,lc) = ai1*c2(1,ik,ip)
    end do

    dc2 = ar1
    ds2 = ai1
    ar2 = ar1
    ai2 = ai1

    do j = 3, ipph
      jc = ipp2 - j
      ar2h = dc2 * ar2 - ds2 * ai2
      ai2 = dc2 * ai2 + ds2 * ar2
      ar2 = ar2h
      do ik = 1, idl1
        ch2(1,ik,l) = ch2(1,ik,l)+ar2*c2(1,ik,j)
        ch2(1,ik,lc) = ch2(1,ik,lc)+ai2*c2(1,ik,jc)
      end do
    end do

  end do

  do j = 2, ipph
    do ik = 1, idl1
      ch2(1,ik,1) = ch2(1,ik,1)+c2(1,ik,j)
    end do
  end do

  if ( ido < l1 ) then

    do i = 1, ido
      do k = 1, l1
        cc(1,i,1,k) = ch(1,i,k,1)
      end do
    end do

  else

    do k = 1, l1
      do i = 1, ido
        cc(1,i,1,k) = ch(1,i,k,1)
      end do
    end do

  end if

  do j = 2, ipph
    jc = ipp2 - j
    j2 = j+j
    do k = 1, l1
      cc(1,ido,j2-2,k) = ch(1,1,k,j)
      cc(1,1,j2-1,k) = ch(1,1,k,jc)
    end do
  end do

  if ( ido == 1 ) then
    return
  end if

  if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      j2 = j + j
      do i = 3, ido, 2
        ic = idp2 - i
        do k = 1, l1
          cc(1,i-1,j2-1,k) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
          cc(1,ic-1,j2-2,k) = ch(1,i-1,k,j)-ch(1,i-1,k,jc)
          cc(1,i,j2-1,k) = ch(1,i,k,j)+ch(1,i,k,jc)
          cc(1,ic,j2-2,k) = ch(1,i,k,jc)-ch(1,i,k,j)
        end do
      end do
   end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      j2 = j + j
      do k = 1, l1
        do i = 3, ido, 2
          ic = idp2 - i
          cc(1,i-1,j2-1,k)  = ch(1,i-1,k,j) + ch(1,i-1,k,jc)
          cc(1,ic-1,j2-2,k) = ch(1,i-1,k,j) - ch(1,i-1,k,jc)
          cc(1,i,j2-1,k)    = ch(1,i,k,j)   + ch(1,i,k,jc)
          cc(1,ic,j2-2,k)   = ch(1,i,k,jc)  - ch(1,i,k,j)
        end do
      end do
    end do

  end if

  return
end
subroutine dcosq1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DCOSQ1B: real double precision backward cosine quarter wave transform, 1D.
!
!  Discussion:
!
!    DCOSQ1B computes the one-dimensional Fourier transform of a sequence 
!    which is a cosine series with odd wave numbers.  This transform is 
!    referred to as the backward transform or Fourier synthesis, transforming
!    the sequence from spectral to physical space.
!
!    This transform is normalized since a call to DCOSQ1B followed
!    by a call to DCOSQ1F (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    17 November 2007
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) N, the number of elements to be transformed 
!    in the sequence.  The transform is most efficient when N is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 8 ) R(LENR); on input, containing the sequence 
!    to be transformed, and on output, containing the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to DCOSQ1I before the first call to routine 
!    DCOSQ1F or DCOSQ1B for a given transform length N.  WSAVE's contents may 
!    be re-used for subsequent calls to DCOSQ1F and DCOSQ1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 8 ) ssqrt2
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) x1

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'DCOSQ1B', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DCOSQ1B', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'DCOSQ1B', 10 ) 
    return
  end if

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt2 = 1.0D+00 / sqrt ( 2.0D+00 )
    x1 = x(1,1) + x(1,2)
    x(1,2) = ssqrt2 * ( x(1,1) - x(1,2) )
    x(1,1) = x1
    return
  end if

  call dcosqb1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DCOSQ1B', -5 )
    return
  end if

  return
end
subroutine dcosq1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DCOSQ1F: real double precision forward cosine quarter wave transform, 1D.
!
!  Discussion:
!
!    DCOSQ1F computes the one-dimensional Fourier transform of a sequence 
!    which is a cosine series with odd wave numbers.  This transform is 
!    referred to as the forward transform or Fourier analysis, transforming 
!    the sequence from physical to spectral space.
!
!    This transform is normalized since a call to DCOSQ1F followed
!    by a call to DCOSQ1B (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!     17 November 2007
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) N, the number of elements to be transformed in 
!    the sequence.  The transform is most efficient when N is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 8 ) R(LENR); on input, containing the sequence 
!    to be transformed, and on output, containing the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to DCOSQ1I before the first call to routine 
!    DCOSQ1F or DCOSQ1B for a given transform length N.  WSAVE's contents may 
!    be re-used for subsequent calls to DCOSQ1F and DCOSQ1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) n
  integer ( kind = 4 ) lenx
  real ( kind = 8 ) ssqrt2
  real ( kind = 8 ) tsqx
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'dcosq1f', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'dcosq1f', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'dcosq1f', 10 )
    return
  end if

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt2 = 1.0D+00 / sqrt ( 2.0D+00 )
    tsqx = ssqrt2 * x(1,2)
    x(1,2) = 0.5D+00 * x(1,1) - tsqx
    x(1,1) = 0.5D+00 * x(1,1) + tsqx
    return
  end if

  call dcosqf1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'dcosq1f', -5 )
    return
  end if

  return
end
subroutine dcosq1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! DCOSQ1I: initialization for DCOSQ1B and DCOSQ1F.
!
!  Discussion:
!
!    DCOSQ1I initializes array WSAVE for use in its companion routines 
!    DCOSQ1F and DCOSQ1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    17 November 2007
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N is a product of small 
!    primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 8 ) WSAVE(LENSAV), containing the prime factors of 
!    N and also containing certain trigonometric values which will be used 
!    in routines DCOSQ1B or DCOSQ1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 8 ) dt
  real ( kind = 8 ) fk
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  real ( kind = 8 ) pih
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'dcosq1i', 3 )
    return
  end if

  pih = 2.0D+00 * atan ( 1.0D+00 )
  dt = pih / real ( n, kind = 8 )
  fk = 0.0D+00

  do k = 1, n
    fk = fk + 1.0D+00
    wsave(k) = cos ( fk * dt )
  end do

  lnsv = n + int ( log ( real ( n, kind = 8 ) ) ) + 4

  call dfft1i ( n, wsave(n+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'dcosq1i', -5 )
    return
  end if

  return
end
subroutine dcosqb1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! DCOSQB1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!     17 November 2007
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) inc

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np2
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) work(*)
  real ( kind = 8 ) wsave(*)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) xim1

  ier = 0
  ns2 = ( n + 1 ) / 2
  np2 = n + 2

  do i = 3, n, 2
    xim1 = x(1,i-1) + x(1,i)
    x(1,i) = 0.5D+00 * ( x(1,i-1) - x(1,i) )
    x(1,i-1) = 0.5D+00 * xim1
  end do

  x(1,1) = 0.5D+00 * x(1,1)
  modn = mod ( n, 2 )

  if ( modn == 0 ) then
    x(1,n) = 0.5D+00 * x(1,n)
  end if

  lenx = inc * ( n - 1 )  + 1
  lnsv = n + int ( log ( real ( n, kind = 8 ) ) ) + 4
  lnwk = n

  call dfft1b ( n, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'dcosqb1', -5 )
    return
  end if

  do k = 2, ns2
    kc = np2 - k
    work(k)  = wsave(k-1) * x(1,kc) + wsave(kc-1) * x(1,k)
    work(kc) = wsave(k-1) * x(1,k)  - wsave(kc-1) * x(1,kc)
  end do

  if ( modn == 0 ) then
    x(1,ns2+1) = wsave(ns2) * ( x(1,ns2+1) + x(1,ns2+1) )
  end if

  do k = 2, ns2
    kc = np2 - k
    x(1,k)  = work(k) + work(kc)
    x(1,kc) = work(k) - work(kc)
  end do

  x(1,1) = x(1,1) + x(1,1)

  return
end
subroutine dcosqf1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! DCOSQF1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    17 November 2007
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) inc

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np2
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) work(*)
  real ( kind = 8 ) wsave(*)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) xim1

  ier = 0
  ns2 = ( n + 1 ) / 2
  np2 = n + 2

  do k = 2, ns2
    kc = np2 - k
    work(k)  = x(1,k) + x(1,kc)
    work(kc) = x(1,k) - x(1,kc)
  end do

  modn = mod ( n, 2 )

  if ( modn == 0 ) then
    work(ns2+1) = x(1,ns2+1) + x(1,ns2+1)
  end if

  do k = 2, ns2
    kc = np2 - k
    x(1,k)  = wsave(k-1) * work(kc) + wsave(kc-1) * work(k)
    x(1,kc) = wsave(k-1) * work(k)  - wsave(kc-1) * work(kc)
  end do

  if ( modn == 0 ) then
    x(1,ns2+1) = wsave(ns2) * work(ns2+1)
  end if

  lenx = inc * ( n - 1 ) + 1
  lnsv = n + int ( log ( real ( n, kind = 8 ) ) ) + 4
  lnwk = n

  call dfft1f ( n, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'dcosqf1', -5 )
    return
  end if

  do i = 3, n, 2
    xim1   = 0.5D+00 * ( x(1,i-1) + x(1,i) )
    x(1,i) = 0.5D+00 * ( x(1,i-1) - x(1,i) )
    x(1,i-1) = xim1
  end do

  return
end
subroutine dcost1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DCOST1B: real double precision backward cosine transform, 1D.
!
!  Discussion:
!
!    DCOST1B computes the one-dimensional Fourier transform of an even 
!    sequence within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the sequence 
!    from spectral to physical space.
!
!    This transform is normalized since a call to DCOST1B followed
!    by a call to COST1F (or vice-versa) reproduces the original array 
!    within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 8 ) R(LENR), containing the sequence 
!    to be transformed.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be
!    initialized with a call to DCOST1I before the first call to routine 
!    DCOST1F or DCOST1B for a given transform length N.  WSAVE's contents 
!    may be re-used for subsequent calls to DCOST1F and DCOST1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N-1.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'DCOST1B', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DCOST1B', 8 )
    return
  end if

  if ( lenwrk < n - 1 ) then
    ier = 3
    call xerfft ( 'DCOST1B', 10 )
    return
  end if

  call dcostb1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DCOST1B', -5 )
    return
  end if

  return
end
subroutine dcost1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DCOST1F: real double precision forward cosine transform, 1D.
!
!  Discussion:
!
!    DCOST1F computes the one-dimensional Fourier transform of an even 
!    sequence within a real array.  This transform is referred to as the 
!    forward transform or Fourier analysis, transforming the sequence 
!    from  physical to spectral space.
!
!    This transform is normalized since a call to DCOST1F followed by a call 
!    to DCOST1B (or vice-versa) reproduces the original array within 
!    roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 8 ) R(LENR), containing the sequence 
!    to be transformed.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be
!    initialized with a call to DCOST1I before the first call to routine 
!    DCOST1F or DCOST1B for a given transform length N.  WSAVE's contents 
!    may be re-used for subsequent calls to DCOST1F and DCOST1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N-1.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'DCOST1F', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DCOST1F', 8 )
    return
  end if

  if ( lenwrk < n - 1 ) then
    ier = 3
    call xerfft ( 'DCOST1F', 10 )
    return
  end if

  call dcostf1 ( n, inc, x, wsave, work, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DCOST1F', -5 )
    return
  end if

  return
end
subroutine dcost1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! DCOST1I: initialization for DCOST1B and DCOST1F.
!
!  Discussion:
!
!    DCOST1I initializes array WSAVE for use in its companion routines 
!    DCOST1F and DCOST1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N-1 is a product
!    of small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, dimension of WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 8 ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!    in routines DCOST1B or DCOST1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 8 ) dt
  real ( kind = 8 ) fk
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) pi
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DCOST1I', 3 )
    return
  end if

  if ( n <= 3 ) then
    return
  end if

  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2
  pi = 4.0E+00 * atan ( 1.0D+00 )
  dt = pi / real ( nm1, kind = 8 )
  fk = 0.0E+00
  do k = 2, ns2
    kc = np1 - k
    fk = fk + 1.0D+00
    wsave(k) = 2.0D+00 * sin ( fk * dt )
    wsave(kc) = 2.0D+00 * cos ( fk * dt )
  end do 

  lnsv = nm1 + int ( log ( real ( nm1, kind = 8 ) ) ) + 4

  call dfft1i ( nm1, wsave(n+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DCOST1I', -5 )
    return
  end if

  return
end
subroutine dcostb1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! DCOSTB1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  real ( kind = 8 ) fnm1s2
  real ( kind = 8 ) fnm1s4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) t1
  real ( kind = 8 ) t2
  real ( kind = 8 ) work(*)
  real ( kind = 8 ) wsave(*)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) x1h
  real ( kind = 8 ) x1p3
  real ( kind = 8 ) x2
  real ( kind = 8 ) xi

  ier = 0
  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    x1h    = x(1,1) + x(1,2)
    x(1,2) = x(1,1) - x(1,2)
    x(1,1) = x1h
    return
  end if

  if ( n == 3 ) then
    x1p3 = x(1,1) + x(1,3)
    x2 = x(1,2)
    x(1,2) = x(1,1) - x(1,3)
    x(1,1) = x1p3 + x2
    x(1,3) = x1p3 - x2
    return
  end if

  x(1,1) = x(1,1) + x(1,1)
  x(1,n) = x(1,n) + x(1,n)
  dsum = x(1,1) - x(1,n)
  x(1,1) = x(1,1) + x(1,n)

  do k = 2, ns2
    kc = np1 - k
    t1 = x(1,k) + x(1,kc)
    t2 = x(1,k) - x(1,kc)
    dsum = dsum + wsave(kc) * t2
    t2 = wsave(k) * t2
    x(1,k) = t1 - t2
    x(1,kc) = t1 + t2
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    x(1,ns2+1) = x(1,ns2+1) + x(1,ns2+1)
  end if

  lenx = inc * ( nm1 - 1 )  + 1
  lnsv = nm1 + int ( log ( real ( nm1, kind = 8 ) ) ) + 4
  lnwk = nm1

  call dfft1f ( nm1, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DCOSTB1', -5 )
    return
  end if

  fnm1s2 = real ( nm1, kind = 8 ) / 2.0D+00
  dsum = 0.5D+00 * dsum
  x(1,1) = fnm1s2 * x(1,1)

  if ( mod ( nm1, 2 ) == 0 ) then
    x(1,nm1) = x(1,nm1) + x(1,nm1)
  end if

  fnm1s4 = real ( nm1, kind = 8 ) / 4.0D+00

  do i = 3, n, 2
    xi = fnm1s4 * x(1,i)
    x(1,i) = fnm1s4 * x(1,i-1)
    x(1,i-1) = dsum
    dsum = dsum + xi
  end do

  if ( modn == 0 ) then
    x(1,n) = dsum
  end if

  return
end
subroutine dcostf1 ( n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! DCOSTF1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) snm1
  real ( kind = 8 ) t1
  real ( kind = 8 ) t2
  real ( kind = 8 ) tx2
  real ( kind = 8 ) work(*)
  real ( kind = 8 ) wsave(*)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) x1h
  real ( kind = 8 ) x1p3
  real ( kind = 8 ) xi

  ier = 0
  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    x1h = x(1,1) + x(1,2)
    x(1,2) = 0.5D+00 * ( x(1,1) - x(1,2) )
    x(1,1) = 0.5D+00 * x1h
    return
  end if

  if ( n == 3 ) then
    x1p3 = x(1,1) + x(1,3)
    tx2 = x(1,2) + x(1,2)
    x(1,2) = 0.5D+00 * ( x(1,1) - x(1,3) )
    x(1,1) = 0.25D+00 * ( x1p3 + tx2 )
    x(1,3) = 0.25D+00 * ( x1p3 - tx2 )
    return
  end if

  dsum = x(1,1) - x(1,n)
  x(1,1) = x(1,1) + x(1,n)
  do k = 2, ns2
    kc = np1 - k
    t1 = x(1,k) + x(1,kc)
    t2 = x(1,k) - x(1,kc)
    dsum = dsum + wsave(kc) * t2
    t2 = wsave(k) * t2
    x(1,k) = t1 - t2
    x(1,kc) = t1 + t2
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    x(1,ns2+1) = x(1,ns2+1) + x(1,ns2+1)
  end if

  lenx = inc * ( nm1 - 1 )  + 1
  lnsv = nm1 + int ( log ( real ( nm1, kind = 8 ) ) ) + 4
  lnwk = nm1

  call dfft1f ( nm1, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DCOSTF1', -5 )
    return
  end if

  snm1 = 1.0D+00 / real ( nm1, kind = 8 )
  dsum = snm1 * dsum

  if ( mod ( nm1, 2 ) == 0 ) then
    x(1,nm1) = x(1,nm1) + x(1,nm1)
  end if

  do i = 3, n, 2
    xi = 0.5D+00 * x(1,i)
    x(1,i) = 0.5D+00 * x(1,i-1)
    x(1,i-1) = dsum
    dsum = dsum + xi
  end do

  if ( modn == 0 ) then
    x(1,n) = dsum
  end if

  x(1,1) = 0.5D+00 * x(1,1)
  x(1,n) = 0.5D+00 * x(1,n)

  return
end
subroutine dfftb1 ( n, in, c, ch, wa, fac )

!*****************************************************************************80
!
!! DFFTB1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) in
  integer ( kind = 4 ) n

  real ( kind = 8 ) c(in,*)
  real ( kind = 8 ) ch(*)
  real ( kind = 8 ) fac(15)
  real ( kind = 8 ) half
  real ( kind = 8 ) halfm
  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) ix2
  integer ( kind = 4 ) ix3
  integer ( kind = 4 ) ix4
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  real ( kind = 8 ) wa(n)

  nf = int ( fac(2) )
  na = 0

  do k1 = 1, nf

    ip = int ( fac(k1+2) )
    na = 1 - na

    if ( 5 < ip ) then
      if ( k1 /= nf ) then
        na = 1 - na
      end if
    end if

  end do

  half = 0.5D+00
  halfm = -0.5D+00
  modn = mod ( n, 2 )
  nl = n - 2
  if ( modn /= 0 ) then
    nl = n - 1
  end if

  if ( na == 0 ) then

    do j = 2, nl, 2
      c(1,j) = half * c(1,j)
      c(1,j+1) = halfm * c(1,j+1)
    end do

  else

    ch(1) = c(1,1)
    ch(n) = c(1,n)

    do j = 2, nl, 2
      ch(j) = half * c(1,j)
      ch(j+1) = halfm * c(1,j+1)
    end do

  end if

  l1 = 1
  iw = 1

  do k1 = 1, nf

    ip = int ( fac(k1+2) )
    l2 = ip * l1
    ido = n / l2
    idl1 = ido * l1

    if ( ip == 4 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido

      if ( na == 0 ) then
        call d1f4kb ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3) )
      else
        call d1f4kb ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3) )
      end if

      na = 1 - na

    else if ( ip == 2 ) then

      if ( na == 0 ) then
        call d1f2kb ( ido, l1, c, in, ch, 1, wa(iw) )
      else
        call d1f2kb ( ido, l1, ch, 1, c, in, wa(iw) )
      end if

      na = 1 - na

    else if ( ip == 3 ) then

      ix2 = iw + ido

      if ( na == 0 ) then
        call d1f3kb ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2) )
      else
        call d1f3kb ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2) )
      end if

      na = 1 - na

    else if ( ip == 5 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido
      ix4 = ix3 + ido

      if ( na == 0 ) then
        call d1f5kb ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      else
        call d1f5kb ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      end if

      na = 1 - na

    else

      if ( na == 0 ) then
        call d1fgkb ( ido, ip, l1, idl1, c, c, c, in, ch, ch, 1, wa(iw) )
      else
        call d1fgkb ( ido, ip, l1, idl1, ch, ch, ch, 1, c, c, in, wa(iw) )
      end if

      if ( ido == 1 ) then
        na = 1 - na
      end if

    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ido

  end do

  return
end
subroutine dfft1b ( n, inc, r, lenr, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DFFT1B: real double precision backward fast Fourier transform, 1D.
!
!  Discussion:
!
!    DFFT1B computes the one-dimensional Fourier transform of a periodic
!    sequence within a real array.  This is referred to as the backward
!    transform or Fourier synthesis, transforming the sequence from 
!    spectral to physical space.  This transform is normalized since a 
!    call to DFFT1B followed by a call to DFFT1F (or vice-versa) reproduces
!    the original array within roundoff error. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    16 November 2007
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence. 
!
!    Input/output, real ( kind = 8 ) R(LENR), on input, the data to be 
!    transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1) + 1. 
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to DFFT1I before the first call to routine
!    DFFT1F or DFFT1B for a given transform length N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4. 
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough.
!
  implicit none

  integer ( kind = 4 ) lenr
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) n
  real ( kind = 8 ) r(lenr)
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lenr < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'rfft1b ', 6 )
    return
  end if

  if ( lensav < n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DFFT1B ', 8 )
    return
  end if

  if ( lenwrk < n ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'DFFT1B - Fatal error!'
    write ( *, '(a)' ) '  LENWRK < N:'
    write ( *, '(a,i6)' ) '  LENWRK = ', lenwrk
    write ( *, '(a,i6)' ) '  N = ', n
    ier = 3
    call xerfft ( 'DFFT1B ', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call dfftb1 ( n, inc, r, work, wsave, wsave(n+1) )

  return
end
subroutine dfft1f ( n, inc, r, lenr, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DFFT1F: real double precision forward fast Fourier transform, 1D.
!
!  Discussion:
!
!    DFFT1F computes the one-dimensional Fourier transform of a periodic
!    sequence within a real array.  This is referred to as the forward 
!    transform or Fourier analysis, transforming the sequence from physical
!    to spectral space.  This transform is normalized since a call to 
!    DFFT1F followed by a call to DFFT1B (or vice-versa) reproduces the 
!    original array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence. 
!
!    Input/output, real ( kind = 8 ) R(LENR), on input, contains the sequence
!    to be transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1) + 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be
!    initialized with a call to DFFT1I before the first call to routine DFFT1F
!    or DFFT1B for a given transform length N. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough:
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough.
!
  implicit none

  integer ( kind = 4 ) lenr
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) r(lenr)

  ier = 0

  if ( lenr < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'DFFT1F', 6 )
    return
  end if

  if ( lensav < n + int ( log ( real ( n, kind = 8  ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DFFT1F', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'DFFT1F', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call dfftf1 ( n, inc, r, work, wsave, wsave(n+1) )

  return
end
subroutine dfftf1 ( n, in, c, ch, wa, fac )

!*****************************************************************************80
!
!! DFFTF1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) in
  integer ( kind = 4 ) n

  real ( kind = 8 ) c(in,*)
  real ( kind = 8 ) ch(*)
  real ( kind = 8 ) fac(15)
  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) ix2
  integer ( kind = 4 ) ix3
  integer ( kind = 4 ) ix4
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) kh
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  real ( kind = 8 ) sn
  real ( kind = 8 ) tsn
  real ( kind = 8 ) tsnm
  real ( kind = 8 ) wa(n)

  nf = int ( fac(2) )
  na = 1
  l2 = n
  iw = n

  do k1 = 1, nf

    kh = nf - k1
    ip = int ( fac(kh+3) )
    l1 = l2 / ip
    ido = n / l2
    idl1 = ido * l1
    iw = iw - ( ip - 1 ) * ido
    na = 1 - na

    if ( ip == 4 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido

      if ( na == 0 ) then
        call d1f4kf ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3) )
      else
        call d1f4kf ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3) )
      end if

    else if ( ip == 2 ) then

      if ( na == 0 ) then
        call d1f2kf ( ido, l1, c, in, ch, 1, wa(iw) )
      else
        call d1f2kf ( ido, l1, ch, 1, c, in, wa(iw) )
      end if

    else if ( ip == 3 ) then

      ix2 = iw + ido

      if ( na == 0 ) then
        call d1f3kf ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2) )
      else 
        call d1f3kf ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2) )
      end if

    else if ( ip == 5 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido
      ix4 = ix3 + ido

      if ( na == 0 ) then
        call d1f5kf ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      else
        call d1f5kf ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      end if

    else

      if ( ido == 1 ) then
        na = 1 - na
      end if

      if ( na == 0 ) then
        call d1fgkf ( ido, ip, l1, idl1, c, c, c, in, ch, ch, 1, wa(iw) )
        na = 1
      else
        call d1fgkf ( ido, ip, l1, idl1, ch, ch, ch, 1, c, c, in, wa(iw) )
        na = 0
      end if

    end if

    l2 = l1

  end do

  sn = 1.0D+00 / real ( n, kind = 8 )
  tsn = 2.0D+00 / real ( n, kind = 8 )
  tsnm = - tsn
  modn = mod ( n, 2 )
  nl = n - 2

  if ( modn /= 0 ) then
    nl = n - 1
  end if

  if ( na == 0 ) then

    c(1,1) = sn * ch(1)
    do j = 2, nl, 2
      c(1,j) = tsn * ch(j)
      c(1,j+1) = tsnm * ch(j+1)
    end do

    if ( modn == 0 ) then
      c(1,n) = sn * ch(n)
    end if

  else

    c(1,1) = sn * c(1,1)

    do j = 2, nl, 2
      c(1,j) = tsn * c(1,j)
      c(1,j+1) = tsnm * c(1,j+1)
    end do

    if ( modn == 0 ) then
      c(1,n) = sn * c(1,n)
    end if

  end if

  return
end
subroutine dfft1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! DFFT1I: initialization for DFFT1B and DFFT1F.
!
!  Discussion:
!
!    DFFT1I initializes array WSAVE for use in its companion routines 
!    DFFT1B and DFFT1F.  The prime factorization of N together with a
!    tabulation of the trigonometric functions are computed and stored
!    in array WSAVE.  Separate WSAVE arrays are required for different
!    values of N. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 8 ) WSAVE(LENSAV), containing the prime factors
!    of N and also containing certain trigonometric values which will be used
!    in routines DFFT1B or DFFT1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) n
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DFFT1I ', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call dffti1 ( n, wsave(1), wsave(n+1) )

  return
end
subroutine dffti1 ( n, wa, fac )

!*****************************************************************************80
!
!! DFFTI1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    Output, real ( kind = 8 ) WA(N), trigonometric information.
!
!    Output, real ( kind = 8 ) FAC(15), factorization information.  
!    FAC(1) is N, FAC(2) is NF, the number of factors, and FAC(3:NF+2) are the 
!    factors.
!
  implicit none

  integer ( kind = 4 ) n

  real ( kind = 8 ) arg
  real ( kind = 8 ) argh
  real ( kind = 8 ) argld
  real ( kind = 8 ) fac(15)
  real ( kind = 8 ) fi
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ib
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ii
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) ipm
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) ld
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nfm1
  integer ( kind = 4 ) nl
  integer ( kind = 4 ) nq
  integer ( kind = 4 ) nr
  integer ( kind = 4 ) ntry
  real ( kind = 8 ) tpi
  real ( kind = 8 ) wa(n)

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
      fac(nf+2) = real ( ntry, kind = 8 )
      nl = nq
!
!  If 2 is a factor, make sure it appears first in the list of factors.
!
      if ( ntry == 2 ) then
        if ( nf /= 1 ) then
          do i = 2, nf
            ib = nf - i + 2
            fac(ib+2) = fac(ib+1)
          end do
          fac(3) = 2.0D+00
        end if
      end if

    end do

  end do

  fac(1) = real ( n, kind = 8 )
  fac(2) = real ( nf, kind = 8 )
  tpi = 8.0D+00 * atan ( 1.0D+00 )
  argh = tpi / real ( n, kind = 8 )
  is = 0
  nfm1 = nf - 1
  l1 = 1

  do k1 = 1, nfm1
    ip = int ( fac(k1+2) )
    ld = 0
    l2 = l1 * ip
    ido = n / l2
    ipm = ip - 1
    do j = 1, ipm
      ld = ld + l1
      i = is
      argld = real ( ld, kind = 8 ) * argh
      fi = 0.0D+00
      do ii = 3, ido, 2
        i = i + 2
        fi = fi + 1.0D+00
        arg = fi * argld
        wa(i-1) = real ( cos ( arg ), kind = 8 )
        wa(i) = real ( sin ( arg ), kind = 8 )
      end do
      is = is + ido
    end do
    l1 = l2
  end do

  return
end
subroutine dsint1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DSINT1B: real double precision backward sine transform, 1D.
!
!  Discussion:
!
!    DSINT1B computes the one-dimensional Fourier transform of an odd 
!    sequence within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequence from spectral to physical space.
!
!    This transform is normalized since a call to DSINT1B followed
!    by a call to DSINT1F (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 8 ) R(LENR), on input, contains the 
!    sequence to be transformed, and on output, the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be
!    initialized with a call to DSINT1I before the first call to routine 
!    SINT1F or SINT1B for a given transform length N.  WSAVE's contents 
!    may be re-used for subsequent calls to DSINT1F and DSINT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N+2.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'DSINT1B', 6 )
    return
  end if

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DSINT1B', 8 )
    return
  end if

  if ( lenwrk < 2 * n + 2 ) then
    ier = 3
    call xerfft ( 'DSINT1B', 10 )
    return
  end if

  call dsintb1 ( n, inc, x, wsave, work, work(n+2), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DSINT1B', -5 )
    return
  end if

  return
end
subroutine dsint1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! DSINT1F: real double precision forward sine transform, 1D.
!
!  Discussion:
!
!    DSINT1F computes the one-dimensional Fourier transform of an odd 
!    sequence within a real array.  This transform is referred to as the 
!    forward transform or Fourier analysis, transforming the sequence
!    from physical to spectral space.
!
!    This transform is normalized since a call to DSINT1F followed
!    by a call to DSINT1B (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 8 ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be
!    initialized with a call to DSINT1I before the first call to routine 
!    DSINT1F or DSINT1B for a given transform length N.  WSAVE's contents 
!    may be re-used for subsequent calls to DSINT1F and DSINT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N+2.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  real ( kind = 8 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'DSINT1F', 6 )
    return
  end if

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DSINT1F', 8 )
    return 
  end if

  if ( lenwrk < 2 * n + 2 ) then
    ier = 3
    call xerfft ( 'DSINT1F', 10 )
    return
  end if

  call dsintf1 ( n, inc, x, wsave, work, work(n+2), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DSINT1F', -5 )
    return
  end if

  return
end
subroutine dsint1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! DSINT1I: initialization for DSINT1B and DSINT1F.
!
!  Discussion:
!
!    DSINT1I initializes array WSAVE for use in its companion routines 
!    DSINT1F and DSINT1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    07 February 2006
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
!    transformed.  The transform is most efficient when N+1 is a product 
!    of small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 8 ) WSAVE(LENSAV), containing the prime factors
!    of N and also containing certain trigonometric values which will be used 
!    in routines DSINT1B or DSINT1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 8 ) dt
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) pi
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'DSINT1I', 3 )
    return
  end if

  pi = 4.0D+00 * atan ( 1.0D+00 )

  if ( n <= 1 ) then
    return
  end if

  ns2 = n / 2
  np1 = n + 1
  dt = pi / real ( np1, kind = 8 )

  do k = 1, ns2
    wsave(k) = 2.0D+00 * sin ( real ( k, kind = 8 ) * dt )
  end do

  lnsv = np1 + int ( log ( real ( np1, kind = 8  ) ) ) + 4

  call dfft1i ( np1, wsave(ns2+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DSINT1I', -5 )
    return
  end if

  return
end
subroutine dsintb1 ( n, inc, x, wsave, xh, work, ier )

!*****************************************************************************80
!
!! DSINTB1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  real ( kind = 8 ) fnp1s4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lnxh
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) srt3s2
  real ( kind = 8 ) t1
  real ( kind = 8 ) t2
  real ( kind = 8 ) work(*)
  real ( kind = 8 ) wsave(*)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) xh(*)
  real ( kind = 8 ) xhold
 
  ier = 0

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    srt3s2 = sqrt ( 3.0D+00 ) / 2.0D+00 
    xhold = srt3s2 * ( x(1,1) + x(1,2) )
    x(1,2) = srt3s2 * ( x(1,1) - x(1,2) )
    x(1,1) = xhold
    return
  end if

  np1 = n + 1
  ns2 = n / 2

  do k = 1, ns2
    kc = np1 - k
    t1 = x(1,k) - x(1,kc)
    t2 = wsave(k) * ( x(1,k) + x(1,kc) )
    xh(k+1) = t1 + t2
    xh(kc+1) = t2 - t1
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    xh(ns2+2) = 4.0D+00 * x(1,ns2+1)
  end if

  xh(1) = 0.0D+00
  lnxh = np1
  lnsv = np1 + int ( log ( real ( np1, kind = 8 ) ) ) + 4
  lnwk = np1

  call dfft1f ( np1, 1, xh, lnxh, wsave(ns2+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DSINTB1', -5 )
    return
  end if

  if ( mod ( np1, 2 ) == 0 ) then
    xh(np1) = xh(np1) + xh(np1)
  end if

  fnp1s4 = real ( np1, kind = 8 ) / 4.0D+00
  x(1,1) = fnp1s4 * xh(1)
  dsum = x(1,1)

  do i = 3, n, 2
    x(1,i-1) = fnp1s4 * xh(i)
    dsum = dsum + fnp1s4 * xh(i-1)
    x(1,i) = dsum
  end do

  if ( modn == 0 ) then
    x(1,n) = fnp1s4 * xh(n+1)
  end if

  return
end
subroutine dsintf1 ( n, inc, x, wsave, xh, work, ier )

!*****************************************************************************80
!
!! DSINTF1 is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 March 2009
!
!  Author:
!
!    Original real single precision by Paul Swarztrauber, Richard Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lnxh
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 8 ) sfnp1
  real ( kind = 8 ) ssqrt3
  real ( kind = 8 ) t1
  real ( kind = 8 ) t2
  real ( kind = 8 ) work(*)
  real ( kind = 8 ) wsave(*)
  real ( kind = 8 ) x(inc,*)
  real ( kind = 8 ) xh(*)
  real ( kind = 8 ) xhold

  ier = 0

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt3 = 1.0D+00 / sqrt ( 3.0D+00 )
    xhold = ssqrt3 * ( x(1,1) + x(1,2) )
    x(1,2) = ssqrt3 * ( x(1,1) - x(1,2) )
    x(1,1) = xhold
    return
  end if

  np1 = n + 1
  ns2 = n / 2

  do k = 1, ns2
    kc = np1 - k
    t1 = x(1,k) - x(1,kc)
    t2 = wsave(k) * ( x(1,k) + x(1,kc) )
    xh(k+1) = t1 + t2
    xh(kc+1) = t2 - t1
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    xh(ns2+2) = 4.0D+00 * x(1,ns2+1)
  end if

  xh(1) = 0.0D+00
  lnxh = np1
  lnsv = np1 + int ( log ( real ( np1, kind = 8 ) ) ) + 4
  lnwk = np1

  call dfft1f ( np1, 1, xh, lnxh, wsave(ns2+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'DSINTF1', -5 )
    return
  end if

  if ( mod ( np1, 2 ) == 0 ) then
    xh(np1) = xh(np1) + xh(np1)
  end if

  sfnp1 = 1.0D+00 / real ( np1, kind = 8 )
  x(1,1) = 0.5D+00 * xh(1)
  dsum = x(1,1)

  do i = 3, n, 2
    x(1,i-1) = 0.5D+00 * xh(i)
    dsum = dsum + 0.5D+00 * xh(i-1)
    x(1,i) = dsum
  end do

  if ( modn == 0 ) then
    x(1,n) = 0.5D+00 * xh(n+1)
  end if

  return
end
subroutine mcsqb1 ( lot, jump, n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! MCSQB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lot

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np2
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(lot,*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xim1

  ier = 0
  lj = ( lot - 1 ) * jump + 1
  ns2 = ( n + 1 ) / 2
  np2 = n + 2

  do i = 3, n, 2
    do m = 1, lj, jump
      xim1 = x(m,i-1) + x(m,i)
      x(m,i) = 0.5E+00 * ( x(m,i-1) - x(m,i) )
      x(m,i-1) = 0.5E+00 * xim1
    end do
  end do

  do m = 1, lj, jump
    x(m,1) = 0.5E+00 * x(m,1)
  end do

  modn = mod ( n, 2 )
  if ( modn == 0 ) then
    do m = 1, lj, jump
      x(m,n) = 0.5E+00 * x(m,n)
    end do
  end if

  lenx = ( lot - 1 ) * jump + inc * ( n - 1 )  + 1
  lnsv = n + int ( log ( real ( n, kind = 4 ) ) ) + 4
  lnwk = lot * n

  call rfftmb ( lot, jump, n, inc, x, lenx, wsave(n+1), lnsv, &
    work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'mcsqb1', -5 )
    return
  end if

  do k = 2, ns2
    kc = np2 - k
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      work(m1,k) = wsave(k-1) * x(m,kc) + wsave(kc-1) * x(m,k)
      work(m1,kc) = wsave(k-1) * x(m,k) - wsave(kc-1) * x(m,kc)
    end do
  end do

  if ( modn == 0 ) then
    do m = 1, lj, jump
      x(m,ns2+1) = wsave(ns2) * ( x(m,ns2+1) + x(m,ns2+1) )
    end do
  end if

  do k = 2, ns2
    kc = np2 - k
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      x(m,k) = work(m1,k) + work(m1,kc)
      x(m,kc) = work(m1,k) - work(m1,kc)
    end do
  end do

  do m = 1, lj, jump
    x(m,1) = x(m,1) + x(m,1)
  end do

  return
end
subroutine mcsqf1 ( lot, jump, n, inc, x, wsave, work, ier )

!*****************************************************************************80
!
!! MCSQF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lot

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np2
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(lot,*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xim1

  ier = 0
  lj = ( lot - 1 ) * jump + 1
  ns2 = ( n + 1 ) / 2
  np2 = n + 2

  do k = 2, ns2
    kc = np2 - k
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      work(m1,k)  = x(m,k) + x(m,kc)
      work(m1,kc) = x(m,k) - x(m,kc)
    end do
  end do

  modn = mod ( n, 2 )

  if ( modn == 0 ) then
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      work(m1,ns2+1) = x(m,ns2+1) + x(m,ns2+1)
    end do
  end if

  do k = 2, ns2
    kc = np2 - k
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      x(m,k)  = wsave(k-1) * work(m1,kc) + wsave(kc-1) * work(m1,k)
      x(m,kc) = wsave(k-1) * work(m1,k)  - wsave(kc-1) * work(m1,kc)
    end do
  end do

  if ( modn == 0 ) then
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      x(m,ns2+1) = wsave(ns2) * work(m1,ns2+1)
    end do
  end if

  lenx = ( lot - 1 ) * jump + inc * ( n - 1 ) + 1
  lnsv = n + int ( log ( real ( n, kind = 4 ) ) ) + 4
  lnwk = lot * n

  call rfftmf ( lot, jump, n, inc, x, lenx, wsave(n+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'mcsqf1', -5 )
    return
  end if

  do i = 3, n, 2
    do m = 1, lj, jump
      xim1 = 0.5E+00 * ( x(m,i-1) + x(m,i) )
      x(m,i) = 0.5E+00 * ( x(m,i-1) - x(m,i) )
      x(m,i-1) = xim1
    end do
  end do

  return
end
subroutine mcstb1 ( lot, jump, n, inc, x, wsave, dsum, work, ier )

!*****************************************************************************80
!
!! MCSTB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum(*)
  real ( kind = 4 ) fnm1s2
  real ( kind = 4 ) fnm1s4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) x1h
  real ( kind = 4 ) x1p3
  real ( kind = 4 ) x2
  real ( kind = 4 ) xi

  ier = 0
  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2
  lj = ( lot - 1 ) * jump + 1

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    do m = 1, lj, jump
      x1h = x(m,1) + x(m,2)
      x(m,2) = x(m,1) - x(m,2)
      x(m,1) = x1h
    end do
    return
  end if

  if ( n == 3 ) then

    do m = 1, lj, jump
      x1p3 = x(m,1) + x(m,3)
      x2 = x(m,2)
      x(m,2) = x(m,1) - x(m,3)
      x(m,1) = x1p3 + x2
      x(m,3) = x1p3 - x2
    end do

    return
  end if

  do m = 1, lj, jump
    x(m,1) = x(m,1) + x(m,1)
    x(m,n) = x(m,n) + x(m,n)
  end do

  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    dsum(m1) = x(m,1) - x(m,n)
    x(m,1) = x(m,1) + x(m,n)
  end do

  do k = 2, ns2
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      kc = np1 - k
      t1 = x(m,k) + x(m,kc)
      t2 = x(m,k) - x(m,kc)
      dsum(m1) = dsum(m1) + wsave(kc) * t2
      t2 = wsave(k) * t2
      x(m,k) = t1 - t2
      x(m,kc) = t1 + t2
    end do
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    do m = 1, lj, jump
      x(m,ns2+1) = x(m,ns2+1) + x(m,ns2+1)
    end do
  end if

  lenx = ( lot - 1 ) * jump + inc * ( nm1 - 1 )  + 1
  lnsv = nm1 + int ( log ( real ( nm1, kind = 4 ) ) ) + 4
  lnwk = lot * nm1

  call rfftmf ( lot, jump, nm1, inc, x, lenx, wsave(n+1), lnsv, work, &
    lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'mcstb1', -5 )
    return
  end if

  fnm1s2 = real ( nm1, kind = 4 ) / 2.0E+00
  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    dsum(m1) = 0.5E+00 * dsum(m1)
    x(m,1) = fnm1s2 * x(m,1)
  end do

  if ( mod ( nm1, 2 ) == 0 ) then
    do m = 1, lj, jump
      x(m,nm1) = x(m,nm1) + x(m,nm1)
    end do
  end if

  fnm1s4 = real ( nm1, kind = 4 ) / 4.0E+00

  do i = 3, n, 2
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      xi = fnm1s4 * x(m,i)
      x(m,i) = fnm1s4 * x(m,i-1)
      x(m,i-1) = dsum(m1)
      dsum(m1) = dsum(m1) + xi
    end do
  end do

  if ( modn /= 0 ) then
    return
  end if

  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    x(m,n) = dsum(m1)
  end do

  return
end
subroutine mcstf1 ( lot, jump, n, inc, x, wsave, dsum, work, ier )

!*****************************************************************************80
!
!! MCSTF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum(*)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nm1
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) snm1
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) tx2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) x1h
  real ( kind = 4 ) x1p3
  real ( kind = 4 ) xi

  ier = 0
  nm1 = n - 1
  np1 = n + 1
  ns2 = n / 2
  lj = ( lot - 1 ) * jump + 1

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then

    do m = 1, lj, jump
      x1h = x(m,1) + x(m,2)
      x(m,2) = 0.5E+00 * ( x(m,1) - x(m,2) )
      x(m,1) = 0.5E+00 * x1h
    end do

    return

  end if

  if ( n == 3 ) then

    do m = 1, lj, jump
      x1p3 = x(m,1) + x(m,3)
      tx2 = x(m,2) + x(m,2)
      x(m,2) = 0.5E+00 * ( x(m,1) - x(m,3) )
      x(m,1) = 0.25E+00 * ( x1p3 + tx2 )
      x(m,3) = 0.25E+00 * ( x1p3 - tx2 )
    end do

    return
  end if

  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    dsum(m1) = x(m,1) - x(m,n)
    x(m,1) = x(m,1) + x(m,n)
  end do

  do k = 2, ns2
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      kc = np1 - k
      t1 = x(m,k) + x(m,kc)
      t2 = x(m,k) - x(m,kc)
      dsum(m1) = dsum(m1) + wsave(kc) * t2
      t2 = wsave(k) * t2
      x(m,k) = t1 - t2
      x(m,kc) = t1 + t2
    end do
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    do m = 1, lj, jump
      x(m,ns2+1) = x(m,ns2+1) + x(m,ns2+1)
    end do
  end if

  lenx = ( lot - 1 ) * jump + inc * ( nm1 - 1 )  + 1
  lnsv = nm1 + int ( log ( real ( nm1, kind = 4 ) ) ) + 4
  lnwk = lot * nm1

  call rfftmf ( lot, jump, nm1, inc, x, lenx, wsave(n+1), lnsv, work, &
    lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'mcstf1', -5 )
    return
  end if

  snm1 = 1.0E+00 / real ( nm1, kind = 4 )
  do m = 1, lot
    dsum(m) = snm1 * dsum(m)
  end do

  if ( mod ( nm1, 2 ) == 0 ) then
    do m = 1, lj, jump
      x(m,nm1) = x(m,nm1) + x(m,nm1)
    end do
  end if

  do i = 3, n, 2
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      xi = 0.5E+00 * x(m,i)
      x(m,i) = 0.5E+00 * x(m,i-1)
      x(m,i-1) = dsum(m1)
      dsum(m1) = dsum(m1) + xi
    end do
  end do

  if ( modn == 0 ) then

    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      x(m,n) = dsum(m1)
    end do

  end if

  do m = 1, lj, jump
    x(m,1) = 0.5E+00 * x(m,1)
    x(m,n) = 0.5E+00 * x(m,n)
  end do

  return
end
subroutine mradb2 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1 )

!*****************************************************************************80
!
!! MRADB2 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,2,l1)
  real ( kind = 4 ) ch(in2,ido,l1,2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) wa1(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,k,1) = cc(m1,1,1,k) + cc(m1,ido,2,k)
      ch(m2,1,k,2) = cc(m1,1,1,k) - cc(m1,ido,2,k)
    end do
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,i-1,k,1) = cc(m1,i-1,1,k) + cc(m1,ic-1,2,k)
          ch(m2,i,k,1)   = cc(m1,i,1,k) - cc(m1,ic,2,k)
          ch(m2,i-1,k,2) = wa1(i-2) * ( cc(m1,i-1,1,k) - cc(m1,ic-1,2,k) ) &
                         - wa1(i-1) * ( cc(m1,i,1,k)   + cc(m1,ic,2,k) )
          ch(m2,i,k,2)   = wa1(i-2) * ( cc(m1,i,1,k)   + cc(m1,ic,2,k) ) &
                         + wa1(i-1) * ( cc(m1,i-1,1,k) - cc(m1,ic-1,2,k) )
        end do
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,ido,k,1) = cc(m1,ido,1,k) + cc(m1,ido,1,k)
      ch(m2,ido,k,2) = -( cc(m1,1,2,k) + cc(m1,1,2,k) )
    end do
  end do

  return
end
subroutine mradb3 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1, wa2 )

!*****************************************************************************80
!
!! MRADB3 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,3,l1)
  real ( kind = 4 ) ch(in2,ido,l1,3)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) taui
  real ( kind = 4 ) taur
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 3.0E+00
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,k,1) = cc(m1,1,1,k) + 2.0E+00 * cc(m1,ido,2,k)
      ch(m2,1,k,2) = cc(m1,1,1,k) + ( 2.0E+00 * taur ) * cc(m1,ido,2,k) &
        - ( 2.0E+00 * taui ) * cc(m1,1,3,k)
      ch(m2,1,k,3) = cc(m1,1,1,k) + ( 2.0E+00 * taur ) * cc(m1,ido,2,k) &
        + 2.0E+00 * taui * cc(m1,1,3,k)
    end do
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      m2 = m2s
      do m1 = 1, m1d, im1

        m2 = m2 + im2

        ch(m2,i-1,k,1) = cc(m1,i-1,1,k)+(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))

        ch(m2,i,k,1) = cc(m1,i,1,k)+(cc(m1,i,3,k)-cc(m1,ic,2,k))

        ch(m2,i-1,k,2) = wa1(i-2)* &
          ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))- &
          (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k)))) - wa1(i-1)* &
          ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))+ &
          (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))))

        ch(m2,i,k,2) = wa1(i-2)* &
          ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))+ &
          (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))) + wa1(i-1)* &
          ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))- &
          (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k))))

        ch(m2,i-1,k,3) = wa2(i-2)* &
          ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))+ &
          (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k)))) - wa2(i-1)* &
          ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))- &
          (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))))

        ch(m2,i,k,3) = wa2(i-2)* &
          ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))- &
          (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))) + wa2(i-1)* &
          ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))+ &
          (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k))))

      end do
    end do
  end do

  return
end
subroutine mradb4 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1, wa2, wa3 )

!*****************************************************************************80
!
!! MRADB4 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,4,l1)
  real ( kind = 4 ) ch(in2,ido,l1,4)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) sqrt2
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  sqrt2 = sqrt ( 2.0E+00 )

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,k,3) = (cc(m1,1,1,k)+cc(m1,ido,4,k)) &
        -(cc(m1,ido,2,k)+cc(m1,ido,2,k))
      ch(m2,1,k,1) = (cc(m1,1,1,k)+cc(m1,ido,4,k)) &
        +(cc(m1,ido,2,k)+cc(m1,ido,2,k))
      ch(m2,1,k,4) = (cc(m1,1,1,k)-cc(m1,ido,4,k)) &
        +(cc(m1,1,3,k)+cc(m1,1,3,k))
      ch(m2,1,k,2) = (cc(m1,1,1,k)-cc(m1,ido,4,k)) &
        -(cc(m1,1,3,k)+cc(m1,1,3,k))
    end do
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,i-1,k,1) = (cc(m1,i-1,1,k)+cc(m1,ic-1,4,k)) &
            +(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))
          ch(m2,i,k,1) = (cc(m1,i,1,k)-cc(m1,ic,4,k)) &
            +(cc(m1,i,3,k)-cc(m1,ic,2,k))
          ch(m2,i-1,k,2) = wa1(i-2)*((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k)) &
            -(cc(m1,i,3,k)+cc(m1,ic,2,k)))-wa1(i-1) &
            *((cc(m1,i,1,k)+cc(m1,ic,4,k))+(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))
          ch(m2,i,k,2) = wa1(i-2)*((cc(m1,i,1,k)+cc(m1,ic,4,k)) &
            +(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))) + wa1(i-1) &
            *((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k))-(cc(m1,i,3,k)+cc(m1,ic,2,k)))
          ch(m2,i-1,k,3) = wa2(i-2)*((cc(m1,i-1,1,k)+cc(m1,ic-1,4,k)) &
            -(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))) - wa2(i-1) &
            *((cc(m1,i,1,k)-cc(m1,ic,4,k))-(cc(m1,i,3,k)-cc(m1,ic,2,k)))
          ch(m2,i,k,3) = wa2(i-2)*((cc(m1,i,1,k)-cc(m1,ic,4,k)) &
            -(cc(m1,i,3,k)-cc(m1,ic,2,k))) + wa2(i-1) &
            *((cc(m1,i-1,1,k)+cc(m1,ic-1,4,k))-(cc(m1,i-1,3,k) &
            +cc(m1,ic-1,2,k)))
          ch(m2,i-1,k,4) = wa3(i-2)*((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k)) &
            +(cc(m1,i,3,k)+cc(m1,ic,2,k))) - wa3(i-1) &
            *((cc(m1,i,1,k)+cc(m1,ic,4,k))-(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))
          ch(m2,i,k,4) = wa3(i-2)*((cc(m1,i,1,k)+cc(m1,ic,4,k)) &
            -(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))) + wa3(i-1) &
            *((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k))+(cc(m1,i,3,k)+cc(m1,ic,2,k)))
        end do
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,ido,k,1) = (cc(m1,ido,1,k)+cc(m1,ido,3,k)) &
        +(cc(m1,ido,1,k)+cc(m1,ido,3,k))
      ch(m2,ido,k,2) = sqrt2*((cc(m1,ido,1,k)-cc(m1,ido,3,k)) &
        -(cc(m1,1,2,k)+cc(m1,1,4,k)))
      ch(m2,ido,k,3) = (cc(m1,1,4,k)-cc(m1,1,2,k)) &
        +(cc(m1,1,4,k)-cc(m1,1,2,k))
      ch(m2,ido,k,4) = -sqrt2*((cc(m1,ido,1,k)-cc(m1,ido,3,k)) &
        +(cc(m1,1,2,k)+cc(m1,1,4,k)))
    end do
  end do

  return
end
subroutine mradb5 ( m, ido, l1, cc, im1, in1, ch, im2, in2, &
  wa1, wa2, wa3, wa4 )

!*****************************************************************************80
!
!! MRADB5 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,5,l1)
  real ( kind = 4 ) ch(in2,ido,l1,5)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) ti11
  real ( kind = 4 ) ti12
  real ( kind = 4 ) tr11
  real ( kind = 4 ) tr12
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)
  real ( kind = 4 ) wa4(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 5.E+00
  tr11 = cos ( arg )
  ti11 = sin ( arg )
  tr12 = cos ( 2.0E+00 * arg )
  ti12 = sin ( 2.0E+00 * arg )

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,k,1) = cc(m1,1,1,k) + 2.0E+00 * cc(m1,ido,2,k) &
        + 2.0E+00 * cc(m1,ido,4,k)
      ch(m2,1,k,2) = ( cc(m1,1,1,k) + tr11 * 2.0E+00 * cc(m1,ido,2,k) &
        + tr12 * 2.0E+00 * cc(m1,ido,4,k) ) - ( ti11 * 2.0E+00 * cc(m1,1,3,k) &
        + ti12 * 2.0E+00 * cc(m1,1,5,k) )
      ch(m2,1,k,3) = ( cc(m1,1,1,k) + tr12 * 2.0E+00 * cc(m1,ido,2,k) &
        + tr11 * 2.0E+00 * cc(m1,ido,4,k) ) - ( ti12 * 2.0E+00 * cc(m1,1,3,k) &
        - ti11 * 2.0E+00 * cc(m1,1,5,k) )
      ch(m2,1,k,4) = ( cc(m1,1,1,k) + tr12 * 2.0E+00 * cc(m1,ido,2,k) &
        + tr11 * 2.0E+00 * cc(m1,ido,4,k) ) + ( ti12 * 2.0E+00 * cc(m1,1,3,k) &
        - ti11 * 2.0E+00 * cc(m1,1,5,k) )
      ch(m2,1,k,5) = ( cc(m1,1,1,k) + tr11 * 2.0E+00 * cc(m1,ido,2,k) &
        + tr12 * 2.0E+00 * cc(m1,ido,4,k) ) + ( ti11 * 2.0E+00 * cc(m1,1,3,k) &
        + ti12 * 2.0E+00 * cc(m1,1,5,k) )
    end do
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2
  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(m2,i-1,k,1) = cc(m1,i-1,1,k)+(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k))
        ch(m2,i,k,1) = cc(m1,i,1,k)+(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
          +(cc(m1,i,5,k)-cc(m1,ic,4,k))
        ch(m2,i-1,k,2) = wa1(i-2)*((cc(m1,i-1,1,k)+tr11* &
          (cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))+tr12 &
          *(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))-(ti11*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))+ti12*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
          -wa1(i-1)*((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
          +tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k)))+(ti11*(cc(m1,i-1,3,k) &
          -cc(m1,ic-1,2,k))+ti12*(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))
        ch(m2,i,k,2) = wa1(i-2)*((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k) &
          -cc(m1,ic,2,k))+tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
          +(ti11*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))+ti12 &
          *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) + wa1(i-1) &
          *((cc(m1,i-1,1,k)+tr11*(cc(m1,i-1,3,k) &
          +cc(m1,ic-1,2,k))+tr12*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k))) &
          -(ti11*(cc(m1,i,3,k)+cc(m1,ic,2,k))+ti12 &
          *(cc(m1,i,5,k)+cc(m1,ic,4,k))))
        ch(m2,i-1,k,3) = wa2(i-2) &
          *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))-(ti12*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
          -wa2(i-1) &
          *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
        cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
          +(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
          *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))
        ch(m2,i,k,3) = wa2(i-2) &
          *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
          cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
          +(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
          *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) &
          + wa2(i-1) &
          *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))-(ti12*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k))))
        ch(m2,i-1,k,4) = wa3(i-2) &
          *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti12*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
          -wa3(i-1) &
          *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
          cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
          -(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
          *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))
        ch(m2,i,k,4) = wa3(i-2) &
          *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
          cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
          -(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
          *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) &
          + wa3(i-1) &
          *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti12*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k))))
        ch(m2,i-1,k,5) = wa4(i-2) &
          *((cc(m1,i-1,1,k)+tr11*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +tr12*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti11*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))+ti12*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
          -wa4(i-1) &
          *((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
          +tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k)))-(ti11*(cc(m1,i-1,3,k) &
          -cc(m1,ic-1,2,k))+ti12*(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))
        ch(m2,i,k,5) = wa4(i-2) &
          *((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
          +tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k)))-(ti11*(cc(m1,i-1,3,k) &
          -cc(m1,ic-1,2,k))+ti12*(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) &
          + wa4(i-1) &
          *((cc(m1,i-1,1,k)+tr11*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
          +tr12*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti11*(cc(m1,i,3,k) &
          +cc(m1,ic,2,k))+ti12*(cc(m1,i,5,k)+cc(m1,ic,4,k))))
      end do
    end do
  end do

  return
end
subroutine mradbg ( m, ido, ip, l1, idl1, cc, c1, c2, im1, in1, &
  ch, ch2, im2, in2, wa )

!*****************************************************************************80
!
!! MRADBG is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1

  real ( kind = 4 ) ai1
  real ( kind = 4 ) ai2
  real ( kind = 4 ) ar1
  real ( kind = 4 ) ar1h
  real ( kind = 4 ) ar2
  real ( kind = 4 ) ar2h
  real ( kind = 4 ) arg
  real ( kind = 4 ) c1(in1,ido,l1,ip)
  real ( kind = 4 ) c2(in1,idl1,ip)
  real ( kind = 4 ) cc(in1,ido,ip,l1)
  real ( kind = 4 ) ch(in2,ido,l1,ip)
  real ( kind = 4 ) ch2(in2,idl1,ip)
  real ( kind = 4 ) dc2
  real ( kind = 4 ) dcp
  real ( kind = 4 ) ds2
  real ( kind = 4 ) dsp
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idij
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) ik
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) j2
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) nbd
  real ( kind = 4 ) tpi
  real ( kind = 4 ) wa(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  tpi = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 )
  arg = tpi / real ( ip, kind = 4 )
  dcp = cos ( arg )
  dsp = sin ( arg )
  idp2 = ido + 2
  nbd = ( ido - 1 ) / 2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  if ( ido < l1 ) then

    do i = 1, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,i,k,1) = cc(m1,i,1,k)
        end do
      end do
    end do

  else

    do k = 1, l1
      do i = 1, ido
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,i,k,1) = cc(m1,i,1,k)
        end do
      end do
    end do

  end if

  do j = 2, ipph
    jc = ipp2 - j
    j2 = j + j
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(m2,1,k,j) = cc(m1,ido,j2-2,k) + cc(m1,ido,j2-2,k)
        ch(m2,1,k,jc) = cc(m1,1,j2-1,k) + cc(m1,1,j2-1,k)
      end do
    end do
  end do

  if ( ido == 1 ) then

  else if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      do i = 3, ido, 2
        ic = idp2 - i
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            ch(m2,i-1,k,j)  = cc(m1,i-1,2*j-1,k) + cc(m1,ic-1,2*j-2,k)
            ch(m2,i-1,k,jc) = cc(m1,i-1,2*j-1,k) - cc(m1,ic-1,2*j-2,k)
            ch(m2,i,k,j)    = cc(m1,i,2*j-1,k)   - cc(m1,ic,2*j-2,k)
            ch(m2,i,k,jc)   = cc(m1,i,2*j-1,k)   + cc(m1,ic,2*j-2,k)
          end do
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do k = 1, l1
        do i = 3, ido, 2
          ic = idp2 - i
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            ch(m2,i-1,k,j)  = cc(m1,i-1,2*j-1,k) + cc(m1,ic-1,2*j-2,k)
            ch(m2,i-1,k,jc) = cc(m1,i-1,2*j-1,k) - cc(m1,ic-1,2*j-2,k)
            ch(m2,i,k,j)    = cc(m1,i,2*j-1,k)   - cc(m1,ic,2*j-2,k)
            ch(m2,i,k,jc)   = cc(m1,i,2*j-1,k)   + cc(m1,ic,2*j-2,k)
          end do
        end do
      end do
    end do

  end if

  ar1 = 1.0E+00
  ai1 = 0.0E+00

  do l = 2, ipph

    lc = ipp2 - l
    ar1h = dcp * ar1 - dsp * ai1
    ai1 =  dcp * ai1 + dsp * ar1
    ar1 = ar1h

    do ik = 1, idl1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        c2(m1,ik,l)  = ch2(m2,ik,1) + ar1 * ch2(m2,ik,2)
        c2(m1,ik,lc) =                ai1 * ch2(m2,ik,ip)
      end do
    end do

    dc2 = ar1
    ds2 = ai1
    ar2 = ar1
    ai2 = ai1

    do j = 3, ipph
      jc = ipp2 - j
      ar2h = dc2 * ar2 - ds2 * ai2
      ai2  = dc2 * ai2 + ds2 * ar2
      ar2 = ar2h
      do ik = 1, idl1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          c2(m1,ik,l)  = c2(m1,ik,l)  + ar2 * ch2(m2,ik,j)
          c2(m1,ik,lc) = c2(m1,ik,lc) + ai2 * ch2(m2,ik,jc)
        end do
      end do
    end do

  end do

  do j = 2, ipph
    do ik = 1, idl1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch2(m2,ik,1) = ch2(m2,ik,1) + ch2(m2,ik,j)
      end do
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(m2,1,k,j)  = c1(m1,1,k,j) - c1(m1,1,k,jc)
        ch(m2,1,k,jc) = c1(m1,1,k,j) + c1(m1,1,k,jc)
      end do
    end do
  end do

  if ( ido == 1 ) then

  else if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      do i = 3, ido, 2
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            ch(m2,i-1,k,j)  = c1(m1,i-1,k,j) - c1(m1,i,k,jc)
            ch(m2,i-1,k,jc) = c1(m1,i-1,k,j) + c1(m1,i,k,jc)
            ch(m2,i,k,j)    = c1(m1,i,k,j)   + c1(m1,i-1,k,jc)
            ch(m2,i,k,jc)   = c1(m1,i,k,j)   - c1(m1,i-1,k,jc)
          end do
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do k = 1, l1
        do i = 3, ido, 2
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            ch(m2,i-1,k,j)  = c1(m1,i-1,k,j) - c1(m1,i,k,jc)
            ch(m2,i-1,k,jc) = c1(m1,i-1,k,j) + c1(m1,i,k,jc)
            ch(m2,i,k,j)    = c1(m1,i,k,j)   + c1(m1,i-1,k,jc)
            ch(m2,i,k,jc)   = c1(m1,i,k,j)   - c1(m1,i-1,k,jc)
          end do
        end do
      end do
    end do

  end if

  if ( ido == 1 ) then
    return
  end if

  do ik = 1, idl1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      c2(m1,ik,1) = ch2(m2,ik,1)
    end do
  end do

  do j = 2, ip
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        c1(m1,1,k,j) = ch(m2,1,k,j)
      end do
    end do
  end do

  if ( l1 < nbd ) then

    is = -ido

    do j = 2, ip
      is = is + ido
      do k = 1, l1
        idij = is
        do i = 3, ido, 2
          idij = idij + 2
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            c1(m1,i-1,k,j) = wa(idij-1) * ch(m2,i-1,k,j) &
                           - wa(idij)   * ch(m2,i,k,j)
            c1(m1,i,k,j) =   wa(idij-1) * ch(m2,i,k,j) &
                           + wa(idij)   * ch(m2,i-1,k,j)
          end do
        end do
      end do
    end do

  else

    is = -ido

    do j = 2, ip
      is = is + ido
      idij = is
      do i = 3, ido, 2
        idij = idij + 2
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            c1(m1,i-1,k,j) = wa(idij-1) * ch(m2,i-1,k,j) &
                           - wa(idij)   * ch(m2,i,k,j)
            c1(m1,i,k,j) =   wa(idij-1) * ch(m2,i,k,j) &
                           + wa(idij)   * ch(m2,i-1,k,j)
          end do
        end do
      end do
    end do

  end if

  return
end
subroutine mradf2 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1 )

!*****************************************************************************80
!
!! MRADF2 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,l1,2)
  real ( kind = 4 ) ch(in2,ido,2,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) wa1(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,1,k)   = cc(m1,1,k,1) + cc(m1,1,k,2)
      ch(m2,ido,2,k) = cc(m1,1,k,1) - cc(m1,1,k,2)
    end do
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,i,1,k) =    cc(m1,i,k,1)   + ( wa1(i-2) * cc(m1,i,k,2) &
                                             - wa1(i-1) * cc(m1,i-1,k,2) )
          ch(m2,ic,2,k)  = -cc(m1,i,k,1)   + ( wa1(i-2) * cc(m1,i,k,2) &
                                             - wa1(i-1) * cc(m1,i-1,k,2) ) 
          ch(m2,i-1,1,k)  = cc(m1,i-1,k,1) + ( wa1(i-2) * cc(m1,i-1,k,2) &
                                             + wa1(i-1) * cc(m1,i,k,2))
          ch(m2,ic-1,2,k) = cc(m1,i-1,k,1) - ( wa1(i-2) * cc(m1,i-1,k,2) &
                                             + wa1(i-1) * cc(m1,i,k,2) )
        end do
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,2,k) = -cc(m1,ido,k,2)
      ch(m2,ido,1,k) = cc(m1,ido,k,1)
    end do
  end do

  return
end
subroutine mradf3 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1, wa2 )

!*****************************************************************************80
!
!! MRADF3 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,l1,3)
  real ( kind = 4 ) ch(in2,ido,3,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) taui
  real ( kind = 4 ) taur
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 3.0E+00
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,1,k)   = cc(m1,1,k,1)        + ( cc(m1,1,k,2) + cc(m1,1,k,3) )
      ch(m2,1,3,k)   =                taui * ( cc(m1,1,k,3) - cc(m1,1,k,2) )
      ch(m2,ido,2,k) = cc(m1,1,k,1) + taur * ( cc(m1,1,k,2) + cc(m1,1,k,3) )
     end do
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(m2,i-1,1,k) = cc(m1,i-1,k,1)+((wa1(i-2)*cc(m1,i-1,k,2)+ &
          wa1(i-1)*cc(m1,i,k,2))+(wa2(i-2)*cc(m1,i-1,k,3) + wa2(i-1)* &
          cc(m1,i,k,3)))
        ch(m2,i,1,k) = cc(m1,i,k,1)+((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3)))
        ch(m2,i-1,3,k) = (cc(m1,i-1,k,1)+taur*((wa1(i-2)* &
          cc(m1,i-1,k,2) + wa1(i-1)*cc(m1,i,k,2))+(wa2(i-2)* &
          cc(m1,i-1,k,3) + wa2(i-1)*cc(m1,i,k,3))))+(taui*((wa1(i-2)* &
          cc(m1,i,k,2) - wa1(i-1)*cc(m1,i-1,k,2))-(wa2(i-2)* &
          cc(m1,i,k,3) - wa2(i-1)*cc(m1,i-1,k,3))))
        ch(m2,ic-1,2,k) = (cc(m1,i-1,k,1)+taur*((wa1(i-2)* &
          cc(m1,i-1,k,2) + wa1(i-1)*cc(m1,i,k,2))+(wa2(i-2)* &
          cc(m1,i-1,k,3) + wa2(i-1)*cc(m1,i,k,3))))-(taui*((wa1(i-2)* &
          cc(m1,i,k,2) - wa1(i-1)*cc(m1,i-1,k,2)) - ( wa2(i-2)* &
          cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))))
        ch(m2,i,3,k) = (cc(m1,i,k,1)+taur*((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa2(i-2)*cc(m1,i,k,3) - wa2(i-1)* &
          cc(m1,i-1,k,3))))+(taui*((wa2(i-2)*cc(m1,i-1,k,3) + wa2(i-1)* &
          cc(m1,i,k,3))-(wa1(i-2)*cc(m1,i-1,k,2) + wa1(i-1)* &
          cc(m1,i,k,2))))
        ch(m2,ic,2,k) = (taui*((wa2(i-2)*cc(m1,i-1,k,3) + wa2(i-1)* &
          cc(m1,i,k,3))-(wa1(i-2)*cc(m1,i-1,k,2) + wa1(i-1)* &
          cc(m1,i,k,2))))-(cc(m1,i,k,1)+taur*((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3))))
      end do
    end do
  end do

  return
end
subroutine mradf4 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1, wa2, wa3 )

!*****************************************************************************80
!
!! MRADF4 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,l1,4)
  real ( kind = 4 ) ch(in2,ido,4,l1)
  real ( kind = 4 ) hsqt2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)

  hsqt2 = sqrt ( 2.0E+00 ) / 2.0E+00 
  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,1,k) =   ( cc(m1,1,k,2) + cc(m1,1,k,4) ) &
                     + ( cc(m1,1,k,1) + cc(m1,1,k,3) )
      ch(m2,ido,4,k) = ( cc(m1,1,k,1) + cc(m1,1,k,3) ) &
                      -( cc(m1,1,k,2) + cc(m1,1,k,4) )
      ch(m2,ido,2,k) =   cc(m1,1,k,1) - cc(m1,1,k,3)
      ch(m2,1,3,k) =     cc(m1,1,k,4) - cc(m1,1,k,2)
    end do
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,i-1,1,k) = ((wa1(i-2)*cc(m1,i-1,k,2) + wa1(i-1)* &
            cc(m1,i,k,2))+(wa3(i-2)*cc(m1,i-1,k,4) + wa3(i-1)* &
            cc(m1,i,k,4)))+(cc(m1,i-1,k,1)+(wa2(i-2)*cc(m1,i-1,k,3)+ &
            wa2(i-1)*cc(m1,i,k,3)))
          ch(m2,ic-1,4,k) = (cc(m1,i-1,k,1)+(wa2(i-2)*cc(m1,i-1,k,3)+ &
            wa2(i-1)*cc(m1,i,k,3)))-((wa1(i-2)*cc(m1,i-1,k,2)+ &
            wa1(i-1)*cc(m1,i,k,2))+(wa3(i-2)*cc(m1,i-1,k,4)+ &
            wa3(i-1)*cc(m1,i,k,4)))
          ch(m2,i,1,k) = ((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
            cc(m1,i-1,k,2))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
            cc(m1,i-1,k,4)))+(cc(m1,i,k,1)+(wa2(i-2)*cc(m1,i,k,3)- &
            wa2(i-1)*cc(m1,i-1,k,3)))
          ch(m2,ic,4,k) = ((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
            cc(m1,i-1,k,2))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
            cc(m1,i-1,k,4)))-(cc(m1,i,k,1)+(wa2(i-2)*cc(m1,i,k,3)- &
            wa2(i-1)*cc(m1,i-1,k,3)))
          ch(m2,i-1,3,k) = ((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
            cc(m1,i-1,k,2))-(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
            cc(m1,i-1,k,4)))+(cc(m1,i-1,k,1)-(wa2(i-2)*cc(m1,i-1,k,3)+ &
            wa2(i-1)*cc(m1,i,k,3)))
          ch(m2,ic-1,2,k) = (cc(m1,i-1,k,1)-(wa2(i-2)*cc(m1,i-1,k,3)+ &
            wa2(i-1)*cc(m1,i,k,3)))-((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
            cc(m1,i-1,k,2))-(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
            cc(m1,i-1,k,4)))
          ch(m2,i,3,k) = ((wa3(i-2)*cc(m1,i-1,k,4) + wa3(i-1)* &
            cc(m1,i,k,4))-(wa1(i-2)*cc(m1,i-1,k,2) + wa1(i-1)* &
            cc(m1,i,k,2)))+(cc(m1,i,k,1)-(wa2(i-2)*cc(m1,i,k,3)- &
            wa2(i-1)*cc(m1,i-1,k,3)))
          ch(m2,ic,2,k) = ((wa3(i-2)*cc(m1,i-1,k,4) + wa3(i-1)* &
            cc(m1,i,k,4))-(wa1(i-2)*cc(m1,i-1,k,2) + wa1(i-1)* &
            cc(m1,i,k,2)))-(cc(m1,i,k,1)-(wa2(i-2)*cc(m1,i,k,3)- &
            wa2(i-1)*cc(m1,i-1,k,3)))
        end do
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,ido,1,k) = cc(m1,ido,k,1) &
        + (  hsqt2 * ( cc(m1,ido,k,2) - cc(m1,ido,k,4) ) ) 
      ch(m2,ido,3,k) = cc(m1,ido,k,1) &
        - (  hsqt2 * ( cc(m1,ido,k,2) - cc(m1,ido,k,4) ) )
      ch(m2,1,2,k) =  -cc(m1,ido,k,3) &
        + ( -hsqt2 * ( cc(m1,ido,k,2) + cc(m1,ido,k,4) ) ) 
      ch(m2,1,4,k) =   cc(m1,ido,k,3) &
        + ( -hsqt2 * ( cc(m1,ido,k,2) + cc(m1,ido,k,4) ) ) 
    end do
  end do

  return
end
subroutine mradf5 ( m, ido, l1, cc, im1, in1, ch, im2, in2, wa1, wa2, wa3, wa4 )

!*****************************************************************************80
!
!! MRADF5 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,l1,5)
  real ( kind = 4 ) ch(in2,ido,5,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  real ( kind = 4 ) ti11
  real ( kind = 4 ) ti12
  real ( kind = 4 ) tr11
  real ( kind = 4 ) tr12
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)
  real ( kind = 4 ) wa4(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 5.0E+00
  tr11 = cos ( arg )
  ti11 = sin ( arg )
  tr12 = cos ( 2.0E+00 * arg )
  ti12 = sin ( 2.0E+00 * arg )

  do k = 1, l1
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch(m2,1,1,k) = cc(m1,1,k,1) + ( cc(m1,1,k,5) + cc(m1,1,k,2) ) &
                                  + ( cc(m1,1,k,4) + cc(m1,1,k,3) )
      ch(m2,ido,2,k) = cc(m1,1,k,1) + tr11 * ( cc(m1,1,k,5) + cc(m1,1,k,2) ) &
                                    + tr12 * ( cc(m1,1,k,4) + cc(m1,1,k,3) )
      ch(m2,1,3,k) = ti11 * ( cc(m1,1,k,5) - cc(m1,1,k,2) ) &
                   + ti12 * ( cc(m1,1,k,4) - cc(m1,1,k,3) )
      ch(m2,ido,4,k) = cc(m1,1,k,1) + tr12 * ( cc(m1,1,k,5) + cc(m1,1,k,2) ) &
                                    + tr11 * ( cc(m1,1,k,4) + cc(m1,1,k,3) )
      ch(m2,1,5,k) = ti12 * ( cc(m1,1,k,5) - cc(m1,1,k,2) ) &
                   - ti11 * ( cc(m1,1,k,4) - cc(m1,1,k,3) )
    end do
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(m2,i-1,1,k) = cc(m1,i-1,k,1)+((wa1(i-2)*cc(m1,i-1,k,2)+ &
          wa1(i-1)*cc(m1,i,k,2))+(wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)* &
          cc(m1,i,k,5)))+((wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
          cc(m1,i,k,3))+(wa3(i-2)*cc(m1,i-1,k,4)+ &
          wa3(i-1)*cc(m1,i,k,4)))
        ch(m2,i,1,k) = cc(m1,i,k,1)+((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
          cc(m1,i-1,k,5)))+((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
          cc(m1,i-1,k,4)))
        ch(m2,i-1,3,k) = cc(m1,i-1,k,1)+tr11* &
          ( wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2) &
          +wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5))+tr12* &
          ( wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3) &
          +wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))+ti11* &
          ( wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2) &
          -(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))+ti12* &
          ( wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3) &
          -(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4)))
        ch(m2,ic-1,2,k) = cc(m1,i-1,k,1)+tr11* &
          ( wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2) &
          +wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5))+tr12* &
          ( wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3) &
          +wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))-(ti11* &
          ( wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2) &
          -(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))+ti12* &
          ( wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3) &
          -(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4))))
        ch(m2,i,3,k) = (cc(m1,i,k,1)+tr11*((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
          cc(m1,i-1,k,5)))+tr12*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
          cc(m1,i-1,k,4))))+(ti11*((wa4(i-2)*cc(m1,i-1,k,5)+ &
          wa4(i-1)*cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
          cc(m1,i,k,2)))+ti12*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
          cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
          cc(m1,i,k,3))))
        ch(m2,ic,2,k) = (ti11*((wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)* &
          cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
          cc(m1,i,k,2)))+ti12*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
          cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
          cc(m1,i,k,3))))-(cc(m1,i,k,1)+tr11*((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
          cc(m1,i-1,k,5)))+tr12*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
          cc(m1,i-1,k,4))))
        ch(m2,i-1,5,k) = (cc(m1,i-1,k,1)+tr12*((wa1(i-2)* &
          cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2))+(wa4(i-2)* &
          cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5)))+tr11*((wa2(i-2)* &
          cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3))+(wa3(i-2)* &
          cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))))+(ti12*((wa1(i-2)* &
          cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2))-(wa4(i-2)* &
          cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))-ti11*((wa2(i-2)* &
          cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))-(wa3(i-2)* &
          cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4))))
        ch(m2,ic-1,4,k) = (cc(m1,i-1,k,1)+tr12*((wa1(i-2)* &
          cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2))+(wa4(i-2)* &
          cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5)))+tr11*((wa2(i-2)* &
          cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3))+(wa3(i-2)* &
          cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))))-(ti12*((wa1(i-2)* &
          cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2))-(wa4(i-2)* &
          cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))-ti11*((wa2(i-2)* &
          cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))-(wa3(i-2)* &
          cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4))))
        ch(m2,i,5,k) = (cc(m1,i,k,1)+tr12*((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
          cc(m1,i-1,k,5)))+tr11*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
          cc(m1,i-1,k,4))))+(ti12*((wa4(i-2)*cc(m1,i-1,k,5)+ &
          wa4(i-1)*cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
          cc(m1,i,k,2)))-ti11*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
          cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
          cc(m1,i,k,3))))
        ch(m2,ic,4,k) = (ti12*((wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)* &
          cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
          cc(m1,i,k,2)))-ti11*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
          cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
          cc(m1,i,k,3))))-(cc(m1,i,k,1)+tr12*((wa1(i-2)*cc(m1,i,k,2)- &
          wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
          cc(m1,i-1,k,5)))+tr11*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
          cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
          cc(m1,i-1,k,4))))
      end do
    end do
  end do

  return
end
subroutine mradfg ( m, ido, ip, l1, idl1, cc, c1, c2, im1, in1, &
  ch, ch2, im2, in2, wa )

!*****************************************************************************80
!
!! MRADFG is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1

  real ( kind = 4 ) ai1
  real ( kind = 4 ) ai2
  real ( kind = 4 ) ar1
  real ( kind = 4 ) ar1h
  real ( kind = 4 ) ar2
  real ( kind = 4 ) ar2h
  real ( kind = 4 ) arg
  real ( kind = 4 ) c1(in1,ido,l1,ip)
  real ( kind = 4 ) c2(in1,idl1,ip)
  real ( kind = 4 ) cc(in1,ido,ip,l1)
  real ( kind = 4 ) ch(in2,ido,l1,ip)
  real ( kind = 4 ) ch2(in2,idl1,ip)
  real ( kind = 4 ) dc2
  real ( kind = 4 ) dcp
  real ( kind = 4 ) ds2
  real ( kind = 4 ) dsp
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idij
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) ik
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) j2
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) nbd
  real ( kind = 4 ) tpi
  real ( kind = 4 ) wa(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  tpi = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 )
  arg = tpi / real ( ip, kind = 4 )
  dcp = cos ( arg )
  dsp = sin ( arg )
  ipph = ( ip + 1 ) / 2
  ipp2 = ip + 2
  idp2 = ido + 2
  nbd = ( ido - 1 ) / 2

  if ( ido == 1 ) then
    do ik = 1, idl1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        c2(m1,ik,1) = ch2(m2,ik,1)
      end do
    end do

  else

    do ik = 1, idl1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch2(m2,ik,1) = c2(m1,ik,1)
      end do
    end do

    do j = 2, ip
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(m2,1,k,j) = c1(m1,1,k,j)
        end do
      end do
    end do

    if ( l1 < nbd ) then
 
      is = -ido

      do j = 2, ip
        is = is + ido
        do k = 1, l1
          idij = is
          do i = 3, ido, 2
            idij = idij + 2
            m2 = m2s
            do m1 = 1, m1d, im1
              m2 = m2 + im2
              ch(m2,i-1,k,j) = wa(idij-1) * c1(m1,i-1,k,j) &
                             + wa(idij)   * c1(m1,i,k,j)
              ch(m2,i,k,j) =   wa(idij-1) * c1(m1,i,k,j)   &
                             - wa(idij)   * c1(m1,i-1,k,j)
            end do
          end do
        end do
      end do

    else
 
      is = -ido
      do j = 2, ip
        is = is + ido
        idij = is
        do i = 3, ido, 2
          idij = idij + 2
          do k = 1, l1
            m2 = m2s
            do m1 = 1, m1d, im1
              m2 = m2 + im2
              ch(m2,i-1,k,j) = wa(idij-1) * c1(m1,i-1,k,j) &
                             + wa(idij)   * c1(m1,i,k,j)
              ch(m2,i,k,j) =   wa(idij-1) * c1(m1,i,k,j)   &
                             - wa(idij)   * c1(m1,i-1,k,j)
            end do
          end do
        end do
      end do

    end if

    if ( nbd < l1 ) then

      do j = 2, ipph
        jc = ipp2 - j
        do i = 3, ido, 2
          do k = 1, l1
            m2 = m2s
            do m1 = 1, m1d, im1
              m2 = m2 + im2
              c1(m1,i-1,k,j)  = ch(m2,i-1,k,j)  + ch(m2,i-1,k,jc)
              c1(m1,i-1,k,jc) = ch(m2,i,k,j)    - ch(m2,i,k,jc)
              c1(m1,i,k,j)    = ch(m2,i,k,j)    + ch(m2,i,k,jc)
              c1(m1,i,k,jc)   = ch(m2,i-1,k,jc) - ch(m2,i-1,k,j)
            end do
          end do
        end do
      end do

    else

      do j = 2, ipph
        jc = ipp2 - j
        do k = 1, l1
          do i = 3, ido, 2
            m2 = m2s
            do m1 = 1, m1d, im1
              m2 = m2 + im2
              c1(m1,i-1,k,j)  = ch(m2,i-1,k,j)  + ch(m2,i-1,k,jc)
              c1(m1,i-1,k,jc) = ch(m2,i,k,j)    - ch(m2,i,k,jc)
              c1(m1,i,k,j)    = ch(m2,i,k,j)    + ch(m2,i,k,jc)
              c1(m1,i,k,jc)   = ch(m2,i-1,k,jc) - ch(m2,i-1,k,j)
            end do
          end do
        end do
      end do

    end if

  end if

  do j = 2, ipph
    jc = ipp2 - j
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        c1(m1,1,k,j)  = ch(m2,1,k,j)  + ch(m2,1,k,jc)
        c1(m1,1,k,jc) = ch(m2,1,k,jc) - ch(m2,1,k,j)
      end do
    end do
  end do

  ar1 = 1.0E+00
  ai1 = 0.0E+00

  do l = 2, ipph

    lc = ipp2 - l
    ar1h = dcp * ar1 - dsp * ai1
    ai1 =  dcp * ai1 + dsp * ar1
    ar1 = ar1h

    do ik = 1, idl1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch2(m2,ik,l)  = c2(m1,ik,1) + ar1 * c2(m1,ik,2)
        ch2(m2,ik,lc) =               ai1 * c2(m1,ik,ip)
      end do
    end do

    dc2 = ar1
    ds2 = ai1
    ar2 = ar1
    ai2 = ai1

    do j = 3, ipph
      jc = ipp2 - j
      ar2h = dc2 * ar2 - ds2 * ai2
      ai2  = dc2 * ai2 + ds2 * ar2
      ar2 = ar2h
      do ik = 1, idl1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch2(m2,ik,l)  = ch2(m2,ik,l)  + ar2 * c2(m1,ik,j)
          ch2(m2,ik,lc) = ch2(m2,ik,lc) + ai2 * c2(m1,ik,jc)
        end do
      end do
    end do

  end do

  do j = 2, ipph
    do ik = 1, idl1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch2(m2,ik,1) = ch2(m2,ik,1) + c2(m1,ik,j)
      end do
    end do
  end do

  if ( ido < l1 ) then

    do i = 1, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(m1,i,1,k) = ch(m2,i,k,1)
        end do
      end do
    end do

  else

    do k = 1, l1
      do i = 1, ido
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(m1,i,1,k) = ch(m2,i,k,1)
        end do
      end do
    end do

  end if

  do j = 2, ipph
    jc = ipp2 - j
    j2 = j + j
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc(m1,ido,j2-2,k) = ch(m2,1,k,j)
        cc(m1,1,j2-1,k) = ch(m2,1,k,jc)
      end do
    end do
  end do

  if ( ido == 1 ) then
    return
  end if

  if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      j2 = j + j
      do i = 3, ido, 2
        ic = idp2 - i
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            cc(m1,i-1,j2-1,k)  = ch(m2,i-1,k,j) + ch(m2,i-1,k,jc)
            cc(m1,ic-1,j2-2,k) = ch(m2,i-1,k,j) - ch(m2,i-1,k,jc)
            cc(m1,i,j2-1,k)    = ch(m2,i,k,j)   + ch(m2,i,k,jc)
            cc(m1,ic,j2-2,k)   = ch(m2,i,k,jc)  - ch(m2,i,k,j)
          end do
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      j2 = j + j
      do k = 1, l1
        do i = 3, ido, 2
          ic = idp2 - i
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            cc(m1,i-1,j2-1,k)  = ch(m2,i-1,k,j) + ch(m2,i-1,k,jc)
            cc(m1,ic-1,j2-2,k) = ch(m2,i-1,k,j) - ch(m2,i-1,k,jc)
            cc(m1,i,j2-1,k)    = ch(m2,i,k,j)   + ch(m2,i,k,jc)
            cc(m1,ic,j2-2,k)   = ch(m2,i,k,jc)  - ch(m2,i,k,j)
          end do
        end do
      end do
    end do

  end if

  return
end
subroutine mrftb1 ( m, im, n, in, c, ch, wa, fac )

!*****************************************************************************80
!
!! MRFTB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) in
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n

  real ( kind = 4 ) c(in,*)
  real ( kind = 4 ) ch(m,*)
  real ( kind = 4 ) fac(15)
  real ( kind = 4 ) half
  real ( kind = 4 ) halfm
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) im
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) ix2
  integer ( kind = 4 ) ix3
  integer ( kind = 4 ) ix4
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  real ( kind = 4 ) wa(n)

  nf = int ( fac(2) )
  na = 0

  do k1 = 1, nf

    ip = int ( fac(k1+2) )
    na = 1 - na

    if ( 5 < ip ) then
      if ( k1 /= nf ) then
        na = 1 - na
      end if
    end if

  end do

  half = 0.5E+00
  halfm = -0.5E+00
  modn = mod ( n, 2 )
  nl = n - 2
  if ( modn /= 0 ) then
    nl = n - 1
  end if

  if ( na == 0 ) then

    do j = 2, nl, 2
      m2 = 1 - im
      do i = 1, m
        m2 = m2 + im
        c(m2,j) = half * c(m2,j)
        c(m2,j+1) = halfm * c(m2,j+1)
      end do
    end do

  else

    m2 = 1 - im

    do i = 1, m
      m2 = m2 + im
      ch(i,1) = c(m2,1)
      ch(i,n) = c(m2,n)
    end do

    do j = 2, nl, 2
      m2 = 1 - im
      do i = 1, m
        m2 = m2 + im
        ch(i,j) = half * c(m2,j)
        ch(i,j+1) = halfm * c(m2,j+1)
      end do
    end do
 
  end if

  l1 = 1
  iw = 1

  do k1 = 1, nf

    ip = int ( fac(k1+2) )
    l2 = ip * l1
    ido = n / l2
    idl1 = ido * l1

    if ( ip == 4 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido

      if ( na == 0 ) then
        call mradb4 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw), wa(ix2), &
          wa(ix3) )
      else
        call mradb4 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw), wa(ix2), &
          wa(ix3) )
      end if

      na = 1 - na

    else if ( ip == 2 ) then

      if ( na == 0 ) then
        call mradb2 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw) )
      else
        call mradb2 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw) )
      end if

      na = 1 - na

    else if ( ip == 3 ) then

      ix2 = iw + ido

      if ( na == 0 ) then
        call mradb3 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw), wa(ix2) )
      else
        call mradb3 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw), wa(ix2) )
      end if

      na = 1 - na

    else if ( ip == 5 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido
      ix4 = ix3 + ido

      if ( na == 0 ) then
        call mradb5 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw), wa(ix2), &
          wa(ix3), wa(ix4) ) 
      else
        call mradb5 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw), wa(ix2), &
          wa(ix3), wa(ix4) )
      end if

      na = 1 - na

    else

      if ( na == 0 ) then
        call mradbg ( m, ido, ip, l1, idl1, c, c, c, im, in, ch, ch, 1, &
          m, wa(iw) )
      else
        call mradbg ( m, ido, ip, l1, idl1, ch, ch, ch, 1, m, c, c, im, &
          in, wa(iw) )
      end if

      if ( ido == 1 ) then
        na = 1 - na
      end if

    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ido

  end do

  return
end
subroutine mrftf1 ( m, im, n, in, c, ch, wa, fac )

!*****************************************************************************80
!
!! MRFTF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) in
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n

  real ( kind = 4 ) c(in,*)
  real ( kind = 4 ) ch(m,*)
  real ( kind = 4 ) fac(15)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) im
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) ix2
  integer ( kind = 4 ) ix3
  integer ( kind = 4 ) ix4
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) kh
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  real ( kind = 4 ) sn
  real ( kind = 4 ) tsn
  real ( kind = 4 ) tsnm
  real ( kind = 4 ) wa(n)

  nf = int ( fac(2) )
  na = 1
  l2 = n
  iw = n

  do k1 = 1, nf

    kh = nf - k1
    ip = int ( fac(kh+3) )
    l1 = l2 / ip
    ido = n / l2
    idl1 = ido * l1
    iw = iw - ( ip - 1 ) * ido
    na = 1 - na

    if ( ip == 4 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido

      if ( na == 0 ) then
        call mradf4 ( m, ido, l1, c, im, in, ch, 1,m, wa(iw), wa(ix2), &
          wa(ix3) )
      else
        call mradf4 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw), wa(ix2), &
          wa(ix3) )
      end if

    else if ( ip == 2 ) then

      if ( na == 0 ) then
        call mradf2 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw) )
      else
        call mradf2 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw) )
      end if

    else if ( ip == 3 ) then

      ix2 = iw + ido

      if ( na == 0 ) then
        call mradf3 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw), wa(ix2) )
      else
        call mradf3 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw), wa(ix2) )
      end if

    else if ( ip == 5 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido
      ix4 = ix3 + ido

      if ( na == 0 ) then
        call mradf5 ( m, ido, l1, c, im, in, ch, 1, m, wa(iw), wa(ix2), &
          wa(ix3), wa(ix4) )
      else
        call mradf5 ( m, ido, l1, ch, 1, m, c, im, in, wa(iw), wa(ix2), &
          wa(ix3), wa(ix4) )
      end if

    else

      if ( ido == 1 ) then
        na = 1 - na
      end if

      if ( na == 0 ) then
        call mradfg ( m, ido, ip, l1, idl1, c, c, c, im, in, ch, ch, 1, &
          m, wa(iw) )
        na = 1
      else
        call mradfg ( m, ido, ip, l1, idl1, ch, ch, ch, 1, m, c, c, im, &
          in, wa(iw) )
        na = 0
      end if

    end if

    l2 = l1

  end do

  sn = 1.0E+00 / real ( n, kind = 4 )
  tsn = 2.0E+00 / real ( n, kind = 4 )
  tsnm = -tsn
  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    nl = n - 1
  else
    nl = n - 2
  end if

  if ( na == 0 ) then

    m2 = 1-im
    do i = 1, m
      m2 = m2 + im
      c(m2,1) = sn * ch(i,1)
    end do

    do j = 2, nl, 2
      m2 = 1 - im
      do i = 1, m
        m2 = m2 + im
        c(m2,j) = tsn * ch(i,j)
        c(m2,j+1) = tsnm * ch(i,j+1)
      end do
    end do

    if ( modn == 0 ) then
      m2 = 1 - im
      do i = 1, m
        m2 = m2 + im
        c(m2,n) = sn * ch(i,n)
      end do
    end if

  else

    m2 = 1-im
    do i = 1, m
      m2 = m2 + im
      c(m2,1) = sn * c(m2,1)
    end do

    do j = 2, nl, 2
      m2 = 1 - im
      do i = 1, m
        m2 = m2 + im
        c(m2,j) = tsn * c(m2,j)
        c(m2,j+1) = tsnm * c(m2,j+1)
      end do
    end do

    if ( modn == 0 ) then

      m2 = 1 - im

      do i = 1, m
        m2 = m2 + im
        c(m2,n) = sn * c(m2,n)
      end do

    end if

  end if

  return
end
subroutine mrfti1 ( n, wa, fac )

!*****************************************************************************80
!
!! MRFTI1 is an FFTPACK5 auxiliary routine.
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
!    Input, integer ( kind = 4 ) N, the number for which factorization and 
!    other information is needed.
!
!    Output, real ( kind = 4 ) WA(N), trigonometric information.
!
!    Output, real ( kind = 4 ) FAC(15), factorization information.  FAC(1) is 
!    N, FAC(2) is NF, the number of factors, and FAC(3:NF+2) are the factors.
!
  implicit none

  integer ( kind = 4 ) n

  real ( kind = 8 ) arg
  real ( kind = 8 ) argh
  real ( kind = 8 ) argld
  real ( kind = 4 ) fac(15)
  real ( kind = 4 ) fi
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ib
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ii
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) ipm
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) ld
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nfm1
  integer ( kind = 4 ) nl
  integer ( kind = 4 ) nq
  integer ( kind = 4 ) nr
  integer ( kind = 4 ) ntry
  real ( kind = 8 ) tpi
  real ( kind = 4 ) wa(n)

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
      fac(nf+2) = real ( ntry, kind = 4 )
      nl = nq
!
!  If 2 is a factor, make sure it appears first in the list of factors.
!
      if ( ntry == 2 ) then
        if ( nf /= 1 ) then
          do i = 2, nf
            ib = nf - i + 2
            fac(ib+2) = fac(ib+1)
          end do
          fac(3) = 2.0E+00
        end if
      end if

    end do

  end do

  fac(1) = real ( n, kind = 4 )
  fac(2) = real ( nf, kind = 4 )
  tpi = 8.0D+00 * atan ( 1.0D+00 )
  argh = tpi / real ( n, kind = 4 )
  is = 0
  nfm1 = nf - 1
  l1 = 1

  do k1 = 1, nfm1
    ip = int ( fac(k1+2) )
    ld = 0
    l2 = l1 * ip
    ido = n / l2
    ipm = ip - 1
    do j = 1, ipm
      ld = ld + l1
      i = is
      argld = real ( ld, kind = 4 ) * argh
      fi = 0.0E+00
      do ii = 3, ido, 2
        i = i + 2
        fi = fi + 1.0E+00
        arg = fi * argld
        wa(i-1) = cos ( arg )
        wa(i) = sin ( arg )
      end do
      is = is + ido
    end do
    l1 = l2
  end do

  return
end
subroutine msntb1 ( lot, jump, n, inc, x, wsave, dsum, xh, work, ier )

!*****************************************************************************80
!
!! MSNTB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lot

  real ( kind = 8 ) dsum(*)
  real ( kind = 4 ) fnp1s4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lnxh
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) srt3s2
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xh(lot,*)
  real ( kind = 4 ) xhold

  ier = 0
  lj = ( lot - 1 ) * jump + 1

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then

    srt3s2 = sqrt ( 3.0E+00 ) / 2.0E+00 

    do m = 1, lj, jump
      xhold =  srt3s2 * ( x(m,1) + x(m,2) )
      x(m,2) = srt3s2 * ( x(m,1) - x(m,2) )
      x(m,1) = xhold
    end do

    return
  end if

  np1 = n + 1
  ns2 = n / 2

  do k = 1, ns2
    kc = np1 - k
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      t1 = x(m,k) - x(m,kc)
      t2 = wsave(k) * ( x(m,k) + x(m,kc) )
      xh(m1,k+1) = t1 + t2
      xh(m1,kc+1) = t2 - t1
    end do
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then

    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      xh(m1,ns2+2) = 4.0E+00 * x(m,ns2+1)
    end do

  end if

  do m = 1, lot
    xh(m,1) = 0.0E+00
  end do

  lnxh = lot - 1 + lot * ( np1 - 1 ) + 1
  lnsv = np1 + int ( log ( real ( np1, kind = 4 ) ) ) + 4
  lnwk = lot * np1

  call rfftmf ( lot, 1, np1, lot, xh, lnxh, wsave(ns2+1), lnsv, work, &
    lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'msntb1', -5 )
    return
  end if

  if ( mod ( np1, 2 ) == 0 ) then
    do m = 1, lot
      xh(m,np1) = xh(m,np1) + xh(m,np1)
    end do
  end if

  fnp1s4 = real ( np1, kind = 4 ) / 4.0E+00

  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    x(m,1) = fnp1s4 * xh(m1,1)
    dsum(m1) = x(m,1)
  end do

  do i = 3, n, 2
    m1 = 0
    do m = 1, lj, jump
      m1 = m1+1
      x(m,i-1) = fnp1s4 * xh(m1,i)
      dsum(m1) = dsum(m1) + fnp1s4 * xh(m1,i-1)
      x(m,i) = dsum(m1)
    end do
  end do

  if ( modn == 0 ) then
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      x(m,n) = fnp1s4 * xh(m1,n+1)
    end do
  end if

  return
end
subroutine msntf1 ( lot, jump, n, inc, x, wsave, dsum, xh, work, ier )

!*****************************************************************************80
!
!! MSNTF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lot

  real ( kind = 8 ) dsum(*)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lnxh
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) sfnp1
  real ( kind = 4 ) ssqrt3
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xh(lot,*)
  real ( kind = 4 ) xhold

  ier = 0
  lj = ( lot - 1 ) * jump + 1

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt3 = 1.0E+00 / sqrt ( 3.0E+00 )
    do m = 1, lj, jump
      xhold =  ssqrt3 * ( x(m,1) + x(m,2) )
      x(m,2) = ssqrt3 * ( x(m,1) - x(m,2) )
      x(m,1) = xhold
    end do
  end if

  np1 = n + 1
  ns2 = n / 2

  do k = 1, ns2
    kc = np1 - k
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      t1 = x(m,k) - x(m,kc)
      t2 = wsave(k) * ( x(m,k) + x(m,kc) )
      xh(m1,k+1) = t1 + t2
      xh(m1,kc+1) = t2 - t1
    end do
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      xh(m1,ns2+2) = 4.0E+00 * x(m,ns2+1)
    end do
  end if

  do m = 1, lot
    xh(m,1) = 0.0E+00
  end do

  lnxh = lot - 1 + lot * ( np1 - 1 ) + 1
  lnsv = np1 + int ( log ( real ( np1, kind = 4 ) ) ) + 4
  lnwk = lot * np1

  call rfftmf ( lot, 1, np1, lot, xh, lnxh, wsave(ns2+1), lnsv, work, &
    lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'msntf1', -5 )
    return
  end if

  if ( mod ( np1, 2 ) == 0 ) then
    do m = 1, lot
      xh(m,np1) = xh(m,np1) + xh(m,np1)
    end do
  end if

  sfnp1 = 1.0E+00 / real ( np1, kind = 4 )
  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    x(m,1) = 0.5E+00 * xh(m1,1)
    dsum(m1) = x(m,1)
  end do

  do i = 3, n, 2
    m1 = 0
    do m = 1, lj, jump
      m1 = m1 + 1
      x(m,i-1) = 0.5E+00 * xh(m1,i)
      dsum(m1) = dsum(m1) + 0.5E+00 * xh(m1,i-1)
      x(m,i) = dsum(m1)
    end do
  end do

  if ( modn /= 0 ) then
    return
  end if

  m1 = 0
  do m = 1, lj, jump
    m1 = m1 + 1
    x(m,n) = 0.5E+00 * xh(m1,n+1)
  end do

  return
end
subroutine r1f2kb ( ido, l1, cc, in1, ch, in2, wa1 )

!*****************************************************************************80
!
!! R1F2KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,2,l1)
  real ( kind = 4 ) ch(in2,ido,l1,2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) wa1(ido)

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k) + cc(1,ido,2,k)
    ch(1,1,k,2) = cc(1,1,1,k) - cc(1,ido,2,k)
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2
    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i

        ch(1,i-1,k,1) = cc(1,i-1,1,k) + cc(1,ic-1,2,k)
        ch(1,i,k,1)   = cc(1,i,1,k)   - cc(1,ic,2,k)

        ch(1,i-1,k,2) = wa1(i-2) * ( cc(1,i-1,1,k) - cc(1,ic-1,2,k) ) &
                      - wa1(i-1) * ( cc(1,i,1,k)   + cc(1,ic,2,k) )
        ch(1,i,k,2)   = wa1(i-2) * ( cc(1,i,1,k)   + cc(1,ic,2,k) ) &
                      + wa1(i-1) * ( cc(1,i-1,1,k) - cc(1,ic-1,2,k) )

      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,ido,k,1) =     cc(1,ido,1,k) + cc(1,ido,1,k)
    ch(1,ido,k,2) = - ( cc(1,1,2,k)   + cc(1,1,2,k) )
  end do

  return
end
subroutine r1f2kf ( ido, l1, cc, in1, ch, in2, wa1 )

!*****************************************************************************80
!
!! R1F2KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) ch(in2,ido,2,l1)
  real ( kind = 4 ) cc(in1,ido,l1,2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) wa1(ido)

  do k = 1, l1
    ch(1,1,1,k)   = cc(1,1,k,1) + cc(1,1,k,2)
    ch(1,ido,2,k) = cc(1,1,k,1) - cc(1,1,k,2)
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        ch(1,i,1,k) = cc(1,i,k,1)   + ( wa1(i-2) * cc(1,i,k,2) &
                                      - wa1(i-1) * cc(1,i-1,k,2) )
        ch(1,ic,2,k) = -cc(1,i,k,1) + ( wa1(i-2) * cc(1,i,k,2) &
                                      - wa1(i-1) * cc(1,i-1,k,2) ) 
        ch(1,i-1,1,k) = cc(1,i-1,k,1)  + ( wa1(i-2) * cc(1,i-1,k,2) &
                                         + wa1(i-1) * cc(1,i,k,2))
        ch(1,ic-1,2,k) = cc(1,i-1,k,1) - ( wa1(i-2) * cc(1,i-1,k,2) &
                                         + wa1(i-1) * cc(1,i,k,2))
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,1,2,k) = -cc(1,ido,k,2)
    ch(1,ido,1,k) = cc(1,ido,k,1)
  end do

  return
end
subroutine r1f3kb ( ido, l1, cc, in1, ch, in2, wa1, wa2 )

!*****************************************************************************80
!
!! R1F3KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,3,l1)
  real ( kind = 4 ) ch(in2,ido,l1,3)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) taui
  real ( kind = 4 ) taur
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)

  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 3.0E+00
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k) + 2.0E+00 * cc(1,ido,2,k)
    ch(1,1,k,2) = cc(1,1,1,k) + 2.0E+00 * taur * cc(1,ido,2,k) &
                              - 2.0E+00 * taui * cc(1,1,3,k)
    ch(1,1,k,3) = cc(1,1,1,k) + 2.0E+00 * taur * cc(1,ido,2,k) &
                              + 2.0E+00 * taui * cc(1,1,3,k)
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,k,1) = cc(1,i-1,1,k)+(cc(1,i-1,3,k)+cc(1,ic-1,2,k))
      ch(1,i,k,1) = cc(1,i,1,k)+(cc(1,i,3,k)-cc(1,ic,2,k))
      ch(1,i-1,k,2) = wa1(i-2)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))- &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k)))) -wa1(i-1)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))+ &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))))
      ch(1,i,k,2) = wa1(i-2)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))+ &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) +wa1(i-1)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))- &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k))))
      ch(1,i-1,k,3) = wa2(i-2)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))+ &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k)))) -wa2(i-1)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))- &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))))
      ch(1,i,k,3) = wa2(i-2)* &
        ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))- &
        (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) +wa2(i-1)* &
        ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))+ &
        (taui*(cc(1,i,3,k)+cc(1,ic,2,k))))
    end do
  end do

  return
end
subroutine r1f3kf ( ido, l1, cc, in1, ch, in2, wa1, wa2 )

!*****************************************************************************80
!
!! R1F3KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,l1,3)
  real ( kind = 4 ) ch(in2,ido,3,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) taui
  real ( kind = 4 ) taur
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)

  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 3.0E+00
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    ch(1,1,1,k) = cc(1,1,k,1)          + ( cc(1,1,k,2) + cc(1,1,k,3) )
    ch(1,1,3,k) =                 taui * ( cc(1,1,k,3) - cc(1,1,k,2) )
    ch(1,ido,2,k) = cc(1,1,k,1) + taur * ( cc(1,1,k,2) + cc(1,1,k,3) )
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,1,k) = cc(1,i-1,k,1)+((wa1(i-2)*cc(1,i-1,k,2)+ &
        wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3)))
      ch(1,i,1,k) = cc(1,i,k,1)+((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3)))
      ch(1,i-1,3,k) = (cc(1,i-1,k,1)+taur*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))))+(taui*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))))
      ch(1,ic-1,2,k) = (cc(1,i-1,k,1)+taur*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))))-(taui*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))))
      ch(1,i,3,k) = (cc(1,i,k,1)+taur*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))))+(taui*((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2))))
      ch(1,ic,2,k) = (taui*((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2))))-(cc(1,i,k,1)+taur*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))))
    end do
  end do

  return
end
subroutine r1f4kb ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3 )

!*****************************************************************************80
!
!! R1F4KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,4,l1)
  real ( kind = 4 ) ch(in2,ido,l1,4)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) sqrt2
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)

  sqrt2 = sqrt ( 2.0E+00 )

  do k = 1, l1
    ch(1,1,k,3) = ( cc(1,1,1,k)   + cc(1,ido,4,k) ) &
                - ( cc(1,ido,2,k) + cc(1,ido,2,k) )
    ch(1,1,k,1) = ( cc(1,1,1,k)   + cc(1,ido,4,k) ) &
                + ( cc(1,ido,2,k) + cc(1,ido,2,k) )
    ch(1,1,k,4) = ( cc(1,1,1,k)   - cc(1,ido,4,k) ) &
                + ( cc(1,1,3,k)   + cc(1,1,3,k) )
    ch(1,1,k,2) = ( cc(1,1,1,k)   - cc(1,ido,4,k) ) &
                - ( cc(1,1,3,k)   + cc(1,1,3,k) )
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        ch(1,i-1,k,1) = (cc(1,i-1,1,k)+cc(1,ic-1,4,k)) &
          +(cc(1,i-1,3,k)+cc(1,ic-1,2,k))
        ch(1,i,k,1) = (cc(1,i,1,k)-cc(1,ic,4,k)) &
          +(cc(1,i,3,k)-cc(1,ic,2,k))
        ch(1,i-1,k,2) = wa1(i-2)*((cc(1,i-1,1,k)-cc(1,ic-1,4,k)) &
          -(cc(1,i,3,k)+cc(1,ic,2,k)))-wa1(i-1) &
          *((cc(1,i,1,k)+cc(1,ic,4,k))+(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))
        ch(1,i,k,2) = wa1(i-2)*((cc(1,i,1,k)+cc(1,ic,4,k)) &
          +(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))+wa1(i-1) &
          *((cc(1,i-1,1,k)-cc(1,ic-1,4,k))-(cc(1,i,3,k)+cc(1,ic,2,k)))
        ch(1,i-1,k,3) = wa2(i-2)*((cc(1,i-1,1,k)+cc(1,ic-1,4,k)) &
          -(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))-wa2(i-1) &
          *((cc(1,i,1,k)-cc(1,ic,4,k))-(cc(1,i,3,k)-cc(1,ic,2,k)))
        ch(1,i,k,3) = wa2(i-2)*((cc(1,i,1,k)-cc(1,ic,4,k)) &
          -(cc(1,i,3,k)-cc(1,ic,2,k)))+wa2(i-1) &
          *((cc(1,i-1,1,k)+cc(1,ic-1,4,k))-(cc(1,i-1,3,k) &
          +cc(1,ic-1,2,k)))
        ch(1,i-1,k,4) = wa3(i-2)*((cc(1,i-1,1,k)-cc(1,ic-1,4,k)) &
          +(cc(1,i,3,k)+cc(1,ic,2,k)))-wa3(i-1) &
          *((cc(1,i,1,k)+cc(1,ic,4,k))-(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))
        ch(1,i,k,4) = wa3(i-2)*((cc(1,i,1,k)+cc(1,ic,4,k)) &
          -(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))+wa3(i-1) &
          *((cc(1,i-1,1,k)-cc(1,ic-1,4,k))+(cc(1,i,3,k)+cc(1,ic,2,k)))
      end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,ido,k,1) = ( cc(1,ido,1,k) + cc(1,ido,3,k) ) &
                  + ( cc(1,ido,1,k) + cc(1,ido,3,k))
    ch(1,ido,k,2) = sqrt2 * ( ( cc(1,ido,1,k) - cc(1,ido,3,k) ) &
                            - ( cc(1,1,2,k)   + cc(1,1,4,k) ) )
    ch(1,ido,k,3) = ( cc(1,1,4,k) - cc(1,1,2,k) ) &
                  + ( cc(1,1,4,k) - cc(1,1,2,k) )
    ch(1,ido,k,4) = -sqrt2 * ( ( cc(1,ido,1,k) - cc(1,ido,3,k) ) &
                             + ( cc(1,1,2,k) + cc(1,1,4,k) ) )
  end do

  return
end
subroutine r1f4kf ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3 )

!*****************************************************************************80
!
!! R1F4KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) cc(in1,ido,l1,4)
  real ( kind = 4 ) ch(in2,ido,4,l1)
  real ( kind = 4 ) hsqt2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)

  hsqt2 = sqrt ( 2.0E+00 ) / 2.0E+00 

  do k = 1, l1
    ch(1,1,1,k)   = ( cc(1,1,k,2) + cc(1,1,k,4) ) &
                  + ( cc(1,1,k,1) + cc(1,1,k,3) )
    ch(1,ido,4,k) = ( cc(1,1,k,1) + cc(1,1,k,3) ) &
                  - ( cc(1,1,k,2) + cc(1,1,k,4) )
    ch(1,ido,2,k) = cc(1,1,k,1) - cc(1,1,k,3)
    ch(1,1,3,k)   = cc(1,1,k,4) - cc(1,1,k,2)
  end do

  if ( ido < 2 ) then
    return
  end if

  if ( 2 < ido ) then

    idp2 = ido + 2

    do k = 1, l1
      do i = 3, ido, 2
        ic = idp2 - i
        ch(1,i-1,1,k) = ((wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
          cc(1,i,k,2))+(wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
          cc(1,i,k,4)))+(cc(1,i-1,k,1)+(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))
        ch(1,ic-1,4,k) = (cc(1,i-1,k,1)+(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))-((wa1(i-2)*cc(1,i-1,k,2)+ &
          wa1(i-1)*cc(1,i,k,2))+(wa3(i-2)*cc(1,i-1,k,4)+ &
          wa3(i-1)*cc(1,i,k,4)))
        ch(1,i,1,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))+(cc(1,i,k,1)+(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
        ch(1,ic,4,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))-(cc(1,i,k,1)+(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
        ch(1,i-1,3,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))-(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))+(cc(1,i-1,k,1)-(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))
        ch(1,ic-1,2,k) = (cc(1,i-1,k,1)-(wa2(i-2)*cc(1,i-1,k,3)+ &
          wa2(i-1)*cc(1,i,k,3)))-((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
          cc(1,i-1,k,2))-(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
          cc(1,i-1,k,4)))
        ch(1,i,3,k) = ((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
          cc(1,i,k,4))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
          cc(1,i,k,2)))+(cc(1,i,k,1)-(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
        ch(1,ic,2,k) = ((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
          cc(1,i,k,4))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
          cc(1,i,k,2)))-(cc(1,i,k,1)-(wa2(i-2)*cc(1,i,k,3)- &
          wa2(i-1)*cc(1,i-1,k,3)))
       end do
    end do

    if ( mod ( ido, 2 ) == 1 ) then
      return
    end if

  end if

  do k = 1, l1
    ch(1,ido,1,k) = (hsqt2*(cc(1,ido,k,2)-cc(1,ido,k,4)))+ cc(1,ido,k,1)
    ch(1,ido,3,k) = cc(1,ido,k,1)-(hsqt2*(cc(1,ido,k,2)- cc(1,ido,k,4)))
    ch(1,1,2,k) = (-hsqt2*(cc(1,ido,k,2)+cc(1,ido,k,4)))- cc(1,ido,k,3)
    ch(1,1,4,k) = (-hsqt2*(cc(1,ido,k,2)+cc(1,ido,k,4)))+ cc(1,ido,k,3)
  end do

  return
end
subroutine r1f5kb ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3, wa4 )

!*****************************************************************************80
!
!! R1F5KB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,5,l1)
  real ( kind = 4 ) ch(in2,ido,l1,5)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) ti11
  real ( kind = 4 ) ti12
  real ( kind = 4 ) tr11
  real ( kind = 4 ) tr12
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)
  real ( kind = 4 ) wa4(ido)

  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 5.0E+00
  tr11 = cos ( arg )
  ti11 = sin ( arg )
  tr12 = cos ( 2.0E+00 * arg )
  ti12 = sin ( 2.0E+00 * arg )

  do k = 1, l1

    ch(1,1,k,1) = cc(1,1,1,k) + 2.0E+00 * cc(1,ido,2,k) &
                              + 2.0E+00 * cc(1,ido,4,k)

    ch(1,1,k,2) = ( cc(1,1,1,k) &
      +   tr11 * 2.0E+00 * cc(1,ido,2,k) + tr12 * 2.0E+00 * cc(1,ido,4,k) ) &
      - ( ti11 * 2.0E+00 * cc(1,1,3,k)   + ti12 * 2.0E+00 * cc(1,1,5,k))

    ch(1,1,k,3) = ( cc(1,1,1,k) &
      +   tr12 * 2.0E+00 * cc(1,ido,2,k) + tr11 * 2.0E+00 * cc(1,ido,4,k) ) &
      - ( ti12 * 2.0E+00 * cc(1,1,3,k)   - ti11 * 2.0E+00 * cc(1,1,5,k))

    ch(1,1,k,4) = ( cc(1,1,1,k) &
      +   tr12 * 2.0E+00 * cc(1,ido,2,k) + tr11 * 2.0E+00 * cc(1,ido,4,k) ) &
      + ( ti12 * 2.0E+00 * cc(1,1,3,k)   - ti11 * 2.0E+00 * cc(1,1,5,k))

    ch(1,1,k,5) = ( cc(1,1,1,k) &
      +   tr11 * 2.0E+00 * cc(1,ido,2,k) + tr12 * 2.0E+00 * cc(1,ido,4,k) ) &
      + ( ti11 * 2.0E+00 * cc(1,1,3,k)   + ti12 * 2.0E+00 * cc(1,1,5,k) )

  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,k,1) = cc(1,i-1,1,k)+(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +(cc(1,i-1,5,k)+cc(1,ic-1,4,k))
      ch(1,i,k,1) = cc(1,i,1,k)+(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +(cc(1,i,5,k)-cc(1,ic,4,k))
      ch(1,i-1,k,2) = wa1(i-2)*((cc(1,i-1,1,k)+tr11* &
        (cc(1,i-1,3,k)+cc(1,ic-1,2,k))+tr12 &
        *(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa1(i-1)*((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))+(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,2) = wa1(i-2)*((cc(1,i,1,k)+tr11*(cc(1,i,3,k) &
        -cc(1,ic,2,k))+tr12*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti11*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))+ti12 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))+wa1(i-1) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k) &
        +cc(1,ic-1,2,k))+tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k))) &
        -(ti11*(cc(1,i,3,k)+cc(1,ic,2,k))+ti12 &
        *(cc(1,i,5,k)+cc(1,ic,4,k))))
      ch(1,i-1,k,3) = wa2(i-2) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa2(i-1) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
      cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,3) = wa2(i-2) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa2(i-1) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k))))
      ch(1,i-1,k,4) = wa3(i-2) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa3(i-1) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
      cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        -(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,4) = wa3(i-2) &
        *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        -(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa3(i-1) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k))))
      ch(1,i-1,k,5) = wa4(i-2) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa4(i-1) &
        *((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))-(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))
      ch(1,i,k,5) = wa4(i-2) &
        *((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))-(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa4(i-1) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k))))
    end do
  end do

  return
end
subroutine r1f5kf ( ido, l1, cc, in1, ch, in2, wa1, wa2, wa3, wa4 )

!*****************************************************************************80
!
!! R1F5KF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 4 ) arg
  real ( kind = 4 ) cc(in1,ido,l1,5)
  real ( kind = 4 ) ch(in2,ido,5,l1)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) k
  real ( kind = 4 ) ti11
  real ( kind = 4 ) ti12
  real ( kind = 4 ) tr11
  real ( kind = 4 ) tr12
  real ( kind = 4 ) wa1(ido)
  real ( kind = 4 ) wa2(ido)
  real ( kind = 4 ) wa3(ido)
  real ( kind = 4 ) wa4(ido)

  arg = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 ) / 5.0E+00
  tr11 = cos ( arg )
  ti11 = sin ( arg )
  tr12 = cos ( 2.0E+00 * arg )
  ti12 = sin ( 2.0E+00 * arg )

  do k = 1, l1

    ch(1,1,1,k) = cc(1,1,k,1) + ( cc(1,1,k,5) + cc(1,1,k,2) ) &
                              + ( cc(1,1,k,4) + cc(1,1,k,3) )

    ch(1,ido,2,k) = cc(1,1,k,1) + tr11 * ( cc(1,1,k,5) + cc(1,1,k,2) ) &
                                + tr12 * ( cc(1,1,k,4) + cc(1,1,k,3) )

    ch(1,1,3,k) =                 ti11 * ( cc(1,1,k,5) - cc(1,1,k,2) ) &
                                + ti12 * ( cc(1,1,k,4) - cc(1,1,k,3) )

    ch(1,ido,4,k) = cc(1,1,k,1) + tr12 * ( cc(1,1,k,5) + cc(1,1,k,2) ) &
                                + tr11 * ( cc(1,1,k,4) + cc(1,1,k,3) )

    ch(1,1,5,k) =                 ti12 * ( cc(1,1,k,5) - cc(1,1,k,2) ) &
                                - ti11 * ( cc(1,1,k,4) - cc(1,1,k,3) )
  end do

  if ( ido == 1 ) then
    return
  end if

  idp2 = ido + 2

  do k = 1, l1
    do i = 3, ido, 2
      ic = idp2 - i
      ch(1,i-1,1,k) = cc(1,i-1,k,1)+((wa1(i-2)*cc(1,i-1,k,2)+ &
        wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
        cc(1,i,k,5)))+((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))+(wa3(i-2)*cc(1,i-1,k,4)+ &
        wa3(i-1)*cc(1,i,k,4)))
      ch(1,i,1,k) = cc(1,i,k,1)+((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4)))
      ch(1,i-1,3,k) = cc(1,i-1,k,1)+tr11* &
        ( wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2) &
        +wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5))+tr12* &
        ( wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3) &
        +wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))+ti11* &
        ( wa1(i-2)*cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2) &
        -(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))+ti12* &
        ( wa2(i-2)*cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3) &
        -(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4)))
      ch(1,ic-1,2,k) = cc(1,i-1,k,1)+tr11* &
        ( wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2) &
        +wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5))+tr12* &
        ( wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3) &
        +wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))-(ti11* &
        ( wa1(i-2)*cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2) &
        -(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))+ti12* &
        ( wa2(i-2)*cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3) &
        -(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))
      ch(1,i,3,k) = (cc(1,i,k,1)+tr11*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr12*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))+(ti11*((wa4(i-2)*cc(1,i-1,k,5)+ &
        wa4(i-1)*cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))+ti12*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))
      ch(1,ic,2,k) = (ti11*((wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
        cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))+ti12*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))-(cc(1,i,k,1)+tr11*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr12*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))
      ch(1,i-1,5,k) = (cc(1,i-1,k,1)+tr12*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)* &
        cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5)))+tr11*((wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))+(wa3(i-2)* &
        cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))))+(ti12*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa4(i-2)* &
        cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))-ti11*((wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))-(wa3(i-2)* &
        cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))
      ch(1,ic-1,4,k) = (cc(1,i-1,k,1)+tr12*((wa1(i-2)* &
        cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)* &
        cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5)))+tr11*((wa2(i-2)* &
        cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))+(wa3(i-2)* &
        cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))))-(ti12*((wa1(i-2)* &
        cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa4(i-2)* &
        cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))-ti11*((wa2(i-2)* &
        cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))-(wa3(i-2)* &
        cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))
      ch(1,i,5,k) = (cc(1,i,k,1)+tr12*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr11*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))+(ti12*((wa4(i-2)*cc(1,i-1,k,5)+ &
        wa4(i-1)*cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))-ti11*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))
      ch(1,ic,4,k) = (ti12*((wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
        cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
        cc(1,i,k,2)))-ti11*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
        cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
        cc(1,i,k,3))))-(cc(1,i,k,1)+tr12*((wa1(i-2)*cc(1,i,k,2)- &
        wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
        cc(1,i-1,k,5)))+tr11*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
        cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
        cc(1,i-1,k,4))))
     end do
  end do

  return
end
subroutine r1fgkb ( ido, ip, l1, idl1, cc, c1, c2, in1, ch, ch2, in2, wa )

!*****************************************************************************80
!
!! R1FGKB is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1

  real ( kind = 4 ) ai1
  real ( kind = 4 ) ai2
  real ( kind = 4 ) ar1
  real ( kind = 4 ) ar1h
  real ( kind = 4 ) ar2
  real ( kind = 4 ) ar2h
  real ( kind = 4 ) arg
  real ( kind = 4 ) c1(in1,ido,l1,ip)
  real ( kind = 4 ) c2(in1,idl1,ip)
  real ( kind = 4 ) cc(in1,ido,ip,l1)
  real ( kind = 4 ) ch(in2,ido,l1,ip)
  real ( kind = 4 ) ch2(in2,idl1,ip)
  real ( kind = 4 ) dc2
  real ( kind = 4 ) dcp
  real ( kind = 4 ) ds2
  real ( kind = 4 ) dsp
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idij
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) ik
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) j2
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) nbd
  real ( kind = 4 ) tpi
  real ( kind = 4 ) wa(ido)

  tpi = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 )
  arg = tpi / real ( ip, kind = 4 )
  dcp = cos ( arg )
  dsp = sin ( arg )
  idp2 = ido + 2
  nbd = ( ido - 1 ) / 2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  if ( ido < l1 ) then
    do i = 1, ido
      do k = 1, l1
        ch(1,i,k,1) = cc(1,i,1,k)
      end do
    end do
  else
    do k = 1, l1
      do i = 1, ido
        ch(1,i,k,1) = cc(1,i,1,k)
      end do
    end do
  end if

  do j = 2, ipph
    jc = ipp2 - j
    j2 = j + j
    do k = 1, l1
      ch(1,1,k,j) = cc(1,ido,j2-2,k)+cc(1,ido,j2-2,k)
      ch(1,1,k,jc) = cc(1,1,j2-1,k)+cc(1,1,j2-1,k)
    end do
  end do

  if ( ido == 1 ) then

  else if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      do i = 3, ido, 2
        ic = idp2 - i
        do k = 1, l1
          ch(1,i-1,k,j) = cc(1,i-1,2*j-1,k)+cc(1,ic-1,2*j-2,k)
          ch(1,i-1,k,jc) = cc(1,i-1,2*j-1,k)-cc(1,ic-1,2*j-2,k)
          ch(1,i,k,j) = cc(1,i,2*j-1,k)-cc(1,ic,2*j-2,k)
          ch(1,i,k,jc) = cc(1,i,2*j-1,k)+cc(1,ic,2*j-2,k)
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do k = 1, l1
        do i = 3, ido, 2
          ic = idp2 - i
          ch(1,i-1,k,j) = cc(1,i-1,2*j-1,k)+cc(1,ic-1,2*j-2,k)
          ch(1,i-1,k,jc) = cc(1,i-1,2*j-1,k)-cc(1,ic-1,2*j-2,k)
          ch(1,i,k,j) = cc(1,i,2*j-1,k)-cc(1,ic,2*j-2,k)
          ch(1,i,k,jc) = cc(1,i,2*j-1,k)+cc(1,ic,2*j-2,k)
        end do
      end do
    end do

  end if

  ar1 = 1.0E+00
  ai1 = 0.0E+00

  do l = 2, ipph

    lc = ipp2 - l
    ar1h = dcp * ar1 - dsp * ai1
    ai1 = dcp * ai1 + dsp * ar1
    ar1 = ar1h

    do ik = 1, idl1
      c2(1,ik,l) = ch2(1,ik,1)+ar1*ch2(1,ik,2)
      c2(1,ik,lc) = ai1*ch2(1,ik,ip)
    end do

    dc2 = ar1
    ds2 = ai1
    ar2 = ar1
    ai2 = ai1

    do j = 3, ipph

      jc = ipp2 - j
      ar2h = dc2*ar2-ds2*ai2
      ai2 = dc2*ai2+ds2*ar2
      ar2 = ar2h

      do ik = 1, idl1
        c2(1,ik,l) = c2(1,ik,l)+ar2*ch2(1,ik,j)
        c2(1,ik,lc) = c2(1,ik,lc)+ai2*ch2(1,ik,jc)
      end do

    end do

  end do

  do j = 2, ipph
    do ik = 1, idl1
      ch2(1,ik,1) = ch2(1,ik,1)+ch2(1,ik,j)
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do k = 1, l1
      ch(1,1,k,j) = c1(1,1,k,j)-c1(1,1,k,jc)
      ch(1,1,k,jc) = c1(1,1,k,j)+c1(1,1,k,jc)
    end do
  end do

  if ( ido == 1 ) then

  else if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      do i = 3, ido, 2
        do k = 1, l1
          ch(1,i-1,k,j)  = c1(1,i-1,k,j) - c1(1,i,k,jc)
          ch(1,i-1,k,jc) = c1(1,i-1,k,j) + c1(1,i,k,jc)
          ch(1,i,k,j)    = c1(1,i,k,j)   + c1(1,i-1,k,jc)
          ch(1,i,k,jc)   = c1(1,i,k,j)   - c1(1,i-1,k,jc)
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do k = 1, l1
        do i = 3, ido, 2
          ch(1,i-1,k,j) = c1(1,i-1,k,j)-c1(1,i,k,jc)
          ch(1,i-1,k,jc) = c1(1,i-1,k,j)+c1(1,i,k,jc)
          ch(1,i,k,j) = c1(1,i,k,j)+c1(1,i-1,k,jc)
          ch(1,i,k,jc) = c1(1,i,k,j)-c1(1,i-1,k,jc)
        end do
      end do
    end do

  end if

  if ( ido == 1 ) then
    return
  end if

  do ik = 1, idl1
    c2(1,ik,1) = ch2(1,ik,1)
  end do

  do j = 2, ip
    do k = 1, l1
      c1(1,1,k,j) = ch(1,1,k,j)
    end do
  end do

  if ( l1 < nbd ) then

    is = -ido
    do j = 2, ip
       is = is + ido
       do k = 1, l1
         idij = is
         do i = 3, ido, 2
           idij = idij + 2
           c1(1,i-1,k,j) = wa(idij-1)*ch(1,i-1,k,j)-wa(idij)* ch(1,i,k,j)
           c1(1,i,k,j) = wa(idij-1)*ch(1,i,k,j)+wa(idij)* ch(1,i-1,k,j)
         end do
       end do
    end do

  else

    is = -ido

    do j = 2, ip
      is = is + ido
      idij = is
      do i = 3, ido, 2
        idij = idij + 2
        do k = 1, l1
           c1(1,i-1,k,j) = wa(idij-1) * ch(1,i-1,k,j) - wa(idij) * ch(1,i,k,j)
           c1(1,i,k,j)   = wa(idij-1) * ch(1,i,k,j)   + wa(idij) * ch(1,i-1,k,j)
        end do
      end do
    end do

  end if

  return
end
subroutine r1fgkf ( ido, ip, l1, idl1, cc, c1, c2, in1, ch, ch2, in2, wa )

!*****************************************************************************80
!
!! R1FGKF is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1

  real ( kind = 4 ) ai1
  real ( kind = 4 ) ai2
  real ( kind = 4 ) ar1
  real ( kind = 4 ) ar1h
  real ( kind = 4 ) ar2
  real ( kind = 4 ) ar2h
  real ( kind = 4 ) arg
  real ( kind = 4 ) c1(in1,ido,l1,ip)
  real ( kind = 4 ) c2(in1,idl1,ip)
  real ( kind = 4 ) cc(in1,ido,ip,l1)
  real ( kind = 4 ) ch(in2,ido,l1,ip)
  real ( kind = 4 ) ch2(in2,idl1,ip)
  real ( kind = 4 ) dc2
  real ( kind = 4 ) dcp
  real ( kind = 4 ) ds2
  real ( kind = 4 ) dsp
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ic
  integer ( kind = 4 ) idij
  integer ( kind = 4 ) idp2
  integer ( kind = 4 ) ik
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) j2
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) nbd
  real ( kind = 4 ) tpi
  real ( kind = 4 ) wa(ido)

  tpi = 2.0E+00 * 4.0E+00 * atan ( 1.0E+00 )
  arg = tpi / real ( ip, kind = 4 )
  dcp = cos ( arg )
  dsp = sin ( arg )
  ipph = ( ip + 1 ) / 2
  ipp2 = ip + 2
  idp2 = ido + 2
  nbd = ( ido - 1 ) / 2

  if ( ido == 1 ) then

    do ik = 1, idl1
      c2(1,ik,1) = ch2(1,ik,1)
    end do
 
  else

    do ik = 1, idl1
      ch2(1,ik,1) = c2(1,ik,1)
    end do

    do j = 2, ip
      do k = 1, l1
        ch(1,1,k,j) = c1(1,1,k,j)
      end do
    end do

    if ( l1 < nbd ) then

      is = -ido

      do j = 2, ip
        is = is + ido
        do k = 1, l1
          idij = is
          do i = 3, ido, 2
            idij = idij + 2
            ch(1,i-1,k,j) = wa(idij-1)*c1(1,i-1,k,j)+wa(idij) *c1(1,i,k,j)
            ch(1,i,k,j) = wa(idij-1)*c1(1,i,k,j)-wa(idij) *c1(1,i-1,k,j)
          end do
        end do
      end do

    else

      is = -ido

      do j = 2, ip
        is = is + ido
        idij = is
        do i = 3, ido, 2
          idij = idij + 2
          do k = 1, l1
            ch(1,i-1,k,j) = wa(idij-1)*c1(1,i-1,k,j)+wa(idij) *c1(1,i,k,j)
            ch(1,i,k,j) = wa(idij-1)*c1(1,i,k,j)-wa(idij) *c1(1,i-1,k,j)
          end do
        end do
      end do

    end if

    if ( nbd < l1 ) then

      do j = 2, ipph
        jc = ipp2 - j
        do i = 3, ido, 2
          do k = 1, l1
            c1(1,i-1,k,j) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
            c1(1,i-1,k,jc) = ch(1,i,k,j)-ch(1,i,k,jc)
            c1(1,i,k,j) = ch(1,i,k,j)+ch(1,i,k,jc)
            c1(1,i,k,jc) = ch(1,i-1,k,jc)-ch(1,i-1,k,j)
          end do
        end do
      end do

    else

      do j = 2, ipph
        jc = ipp2 - j
        do k = 1, l1
          do i = 3, ido, 2
            c1(1,i-1,k,j) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
            c1(1,i-1,k,jc) = ch(1,i,k,j)-ch(1,i,k,jc)
            c1(1,i,k,j) = ch(1,i,k,j)+ch(1,i,k,jc)
            c1(1,i,k,jc) = ch(1,i-1,k,jc)-ch(1,i-1,k,j)
          end do
        end do
      end do

    end if

  end if

  do j = 2, ipph
    jc = ipp2 - j
    do k = 1, l1
      c1(1,1,k,j) = ch(1,1,k,j)+ch(1,1,k,jc)
      c1(1,1,k,jc) = ch(1,1,k,jc)-ch(1,1,k,j)
    end do
  end do

  ar1 = 1.0E+00
  ai1 = 0.0E+00

  do l = 2, ipph

    lc = ipp2 - l
    ar1h = dcp * ar1 - dsp * ai1
    ai1 = dcp * ai1 + dsp * ar1
    ar1 = ar1h

    do ik = 1, idl1
      ch2(1,ik,l) = c2(1,ik,1)+ar1*c2(1,ik,2)
      ch2(1,ik,lc) = ai1*c2(1,ik,ip)
    end do

    dc2 = ar1
    ds2 = ai1
    ar2 = ar1
    ai2 = ai1

    do j = 3, ipph
      jc = ipp2 - j
      ar2h = dc2 * ar2 - ds2 * ai2
      ai2 = dc2 * ai2 + ds2 * ar2
      ar2 = ar2h
      do ik = 1, idl1
        ch2(1,ik,l) = ch2(1,ik,l)+ar2*c2(1,ik,j)
        ch2(1,ik,lc) = ch2(1,ik,lc)+ai2*c2(1,ik,jc)
      end do
    end do

  end do

  do j = 2, ipph
    do ik = 1, idl1
      ch2(1,ik,1) = ch2(1,ik,1)+c2(1,ik,j)
    end do
  end do

  if ( ido < l1 ) then

    do i = 1, ido
      do k = 1, l1
        cc(1,i,1,k) = ch(1,i,k,1)
      end do
    end do

  else

    do k = 1, l1
      do i = 1, ido
        cc(1,i,1,k) = ch(1,i,k,1)
      end do
    end do

  end if

  do j = 2, ipph
    jc = ipp2 - j
    j2 = j+j
    do k = 1, l1
      cc(1,ido,j2-2,k) = ch(1,1,k,j)
      cc(1,1,j2-1,k) = ch(1,1,k,jc)
    end do
  end do

  if ( ido == 1 ) then
    return
  end if

  if ( nbd < l1 ) then

    do j = 2, ipph
      jc = ipp2 - j
      j2 = j + j
      do i = 3, ido, 2
        ic = idp2 - i
        do k = 1, l1
          cc(1,i-1,j2-1,k) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
          cc(1,ic-1,j2-2,k) = ch(1,i-1,k,j)-ch(1,i-1,k,jc)
          cc(1,i,j2-1,k) = ch(1,i,k,j)+ch(1,i,k,jc)
          cc(1,ic,j2-2,k) = ch(1,i,k,jc)-ch(1,i,k,j)
        end do
      end do
   end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      j2 = j + j
      do k = 1, l1
        do i = 3, ido, 2
          ic = idp2 - i
          cc(1,i-1,j2-1,k)  = ch(1,i-1,k,j) + ch(1,i-1,k,jc)
          cc(1,ic-1,j2-2,k) = ch(1,i-1,k,j) - ch(1,i-1,k,jc)
          cc(1,i,j2-1,k)    = ch(1,i,k,j)   + ch(1,i,k,jc)
          cc(1,ic,j2-2,k)   = ch(1,i,k,jc)  - ch(1,i,k,j)
        end do
      end do
    end do

  end if

  return
end
subroutine r4_factor ( n, nf, fac )

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
subroutine r4_mcfti1 ( n, wa, fnf, fac )

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
subroutine r4_tables ( ido, ip, wa )

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
subroutine r8_factor ( n, nf, fac )

!*****************************************************************************80
!
!! R8_FACTOR factors of an integer for real double precision computations.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 August 2009
!
!  Author:
!
!    Original real single precision version by Paul Swarztrauber, Dick Valent.
!    Real double precision version by John Burkardt.
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
!    Output, real ( kind = 8 ) FAC(*), a list of factors of N.
!
  implicit none

  real ( kind = 8 ) fac(*)
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
      fac(nf) = real ( ntry, kind = 8 )
      nl = nq

    end do

  end do

  return
end
subroutine r8_mcfti1 ( n, wa, fnf, fac )

!*****************************************************************************80
!
!! R8_MCFTI1 sets up factors and tables, real double precision arithmetic.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 August 2009
!
!  Author:
!
!    Original real single precision version by Paul Swarztrauber, Dick Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  real ( kind = 8 ) fac(*)
  real ( kind = 8 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nf
  real ( kind = 8 ) wa(*)
!
!  Get the factorization of N.
!
  call r8_factor ( n, nf, fac )
  fnf = real ( nf, kind = 8 )
  iw = 1
  l1 = 1
!
!  Set up the trigonometric tables.
!
  do k1 = 1, nf
    ip = int ( fac(k1) )
    l2 = l1 * ip
    ido = n / l2
    call r8_tables ( ido, ip, wa(iw) )
    iw = iw + ( ip - 1 ) * ( ido + ido )
    l1 = l2
  end do

  return
end
subroutine r8_tables ( ido, ip, wa )

!*****************************************************************************80
!
!! R8_TABLES computes trigonometric tables, real double precision arithmetic.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    27 August 2009
!
!  Author:
!
!    Original real single precision version by Paul Swarztrauber, Dick Valent.
!    Real double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip

  real ( kind = 8 ) arg1
  real ( kind = 8 ) arg2
  real ( kind = 8 ) arg3
  real ( kind = 8 ) arg4
  real ( kind = 8 ) argz
  integer ( kind = 4 ) i
  integer ( kind = 4 ) j
  real ( kind = 8 ) tpi
  real ( kind = 8 ) wa(ido,ip-1,2)

  tpi = 8.0D+00 * atan ( 1.0D+00 )
  argz = tpi / real ( ip, kind = 8 )
  arg1 = tpi / real ( ido * ip, kind = 8 )

  do j = 2, ip

    arg2 = real ( j - 1, kind = 8 ) * arg1

    do i = 1, ido
      arg3 = real ( i - 1, kind = 8 ) * arg2
      wa(i,j-1,1) = cos ( arg3 )
      wa(i,j-1,2) = sin ( arg3 )
    end do

    if ( 5 < ip ) then
      arg4 = real ( j - 1, kind = 8 ) * argz
      wa(1,j-1,1) = cos ( arg4 )
      wa(1,j-1,2) = sin ( arg4 )
    end if

  end do

  return
end
subroutine rfft1b ( n, inc, r, lenr, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! RFFT1B: real single precision backward fast Fourier transform, 1D.
!
!  Discussion:
!
!    RFFT1B computes the one-dimensional Fourier transform of a periodic
!    sequence within a real array.  This is referred to as the backward
!    transform or Fourier synthesis, transforming the sequence from 
!    spectral to physical space.  This transform is normalized since a 
!    call to RFFT1B followed by a call to RFFT1F (or vice-versa) reproduces
!    the original array within roundoff error. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    25 March 2005
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
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence. 
!
!    Input/output, real ( kind = 4 ) R(LENR), on input, the data to be 
!    transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1) + 1. 
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFT1I before the first call to routine
!    RFFT1F or RFFT1B for a given transform length N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4. 
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough.
!
  implicit none

  integer ( kind = 4 ) lenr
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) n
  real ( kind = 4 ) r(lenr)
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lenr < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'rfft1b ', 6 )
    return
  end if

  if ( lensav < n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'rfft1b ', 8 )
    return
  end if

  if ( lenwrk < n ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'RFFT1B - Fatal error!'
    write ( *, '(a)' ) '  LENWRK < N:'
    write ( *, '(a,i6)' ) '  LENWRK = ', lenwrk
    write ( *, '(a,i6)' ) '  N = ', n
    ier = 3
    call xerfft ( 'rfft1b ', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call rfftb1 ( n, inc, r, work, wsave, wsave(n+1) )

  return
end
subroutine rfft1f ( n, inc, r, lenr, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! RFFT1F: real single precision forward fast Fourier transform, 1D.
!
!  Discussion:
!
!    RFFT1F computes the one-dimensional Fourier transform of a periodic
!    sequence within a real array.  This is referred to as the forward 
!    transform or Fourier analysis, transforming the sequence from physical
!    to spectral space.  This transform is normalized since a call to 
!    RFFT1F followed by a call to RFFT1B (or vice-versa) reproduces the 
!    original array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    25 March 2005
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
!    array R, of two consecutive elements within the sequence. 
!
!    Input/output, real ( kind = 4 ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1) + 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFT1I before the first call to routine RFFT1F
!    or RFFT1B for a given transform length N. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array. 
!    LENWRK must be at least N. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough:
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough.
!
  implicit none

  integer ( kind = 4 ) lenr
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) r(lenr)

  ier = 0

  if ( lenr < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'rfft1f ', 6 )
    return
  end if

  if ( lensav < n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'rfft1f ', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'rfft1f ', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call rfftf1 ( n, inc, r, work, wsave, wsave(n+1) )

  return
end
subroutine rfft1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! RFFT1I: initialization for RFFT1B and RFFT1F.
!
!  Discussion:
!
!    RFFT1I initializes array WSAVE for use in its companion routines 
!    RFFT1B and RFFT1F.  The prime factorization of N together with a
!    tabulation of the trigonometric functions are computed and stored
!    in array WSAVE.  Separate WSAVE arrays are required for different
!    values of N. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    25 March 2005
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
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors of
!    N and also containing certain trigonometric values which will be used in
!    routines RFFT1B or RFFT1F.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) n
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'rfft1i ', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call rffti1 ( n, wsave(1), wsave(n+1) )

  return
end
subroutine rfft2b ( ldim, l, m, r, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! RFFT2B: real single precision backward fast Fourier transform, 2D.
!
!  Discussion:
!
!    RFFT2B computes the two-dimensional discrete Fourier transform of the
!    complex Fourier coefficients a real periodic array.  This transform is
!    known as the backward transform or Fourier synthesis, transforming from
!    spectral to physical space.  Routine RFFT2B is normalized: a call to 
!    RFFT2B followed by a call to RFFT2F (or vice-versa) reproduces the 
!    original array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    26 March 2005
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
!    Input, integer ( kind = 4 ) LDIM, the first dimension of the 2D real 
!    array R, which must be at least 2*(L/2+1).
!
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension of the two-dimensional real array R.  The value of 
!    L must be less than or equal to that of LDIM.  The transform is most 
!    efficient when L is a product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension of the two-dimensional real array R.  The transform
!    is most efficient when M is a product of small primes.
!
!    Input/output, real ( kind = 4 ) R(LDIM,M), the real array of two 
!    dimensions.  On input, R contains the L/2+1-by-M complex subarray of 
!    spectral coefficients, on output, the physical coefficients.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFT2I before the first call to routine RFFT2F 
!    or RFFT2B with lengths L and M.  WSAVE's contents may be re-used for 
!    subsequent calls to RFFT2F and RFFT2B with the same transform lengths 
!    L and M. 
!
!    Input, integer ( kind = 4 ) LENSAV, the number of elements in the WSAVE 
!    array.  LENSAV must be at least L + M + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).  WORK provides workspace, and 
!    its contents need not be saved between calls to routines RFFT2B and RFFT2F.
!
!    Input, integer ( kind = 4 )  LENWRK, the number of elements in the WORK 
!    array.  LENWRK must be at least LDIM*M.
!
!    Output, integer ( kind = 4 ) IER, the error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    6, input parameter LDIM < 2*(L/2+1);
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) ldim
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk
  integer ( kind = 4 ) m

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) j
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lwsav
  integer ( kind = 4 ) mwsav
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) r(ldim,m)

  ier = 0
!
!  Verify LENSAV.
!
  lwsav = l + int ( log ( real ( l, kind = 4 ) ) ) + 4
  mwsav = 2 * m + int ( log ( real ( m, kind = 4 ) ) ) + 4

  if ( lensav < lwsav + mwsav ) then
    ier = 2
    call xerfft ( 'rfft2b', 6 )
    return
  end if
!
!  Verify LENWRK.
!
  if ( lenwrk < 2 * ( l / 2 + 1 ) * m ) then
    ier = 3
    call xerfft ( 'rfft2b', 8 )
    return
  end if
!
!  Verify LDIM is as big as L.
!
  if ( ldim < 2 * ( l / 2 + 1 ) ) then
    ier = 5
    call xerfft ( 'rfft2b', -6 )
    return
  end if
!
!  Transform second dimension of array.
!
  call cfftmb ( l/2+1, 1, m, ldim/2, r, m*ldim/2, &
    wsave(l+int(log( real ( l, kind = 4 )))+5), &
    2*m+int(log( real ( m, kind = 4 )))+4, work, &
    2*(l/2+1)*m, ier1 )

  if ( ier1 /= 0 ) then
     ier = 20
     call xerfft ( 'rfft2b', -5 )
     return
  end if
!
!  Reshuffle.
!
  do j = 1, m
    do i = 2, l
      r(i,j) = r(i+1,j)
    end do
  end do
!
!  Transform first dimension of array.
!
  call rfftmb ( m, ldim, l, 1, r, m*ldim, wsave(1), &
    l+int(log( real ( l, kind = 4 )))+4, work, 2*(l/2+1)*m, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'rfft2f', -5 )
    return
  end if

  return
end
subroutine rfft2f ( ldim, l, m, r, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
! RFFT2F: real single precision forward fast Fourier transform, 2D.
!
!  Discussion:
!
!    RFFT2F computes the two-dimensional discrete Fourier transform of a 
!    real periodic array.  This transform is known as the forward transform 
!    or Fourier analysis, transforming from physical to spectral space. 
!    Routine RFFT2F is normalized: a call to RFFT2F followed by a call to 
!    RFFT2B (or vice-versa) reproduces the original array within roundoff 
!    error. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    26 March 2005
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
!    Input, integer ( kind = 4 ) LDIM, the first dimension of the 2D real
!    array R, which must be at least 2*(L/2+1). 
!
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension of the two-dimensional real array R.  The value 
!    of L must be less than or equal to that of LDIM.  The transform is most 
!    efficient when L is a product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension of the two-dimensional real array R.  The
!    transform is most efficient when M is a product of small primes. 
!
!    Input/output, real ( kind = 4 ) R(LDIM,M), the real array of two 
!    dimensions.  On input, containing the L-by-M physical data to be 
!    transformed.  On output, the spectral coefficients.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFT2I before the first call to routine RFFT2F
!    or RFFT2B with lengths L and M.  WSAVE's contents may be re-used for
!    subsequent calls to RFFT2F and RFFT2B with the same transform lengths. 
!
!    Input, integer ( kind = 4 ) LENSAV, the number of elements in the WSAVE 
!    array.  LENSAV must be at least L + M + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8. 
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK), provides workspace, and its 
!    contents need not be saved between calls to routines RFFT2F and RFFT2B. 
!
!    Input, integer ( kind = 4 ) LENWRK, the number of elements in the WORK 
!    array.  LENWRK must be at least LDIM*M.
!
!    Output, integer ( kind = 4 ) IER, the error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    6, input parameter LDIM < 2*(L+1);
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) ldim
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk
  integer ( kind = 4 ) m

  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) j
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lwsav
  integer ( kind = 4 ) mwsav
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) r(ldim,m)

  ier = 0
!
!  Verify LENSAV.
!
  lwsav = l + int ( log ( real ( l, kind = 4 ) ) ) + 4
  mwsav = 2 * m + int ( log ( real ( m, kind = 4 ) ) ) + 4

  if ( lensav < lwsav + mwsav ) then
    ier = 2
    call xerfft ( 'rfft2f', 6 )
    return
  end if
!
!  Verify LENWRK.
!
  if ( lenwrk < 2 * ( l / 2 + 1 ) * m ) then
    ier = 3
    call xerfft ( 'rfft2f', 8 )
    return
  end if
!
!  Verify LDIM is as big as L.
!
  if ( ldim < 2 * ( l / 2 + 1 ) ) then
    ier = 5
    call xerfft ( 'rfft2f', -6 )
    return
  end if
!
!  Transform first dimension of array.
!
  call rfftmf ( m, ldim, l, 1, r, m*ldim, wsave(1), &
    l+int(log( real ( l, kind = 4 )))+4, work,2*(l/2+1)*m, ier1 )

  if ( ier1 /= 0 ) then
     ier = 20
     call xerfft ( 'rfft2f', -5 )
     return
  end if
!
!  Reshuffle to add in Nyquist imaginary components.
!
  do j = 1, m
    if ( mod ( l, 2 ) == 0 ) then
      r(l+2,j) = 0.0E+00
    end if
    do i = l, 2, -1
      r(i+1,j) = r(i,j)
    end do
    r(2,j) = 0.0E+00
  end do
!
!  Transform second dimension of array.
!
  call cfftmf ( l/2+1, 1, m, ldim/2, r, m*ldim/2, &
    wsave(l+int(log( real ( l, kind = 4 )))+5), &
    2*m+int(log( real ( m, kind = 4 )))+4, work, 2*(l/2+1)*m, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'rfft2f', -5 )
    return
  end if

  return
end
subroutine rfft2i ( l, m, wsave, lensav, ier )

!*****************************************************************************80
!
!! RFFT2I: initialization for RFFT2B and RFFT2F.
!
!  Discussion:
!
!    RFFT2I initializes real array WSAVE for use in its companion routines
!    RFFT2F and RFFT2B for computing the two-dimensional fast Fourier 
!    transform of real data.  Prime factorizations of L and M, together with
!    tabulations of the trigonometric functions, are computed and stored in
!    array WSAVE.  RFFT2I must be called prior to the first call to RFFT2F 
!    or RFFT2B.  Separate WSAVE arrays are required for different values of 
!    L or M.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    26 March 2005
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
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension.  The transform is most efficient when L is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension.  The transform is most efficient when M is a 
!    product of small primes. 
!
!    Input, integer ( kind = 4 ) LENSAV, the number of elements in the WSAVE 
!    array.  LENSAV must be at least L + M + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors 
!    of L and M, and also containing certain trigonometric values which 
!    will be used in routines RFFT2B or RFFT2F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lwsav
  integer ( kind = 4 ) m
  integer ( kind = 4 ) mwsav
  real ( kind = 4 ) wsave(lensav)

  ier = 0
!
!  Verify LENSAV.
!
  lwsav = l + int ( log ( real ( l, kind = 4 ) ) ) + 4
  mwsav = 2 * m + int ( log ( real ( m, kind = 4 ) ) ) + 4

  if ( lensav < lwsav + mwsav ) then
    ier = 2
    call xerfft ( 'rfft2i', 4 )
    return
  end if

  call rfftmi ( l, wsave(1), l + int(log( real ( l, kind = 4 ))) + 4, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'rfft2i', -5 )
    return
  end if

  call cfftmi ( m, wsave(l+int(log( real ( l, kind = 4 )))+5), &
    2*m+int(log( real ( m, kind = 4 )))+4, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'rfft2i', -5 )
    return
  end if

  return
end
subroutine rfftb1 ( n, in, c, ch, wa, fac )

!*****************************************************************************80
!
!! RFFTB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) in
  integer ( kind = 4 ) n

  real ( kind = 4 ) c(in,*)
  real ( kind = 4 ) ch(*)
  real ( kind = 4 ) fac(15)
  real ( kind = 4 ) half
  real ( kind = 4 ) halfm
  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) ix2
  integer ( kind = 4 ) ix3
  integer ( kind = 4 ) ix4
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  real ( kind = 4 ) wa(n)

  nf = int ( fac(2) )
  na = 0

  do k1 = 1, nf

    ip = int ( fac(k1+2) )
    na = 1 - na

    if ( 5 < ip ) then
      if ( k1 /= nf ) then
        na = 1 - na
      end if
    end if

  end do

  half = 0.5E+00
  halfm = -0.5E+00
  modn = mod ( n, 2 )
  nl = n - 2
  if ( modn /= 0 ) then
    nl = n - 1
  end if

  if ( na == 0 ) then

    do j = 2, nl, 2
      c(1,j) = half * c(1,j)
      c(1,j+1) = halfm * c(1,j+1)
    end do

  else

    ch(1) = c(1,1)
    ch(n) = c(1,n)

    do j = 2, nl, 2
      ch(j) = half*c(1,j)
      ch(j+1) = halfm*c(1,j+1)
    end do

  end if

  l1 = 1
  iw = 1

  do k1 = 1, nf

    ip = int ( fac(k1+2) )
    l2 = ip * l1
    ido = n / l2
    idl1 = ido * l1

    if ( ip == 4 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido

      if ( na == 0 ) then
        call r1f4kb ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3) )
      else
        call r1f4kb ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3) )
      end if

      na = 1 - na

    else if ( ip == 2 ) then

      if ( na == 0 ) then
        call r1f2kb ( ido, l1, c, in, ch, 1, wa(iw) )
      else
        call r1f2kb ( ido, l1, ch, 1, c, in, wa(iw) )
      end if

      na = 1 - na

    else if ( ip == 3 ) then

      ix2 = iw + ido

      if ( na == 0 ) then
        call r1f3kb ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2) )
      else
        call r1f3kb ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2) )
      end if

      na = 1 - na

    else if ( ip == 5 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido
      ix4 = ix3 + ido

      if ( na == 0 ) then
        call r1f5kb ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      else
        call r1f5kb ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      end if

      na = 1 - na

    else

      if ( na == 0 ) then
        call r1fgkb ( ido, ip, l1, idl1, c, c, c, in, ch, ch, 1, wa(iw) )
      else
        call r1fgkb ( ido, ip, l1, idl1, ch, ch, ch, 1, c, c, in, wa(iw) )
      end if

      if ( ido == 1 ) then
        na = 1 - na
      end if

    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ido

  end do

  return
end
subroutine rfftf1 ( n, in, c, ch, wa, fac )

!*****************************************************************************80
!
!! RFFTF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) in
  integer ( kind = 4 ) n

  real ( kind = 4 ) c(in,*)
  real ( kind = 4 ) ch(*)
  real ( kind = 4 ) fac(15)
  integer ( kind = 4 ) idl1
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) ix2
  integer ( kind = 4 ) ix3
  integer ( kind = 4 ) ix4
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) kh
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nl
  real ( kind = 4 ) sn
  real ( kind = 4 ) tsn
  real ( kind = 4 ) tsnm
  real ( kind = 4 ) wa(n)

  nf = int ( fac(2) )
  na = 1
  l2 = n
  iw = n

  do k1 = 1, nf

    kh = nf - k1
    ip = int ( fac(kh+3) )
    l1 = l2 / ip
    ido = n / l2
    idl1 = ido * l1
    iw = iw - ( ip - 1 ) * ido
    na = 1 - na

    if ( ip == 4 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido

      if ( na == 0 ) then
        call r1f4kf ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3) )
      else
        call r1f4kf ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3) )
      end if

    else if ( ip == 2 ) then

      if ( na == 0 ) then
        call r1f2kf ( ido, l1, c, in, ch, 1, wa(iw) )
      else
        call r1f2kf ( ido, l1, ch, 1, c, in, wa(iw) )
      end if

    else if ( ip == 3 ) then

      ix2 = iw + ido

      if ( na == 0 ) then
        call r1f3kf ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2) )
      else
        call r1f3kf ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2) )
      end if

    else if ( ip == 5 ) then

      ix2 = iw + ido
      ix3 = ix2 + ido
      ix4 = ix3 + ido

      if ( na == 0 ) then
        call r1f5kf ( ido, l1, c, in, ch, 1, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      else
        call r1f5kf ( ido, l1, ch, 1, c, in, wa(iw), wa(ix2), wa(ix3), wa(ix4) )
      end if

    else

      if ( ido == 1 ) then
        na = 1 - na
      end if

      if ( na == 0 ) then
        call r1fgkf ( ido, ip, l1, idl1, c, c, c, in, ch, ch, 1, wa(iw) )
        na = 1
      else
        call r1fgkf ( ido, ip, l1, idl1, ch, ch, ch, 1, c, c, in, wa(iw) )
        na = 0
      end if

    end if

    l2 = l1

  end do

  sn = 1.0E+00 / real ( n, kind = 4 )
  tsn = 2.0E+00 / real ( n, kind = 4 )
  tsnm = -tsn
  modn = mod ( n, 2 )
  nl = n - 2

  if ( modn /= 0 ) then
    nl = n - 1
  end if

  if ( na == 0 ) then

    c(1,1) = sn * ch(1)
    do j = 2, nl, 2
      c(1,j) = tsn * ch(j)
      c(1,j+1) = tsnm * ch(j+1)
    end do

    if ( modn == 0 ) then
      c(1,n) = sn * ch(n)
    end if

  else

    c(1,1) = sn * c(1,1)

    do j = 2, nl, 2
      c(1,j) = tsn * c(1,j)
      c(1,j+1) = tsnm * c(1,j+1)
    end do

    if ( modn == 0 ) then
      c(1,n) = sn * c(1,n)
    end if

  end if

  return
end
subroutine rffti1 ( n, wa, fac )

!*****************************************************************************80
!
!! RFFTI1 is an FFTPACK5 auxiliary routine.
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
!    Input, integer ( kind = 4 ) N, the number for which factorization 
!    and other information is needed.
!
!    Output, real ( kind = 4 ) WA(N), trigonometric information.
!
!    Output, real ( kind = 4 ) FAC(15), factorization information.  
!    FAC(1) is N, FAC(2) is NF, the number of factors, and FAC(3:NF+2) are the 
!    factors.
!
  implicit none

  integer ( kind = 4 ) n

  real ( kind = 8 ) arg
  real ( kind = 8 ) argh
  real ( kind = 8 ) argld
  real ( kind = 4 ) fac(15)
  real ( kind = 4 ) fi
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ib
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) ii
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) ipm
  integer ( kind = 4 ) is
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) ld
  integer ( kind = 4 ) nf
  integer ( kind = 4 ) nfm1
  integer ( kind = 4 ) nl
  integer ( kind = 4 ) nq
  integer ( kind = 4 ) nr
  integer ( kind = 4 ) ntry
  real ( kind = 8 ) tpi
  real ( kind = 4 ) wa(n)

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
      fac(nf+2) = real ( ntry, kind = 4 )
      nl = nq
!
!  If 2 is a factor, make sure it appears first in the list of factors.
!
      if ( ntry == 2 ) then
        if ( nf /= 1 ) then
          do i = 2, nf
            ib = nf - i + 2
            fac(ib+2) = fac(ib+1)
          end do
          fac(3) = 2.0E+00
        end if
      end if

    end do

  end do

  fac(1) = real ( n, kind = 4 )
  fac(2) = real ( nf, kind = 4 )
  tpi = 8.0D+00 * atan ( 1.0D+00 )
  argh = tpi / real ( n, kind = 4 )
  is = 0
  nfm1 = nf-1
  l1 = 1

  do k1 = 1, nfm1
    ip = int ( fac(k1+2) )
    ld = 0
    l2 = l1 * ip
    ido = n / l2
    ipm = ip - 1
    do j = 1, ipm
      ld = ld + l1
      i = is
      argld = real ( ld, kind = 4 ) * argh
      fi = 0.0E+00
      do ii = 3, ido, 2
        i = i + 2
        fi = fi + 1.0E+00
        arg = fi * argld
        wa(i-1) = real ( cos ( arg ), kind = 4 )
        wa(i) = real ( sin ( arg ), kind = 4 )
      end do
      is = is + ido
    end do
    l1 = l2
  end do

  return
end
subroutine rfftmb ( lot, jump, n, inc, r, lenr, wsave, lensav, &
  work, lenwrk, ier )

!*****************************************************************************80
!
!! RFFTMB: real single precision backward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    RFFTMB computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a real array.  This transform is referred 
!    to as the backward transform or Fourier synthesis, transforming the
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to RFFTMB followed
!    by a call to RFFTMF (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), real array containing LOT 
!    sequences, each having length N.  R can have any number of dimensions, 
!    but the total number of locations must be at least LENR.  On input, the
!    spectral data to be transformed, on output the physical data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array. 
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1) + 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFTMI before the first call to routine RFFTMF 
!    or RFFTMB for a given transform length N.  
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array. 
!    LENSAV must  be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC, JUMP, N, LOT are not consistent.
!
  implicit none

  integer ( kind = 4 ) lenr
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) r(lenr)
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  logical              xercon

  ier = 0

  if ( lenr < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'rfftmb ', 6 )
    return
  end if

  if ( lensav < n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'rfftmb ', 8 )
    return
  end if

  if ( lenwrk < lot * n ) then
    ier = 3
    call xerfft ( 'rfftmb ', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'rfftmb ', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call mrftb1 ( lot, jump, n, inc, r, work, wsave, wsave(n+1) )

  return
end
subroutine rfftmf ( lot, jump, n, inc, r, lenr, wsave, lensav, &
  work, lenwrk, ier )

!*****************************************************************************80
!
!! RFFTMF: real single precision forward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    RFFTMF computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a real array.  This transform is referred 
!    to as the forward transform or Fourier analysis, transforming the 
!    sequences from physical to spectral space.
!
!    This transform is normalized since a call to RFFTMF followed
!    by a call to RFFTMB (or vice-versa) reproduces the original array
!    within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), real array containing LOT 
!    sequences, each having length N.  R can have any number of dimensions, but 
!    the total number of locations must be at least LENR.  On input, the
!    physical data to be transformed, on output the spectral data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1) + 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFTMI before the first call to routine RFFTMF 
!    or RFFTMB for a given transform length N.  
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC, JUMP, N, LOT are not consistent.
!
  implicit none

  integer ( kind = 4 ) lenr
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) r(lenr)
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  logical              xercon

  ier = 0

  if ( lenr < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'rfftmf ', 6 )
    return
  end if

  if ( lensav < n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'rfftmf ', 8 )
    return
  end if

  if ( lenwrk < lot * n ) then
    ier = 3
    call xerfft ( 'rfftmf ', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'rfftmf ', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call mrftf1 ( lot, jump, n, inc, r, work, wsave, wsave(n+1) )

  return
end
subroutine rfftmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! RFFTMI: initialization for RFFTMB and RFFTMF.
!
!  Discussion:
!
!    RFFTMI initializes array WSAVE for use in its companion routines 
!    RFFTMB and RFFTMF.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    27 March 2005
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), work array containing the prime 
!    factors of N and also containing certain trigonometric 
!    values which will be used in routines RFFTMB or RFFTMF.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) n
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'rfftmi ', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  call mrfti1 ( n, wsave(1), wsave(n+1) )

  return
end
subroutine sinq1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! SINQ1B: real single precision backward sine quarter wave transform, 1D.
!
!  Discussion:
!
!    SINQ1B computes the one-dimensional Fourier transform of a sequence 
!    which is a sine series with odd wave numbers.  This transform is 
!    referred to as the backward transform or Fourier synthesis, 
!    transforming the sequence from spectral to physical space.
!
!    This transform is normalized since a call to SINQ1B followed
!    by a call to SINQ1F (or vice-versa) reproduces the original
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
!    01 April 2005
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
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), on input, the sequence to be 
!    transformed.  On output, the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQ1I before the first call to routine SINQ1F 
!    or SINQ1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to SINQ1F and SINQ1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer ( kind = 4 ) IER, the error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xhold

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sinq1b', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sinq1b', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'sinq1b', 10 )
    return
  end if

  if ( n == 1 ) then
    x(1,1) = 4.0E+00 * x(1,1)
    return
  end if

  ns2 = n / 2

  do k = 2, n, 2
    x(1,k) = -x(1,k)
  end do

  call cosq1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sinq1b', -5 )
    return
  end if

  do k = 1, ns2
    kc = n - k
    xhold = x(1,k)
    x(1,k) = x(1,kc+1)
    x(1,kc+1) = xhold
  end do

  return
end
subroutine sinq1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! SINQ1F: real single precision forward sine quarter wave transform, 1D.
!
!  Discussion:
! 
!    SINQ1F computes the one-dimensional Fourier transform of a sequence 
!    which is a sine series of odd wave numbers.  This transform is 
!    referred to as the forward transform or Fourier analysis, transforming 
!    the sequence from physical to spectral space.
!
!    This transform is normalized since a call to SINQ1F followed
!    by a call to SINQ1B (or vice-versa) reproduces the original
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
!    01 April 2005
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
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), on input, the sequence to be 
!    transformed.  On output, the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQ1I before the first call to routine SINQ1F 
!    or SINQ1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINQ1F and SINQ1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer ( kind = 4 ) IER, the error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xhold

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sinq1f', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sinq1f', 8 )
    return
  end if

  if ( lenwrk < n ) then
    ier = 3
    call xerfft ( 'sinq1f', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  ns2 = n / 2

  do k = 1, ns2
    kc = n - k
    xhold = x(1,k)
    x(1,k) = x(1,kc+1)
    x(1,kc+1) = xhold
  end do

  call cosq1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sinq1f', -5 )
    return
  end if

  do k = 2, n, 2
    x(1,k) = -x(1,k)
  end do

  return
end
subroutine sinq1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! SINQ1I: initialization for SINQ1B and SINQ1F.
!
!  Discussion:
!
!    SINQ1I initializes array WSAVE for use in its companion routines 
!    SINQ1F and SINQ1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    01 April 2005
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
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!   in routines SINQ1B or SINQ1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) n
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sinq1i', 3 )
    return
  end if

  call cosq1i ( n, wsave, lensav, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sinq1i', -5 )
    return
  end if

  return
end
subroutine sinqmb ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

!*****************************************************************************80
!
!! SINQMB: real single precision backward sine quarter wave, multiple vectors.
!
!  Discussion:
!
!    SINQMB computes the one-dimensional Fourier transform of multiple 
!    sequences within a real array, where each of the sequences is a 
!    sine series with odd wave numbers.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to SINQMB followed
!    by a call to SINQMF (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    03 April 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQMI before the first call to routine SINQMF
!    or SINQMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINQMF and SINQMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon
  real ( kind = 4 ) xhold

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sinqmb', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sinqmb', 8 )
    return
  end if

  if ( lenwrk < lot * n ) then
    ier = 3
    call xerfft ( 'sinqmb', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'sinqmb', -1 )
    return
  end if

  lj = ( lot - 1 ) * jump + 1

  if ( n <= 1 ) then
    do m = 1, lj, jump
      x(m,1) = 4.0E+00 * x(m,1)
    end do
    return
  end if

  ns2 = n / 2

  do k = 2, n, 2
    do m = 1, lj, jump
      x(m,k) = -x(m,k)
    end do
  end do

  call cosqmb ( lot, jump, n, inc, x, lenx, wsave, lensav, work, lenwrk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sinqmb', -5 )
    return
  end if

  do k = 1, ns2
    kc = n - k
    do m = 1, lj, jump
      xhold = x(m,k)
      x(m,k) = x(m,kc+1)
      x(m,kc+1) = xhold
    end do
  end do

  return
end
subroutine sinqmf ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

!*****************************************************************************80
!
!! SINQMF: real single precision forward sine quarter wave, multiple vectors.
!
!  Discussion:
!
!    SINQMF computes the one-dimensional Fourier transform of multiple 
!    sequences within a real array, where each sequence is a sine series 
!    with odd wave numbers.  This transform is referred to as the forward
!    transform or Fourier synthesis, transforming the sequences from 
!    spectral to physical space.
!
!    This transform is normalized since a call to SINQMF followed
!    by a call to SINQMB (or vice-versa) reproduces the original
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
!    03 April 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed
!    within array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations,
!    in array R, of the first elements of two consecutive sequences to 
!    be transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQMI before the first call to routine SINQMF 
!    or SINQMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINQMF and SINQMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lj
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon
  real ( kind = 4 ) xhold

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sinqmf', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sinqmf', 8 )
    return
  end if

  if ( lenwrk < lot * n ) then
    ier = 3
    call xerfft ( 'sinqmf', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'sinqmf', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  ns2 = n / 2
  lj = ( lot - 1 ) * jump + 1

  do k = 1, ns2
     kc = n - k
     do m = 1, lj, jump
       xhold = x(m,k)
       x(m,k) = x(m,kc+1)
       x(m,kc+1) = xhold
     end do
  end do

  call cosqmf ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
    lenwrk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sinqmf', -5 )
    return
  end if

  do k = 2, n, 2
     do m = 1, lj, jump
       x(m,k) = -x(m,k)
     end do
  end do

  return
end
subroutine sinqmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! SINQMI: initialization for SINQMB and SINQMF.
!
!  Discussion:
!
!    SINQMI initializes array WSAVE for use in its companion routines 
!    SINQMF and SINQMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    03 April 2005
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!    in routines SINQMB or SINQMF.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) n
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sinqmi', 3 )
    return
  end if

  call cosqmi ( n, wsave, lensav, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sinqmi', -5 )
    return
  end if

  return
end
subroutine sint1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! SINT1B: real single precision backward sine transform, 1D.
!
!  Discussion:
!
!    SINT1B computes the one-dimensional Fourier transform of an odd 
!    sequence within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequence from spectral to physical space.
!
!    This transform is normalized since a call to SINT1B followed
!    by a call to SINT1F (or vice-versa) reproduces the original
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
!    30 March 2005
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
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINT1I before the first call to routine SINT1F 
!    or SINT1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINT1F and SINT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N+2.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sint1b', 6 )
    return
  end if

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sint1b', 8 )
    return
  end if

  if ( lenwrk < 2 * n + 2 ) then
    ier = 3
    call xerfft ( 'sint1b', 10 )
    return
  end if

  call sintb1 ( n, inc, x, wsave, work, work(n+2), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sint1b', -5 )
    return
  end if

  return
end
subroutine sint1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! SINT1F: real single precision forward sine transform, 1D.
!
!  Discussion:
!
!    SINT1F computes the one-dimensional Fourier transform of an odd 
!    sequence within a real array.  This transform is referred to as the 
!    forward transform or Fourier analysis, transforming the sequence
!    from physical to spectral space.
!
!    This transform is normalized since a call to SINT1F followed
!    by a call to SINT1B (or vice-versa) reproduces the original
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
!    30 March 2005
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
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed sequence.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINT1I before the first call to routine SINT1F 
!    or SINT1B for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to SINT1F and SINT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N+2.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)

  ier = 0

  if ( lenx < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'SINT1F', 6 )
    return
  end if

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'SINT1F', 8 )
    return 
  end if

  if ( lenwrk < 2 * n + 2 ) then
    ier = 3
    call xerfft ( 'SINT1F', 10 )
    return
  end if

  call sintf1 ( n, inc, x, wsave, work, work(n+2), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'SINT1F', -5 )
    return
  end if

  return
end
subroutine sint1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! SINT1I: initialization for SINT1B and SINT1F.
!
!  Discussion:
!
!    SINT1I initializes array WSAVE for use in its companion routines 
!    SINT1F and SINT1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    30 March 2005
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
!    transformed.  The transform is most efficient when N+1 is a product 
!    of small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!    in routines SINT1B or SINT1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 4 ) dt
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) pi
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'SINT1I', 3 )
    return
  end if

  pi = 4.0E+00 * atan ( 1.0E+00 )

  if ( n <= 1 ) then
    return
  end if

  ns2 = n / 2
  np1 = n + 1
  dt = pi / real ( np1, kind = 4 )

  do k = 1, ns2
    wsave(k) = 2.0E+00 * sin ( real ( k, kind = 4 ) * dt )
  end do

  lnsv = np1 + int ( log ( real ( np1, kind = 4 ) ) ) + 4

  call rfft1i ( np1, wsave(ns2+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'SINT1I', -5 )
    return
  end if

  return
end
subroutine sintb1 ( n, inc, x, wsave, xh, work, ier )

!*****************************************************************************80
!
!! SINTB1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  real ( kind = 4 ) fnp1s4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lnxh
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) srt3s2
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xh(*)
  real ( kind = 4 ) xhold
 
  ier = 0

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    srt3s2 = sqrt ( 3.0E+00 ) / 2.0E+00 
    xhold = srt3s2 * ( x(1,1) + x(1,2) )
    x(1,2) = srt3s2 * ( x(1,1) - x(1,2) )
    x(1,1) = xhold
    return
  end if

  np1 = n + 1
  ns2 = n / 2

  do k = 1, ns2
    kc = np1 - k
    t1 = x(1,k) - x(1,kc)
    t2 = wsave(k) * ( x(1,k) + x(1,kc) )
    xh(k+1) = t1 + t2
    xh(kc+1) = t2 - t1
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    xh(ns2+2) = 4.0E+00 * x(1,ns2+1)
  end if

  xh(1) = 0.0E+00
  lnxh = np1
  lnsv = np1 + int ( log ( real ( np1, kind = 4 ) ) ) + 4
  lnwk = np1

  call rfft1f ( np1, 1, xh, lnxh, wsave(ns2+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sintb1', -5 )
    return
  end if

  if ( mod ( np1, 2 ) == 0 ) then
    xh(np1) = xh(np1) + xh(np1)
  end if

  fnp1s4 = real ( np1, kind = 4 ) / 4.0E+00
  x(1,1) = fnp1s4 * xh(1)
  dsum = x(1,1)

  do i = 3, n, 2
    x(1,i-1) = fnp1s4 * xh(i)
    dsum = dsum + fnp1s4 * xh(i-1)
    x(1,i) = dsum
  end do

  if ( modn == 0 ) then
    x(1,n) = fnp1s4 * xh(n+1)
  end if

  return
end
subroutine sintf1 ( n, inc, x, wsave, xh, work, ier )

!*****************************************************************************80
!
!! SINTF1 is an FFTPACK5 auxiliary routine.
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
  implicit none

  integer ( kind = 4 ) inc

  real ( kind = 8 ) dsum
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kc
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) lnwk
  integer ( kind = 4 ) lnxh
  integer ( kind = 4 ) modn
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) sfnp1
  real ( kind = 4 ) ssqrt3
  real ( kind = 4 ) t1
  real ( kind = 4 ) t2
  real ( kind = 4 ) work(*)
  real ( kind = 4 ) wsave(*)
  real ( kind = 4 ) x(inc,*)
  real ( kind = 4 ) xh(*)
  real ( kind = 4 ) xhold

  ier = 0

  if ( n < 2 ) then
    return
  end if

  if ( n == 2 ) then
    ssqrt3 = 1.0E+00 / sqrt ( 3.0E+00 )
    xhold = ssqrt3 * ( x(1,1) + x(1,2) )
    x(1,2) = ssqrt3 * ( x(1,1) - x(1,2) )
    x(1,1) = xhold
    return
  end if

  np1 = n + 1
  ns2 = n / 2

  do k = 1, ns2
    kc = np1 - k
    t1 = x(1,k) - x(1,kc)
    t2 = wsave(k) * ( x(1,k) + x(1,kc) )
    xh(k+1) = t1 + t2
    xh(kc+1) = t2 - t1
  end do

  modn = mod ( n, 2 )

  if ( modn /= 0 ) then
    xh(ns2+2) = 4.0E+00 * x(1,ns2+1)
  end if

  xh(1) = 0.0E+00
  lnxh = np1
  lnsv = np1 + int ( log ( real ( np1, kind = 4 ) ) ) + 4
  lnwk = np1

  call rfft1f ( np1, 1, xh, lnxh, wsave(ns2+1), lnsv, work, lnwk, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sintf1', -5 )
    return
  end if

  if ( mod ( np1, 2 ) == 0 ) then
    xh(np1) = xh(np1) + xh(np1)
  end if

  sfnp1 = 1.0E+00 / real ( np1, kind = 4 )
  x(1,1) = 0.5E+00 * xh(1)
  dsum = x(1,1)

  do i = 3, n, 2
    x(1,i-1) = 0.5E+00 * xh(i)
    dsum = dsum + 0.5E+00 * xh(i-1)
    x(1,i) = dsum
  end do

  if ( modn == 0 ) then
    x(1,n) = 0.5E+00 * xh(n+1)
  end if

  return
end
subroutine sintmb ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

!*****************************************************************************80
!
!! SINTMB: real single precision backward sine transform, multiple vectors.
!
!  Discussion:
!
!    SINTMB computes the one-dimensional Fourier transform of multiple 
!    odd sequences within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to SINTMB followed
!    by a call to SINTMF (or vice-versa) reproduces the original
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
!    02 April 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed
!    within the array R.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINTMI before the first call to routine SINTMF 
!    or SINTMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINTMF and SINTMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(2*N+4).
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) iw2
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'SINMTB', 6 )
    return
  end if

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'SINMTB', 8 )
    return
  end if

  if ( lenwrk < lot * ( 2 * n + 4 ) ) then
    ier = 3
    call xerfft ( 'SINMTB', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'SINMTB', -1 )
    return
  end if

  iw1 = lot + lot + 1
  iw2 = iw1 + lot * ( n + 1 )

  call msntb1 ( lot, jump, n, inc, x, wsave, work, work(iw1), work(iw2), ier1 )
 
  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'SINMTB', -5 )
    return
  end if

  return
end
subroutine sintmf ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

!*****************************************************************************80
!
!! SINTMF: real single precision forward sine transform, multiple vectors.
!
!  Discussion:
!
!    SINTMF computes the one-dimensional Fourier transform of multiple 
!    odd sequences within a real array.  This transform is referred to as 
!    the forward transform or Fourier analysis, transforming the sequences 
!    from physical to spectral space.
!
!    This transform is normalized since a call to SINTMF followed
!    by a call to SINTMB (or vice-versa) reproduces the original
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
!    02 April 2005
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be 
!    transformed within.
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, 
!    in array R, of the first elements of two consecutive sequences.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( kind = 4 ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer ( kind = 4 ) LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( kind = 4 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINTMI before the first call to routine SINTMF
!    or SINTMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINTMF and SINTMB with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 4 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(2*N+4).
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) inc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) iw2
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lenx
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 4 ) work(lenwrk)
  real ( kind = 4 ) wsave(lensav)
  real ( kind = 4 ) x(inc,*)
  logical              xercon

  ier = 0

  if ( lenx < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sintmf', 6 )
    return
  end if

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sintmf', 8 )
    return
  end if

  if ( lenwrk < lot * ( 2 * n + 4 ) ) then
    ier = 3
    call xerfft ( 'sintmf', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'sintmf', -1 )
    return
  end if

  iw1 = lot + lot + 1
  iw2 = iw1 + lot * ( n + 1 )

  call msntf1 ( lot, jump, n, inc, x, wsave, work,work(iw1), work(iw2), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sintmf', -5 )
    return
  end if

  return
end
subroutine sintmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! SINTMI: initialization for SINTMB and SINTMF.
!
!  Discussion:
!
!    SINTMI initializes array WSAVE for use in its companion routines 
!    SINTMF and SINTMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    02 April 2005
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( kind = 4 ) WSAVE(LENSAV), containing the prime factors
!    of N and also containing certain trigonometric values which will be used
!    in routines SINTMB or SINTMF.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  real ( kind = 4 ) dt
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lnsv
  integer ( kind = 4 ) n
  integer ( kind = 4 ) np1
  integer ( kind = 4 ) ns2
  real ( kind = 4 ) pi
  real ( kind = 4 ) wsave(lensav)

  ier = 0

  if ( lensav < n / 2 + n + int ( log ( real ( n, kind = 4 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sintmi', 3 )
    return
  end if

  pi = 4.0E+00 * atan ( 1.0E+00 )

  if ( n <= 1 ) then
    return
  end if

  ns2 = n / 2
  np1 = n + 1
  dt = pi / real ( np1, kind = 4 )

  do k = 1, ns2
    wsave(k) = 2.0E+00 * sin ( real ( k, kind = 4 ) * dt )
  end do

  lnsv = np1 + int ( log ( real ( np1, kind = 4 ) ) ) + 4

  call rfftmi ( np1, wsave(ns2+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sintmi', -5 )
    return
  end if

  return
end
function xercon ( inc, jump, n, lot )

!*****************************************************************************80
!
!! XERCON checks INC, JUMP, N and LOT for consistency.
!
!  Discussion:
!
!    Positive integers INC, JUMP, N and LOT are "consistent" if,
!    for any values I1 and I2 < N, and J1 and J2 < LOT,
!
!      I1 * INC + J1 * JUMP = I2 * INC + J2 * JUMP
!
!    can only occur if I1 = I2 and J1 = J2.
!
!    For multiple FFT's to execute correctly, INC, JUMP, N and LOT must
!    be consistent, or else at least one array element will be
!    transformed more than once.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    25 March 2005
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
!    Input, integer ( kind = 4 ) INC, JUMP, N, LOT, the parameters to check.
!
!    Output, logical XERCON, is TRUE if the parameters are consistent.
!
  implicit none

  integer ( kind = 4 ) i
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) j
  integer ( kind = 4 ) jnew
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lcm
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  logical              xercon

  i = inc
  j = jump

  do while ( j /= 0 )
    jnew = mod ( i, j )
    i = j
    j = jnew
  end do
!
!  LCM = least common multiple of INC and JUMP.
!
  lcm = ( inc * jump ) / i

  if ( lcm <= ( n - 1 ) * inc .and. lcm <= ( lot - 1 ) * jump ) then
    xercon = .false.
  else
    xercon = .true.
  end if

  return
end
subroutine xerfft ( srname, info )

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
subroutine z1f2kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F2KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,2)
  real ( kind = 8 ) ch(in2,l1,2,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,1,2)

  if ( 1 < ido .or. na == 1 ) then

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

        ch(2,k,2,i) = wa(i,1,1) * ti2 + wa(i,1,2) * tr2
        ch(1,k,2,i) = wa(i,1,1) * tr2 - wa(i,1,2) * ti2

      end do
    end do

  else

    do k = 1, l1

      chold1      = cc(1,k,1,1) + cc(1,k,1,2)
      cc(1,k,1,2) = cc(1,k,1,1) - cc(1,k,1,2)
      cc(1,k,1,1) = chold1

      chold2      = cc(2,k,1,1) + cc(2,k,1,2)
      cc(2,k,1,2) = cc(2,k,1,1) - cc(2,k,1,2)
      cc(2,k,1,1) = chold2

    end do

  end if

  return
end
subroutine z1f2kf ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F2KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,2)
  real ( kind = 8 ) ch(in2,l1,2,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,1,2)

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

    sn = 1.0D+00 / real ( 2 * l1, kind = 8 )

    do k = 1, l1
      ch(1,k,1,1) = sn * ( cc(1,k,1,1) + cc(1,k,1,2) )
      ch(1,k,2,1) = sn * ( cc(1,k,1,1) - cc(1,k,1,2) )
      ch(2,k,1,1) = sn * ( cc(2,k,1,1) + cc(2,k,1,2) )
      ch(2,k,2,1) = sn * ( cc(2,k,1,1) - cc(2,k,1,2) )
    end do

  else

    sn = 1.0D+00 / real ( 2 * l1, kind = 8 )

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
subroutine z1f3kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F3KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,3)
  real ( kind = 8 ) ch(in2,l1,3,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ), parameter :: taui =  0.866025403784439D+00
  real ( kind = 8 ), parameter :: taur = -0.5D+00
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,2,2)

  if ( 1 < ido .or. na == 1 ) then

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

        ch(2,k,2,i) = wa(i,1,1) * di2 + wa(i,1,2) * dr2
        ch(1,k,2,i) = wa(i,1,1) * dr2 - wa(i,1,2) * di2
        ch(2,k,3,i) = wa(i,2,1) * di3 + wa(i,2,2) * dr3
        ch(1,k,3,i) = wa(i,2,1) * dr3 - wa(i,2,2) * di3

      end do
    end do

  else

    do k = 1, l1

      tr2 = cc(1,k,1,2)+cc(1,k,1,3)
      cr2 = cc(1,k,1,1)+taur*tr2
      cc(1,k,1,1) = cc(1,k,1,1)+tr2
      ti2 = cc(2,k,1,2)+cc(2,k,1,3)
      ci2 = cc(2,k,1,1)+taur*ti2
      cc(2,k,1,1) = cc(2,k,1,1)+ti2
      cr3 = taui*(cc(1,k,1,2)-cc(1,k,1,3))
      ci3 = taui*(cc(2,k,1,2)-cc(2,k,1,3))

      cc(1,k,1,2) = cr2 - ci3
      cc(1,k,1,3) = cr2 + ci3
      cc(2,k,1,2) = ci2 + cr3
      cc(2,k,1,3) = ci2 - cr3

    end do

  end if

  return
end
subroutine z1f3kf ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F3KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,3)
  real ( kind = 8 ) ch(in2,l1,3,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ), parameter :: taui = -0.866025403784439D+00
  real ( kind = 8 ), parameter :: taur = -0.5D+00
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,2,2)

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

    sn = 1.0D+00 / real ( 3 * l1, kind = 8 )

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

    sn = 1.0D+00 / real ( 3 * l1, kind = 8 )

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
subroutine z1f4kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F4KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,4)
  real ( kind = 8 ) ch(in2,l1,4,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) ti1
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) tr1
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) wa(ido,3,2)

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      ti1 = cc(2,k,1,1)-cc(2,k,1,3)
      ti2 = cc(2,k,1,1)+cc(2,k,1,3)
      tr4 = cc(2,k,1,4)-cc(2,k,1,2)
      ti3 = cc(2,k,1,2)+cc(2,k,1,4)
      tr1 = cc(1,k,1,1)-cc(1,k,1,3)
      tr2 = cc(1,k,1,1)+cc(1,k,1,3)
      ti4 = cc(1,k,1,2)-cc(1,k,1,4)
      tr3 = cc(1,k,1,2)+cc(1,k,1,4)
      ch(1,k,1,1) = tr2+tr3
      ch(1,k,3,1) = tr2-tr3
      ch(2,k,1,1) = ti2+ti3
      ch(2,k,3,1) = ti2-ti3
      ch(1,k,2,1) = tr1+tr4
      ch(1,k,4,1) = tr1-tr4
      ch(2,k,2,1) = ti1+ti4
      ch(2,k,4,1) = ti1-ti4
    end do

    do i = 2, ido
      do k = 1, l1
        ti1 = cc(2,k,i,1)-cc(2,k,i,3)
        ti2 = cc(2,k,i,1)+cc(2,k,i,3)
        ti3 = cc(2,k,i,2)+cc(2,k,i,4)
        tr4 = cc(2,k,i,4)-cc(2,k,i,2)
        tr1 = cc(1,k,i,1)-cc(1,k,i,3)
        tr2 = cc(1,k,i,1)+cc(1,k,i,3)
        ti4 = cc(1,k,i,2)-cc(1,k,i,4)
        tr3 = cc(1,k,i,2)+cc(1,k,i,4)
        ch(1,k,1,i) = tr2+tr3
        cr3 = tr2-tr3
        ch(2,k,1,i) = ti2+ti3
        ci3 = ti2-ti3
        cr2 = tr1+tr4
        cr4 = tr1-tr4
        ci2 = ti1+ti4
        ci4 = ti1-ti4

        ch(1,k,2,i) = wa(i,1,1)*cr2-wa(i,1,2)*ci2
        ch(2,k,2,i) = wa(i,1,1)*ci2+wa(i,1,2)*cr2
        ch(1,k,3,i) = wa(i,2,1)*cr3-wa(i,2,2)*ci3
        ch(2,k,3,i) = wa(i,2,1)*ci3+wa(i,2,2)*cr3
        ch(1,k,4,i) = wa(i,3,1)*cr4-wa(i,3,2)*ci4
        ch(2,k,4,i) = wa(i,3,1)*ci4+wa(i,3,2)*cr4

       end do
    end do

  else

    do k = 1, l1
       ti1 = cc(2,k,1,1)-cc(2,k,1,3)
       ti2 = cc(2,k,1,1)+cc(2,k,1,3)
       tr4 = cc(2,k,1,4)-cc(2,k,1,2)
       ti3 = cc(2,k,1,2)+cc(2,k,1,4)
       tr1 = cc(1,k,1,1)-cc(1,k,1,3)
       tr2 = cc(1,k,1,1)+cc(1,k,1,3)
       ti4 = cc(1,k,1,2)-cc(1,k,1,4)
       tr3 = cc(1,k,1,2)+cc(1,k,1,4)
       cc(1,k,1,1) = tr2+tr3
       cc(1,k,1,3) = tr2-tr3
       cc(2,k,1,1) = ti2+ti3
       cc(2,k,1,3) = ti2-ti3
       cc(1,k,1,2) = tr1+tr4
       cc(1,k,1,4) = tr1-tr4
       cc(2,k,1,2) = ti1+ti4
       cc(2,k,1,4) = ti1-ti4
    end do

  end if

  return
end
subroutine z1f4kf ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F4KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,4)
  real ( kind = 8 ) ch(in2,l1,4,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) ti1
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) tr1
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) wa(ido,3,2)

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

    sn = 1.0D+00 / real ( 4 * l1, kind = 8 )

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

    sn = 1.0D+00 / real ( 4 * l1, kind = 8 )

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
subroutine z1f5kb ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F5KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,5)
  real ( kind = 8 ) ch(in2,l1,5,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) ci5
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  real ( kind = 8 ) cr5
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) di4
  real ( kind = 8 ) di5
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  real ( kind = 8 ) dr4
  real ( kind = 8 ) dr5
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) ti5
  real ( kind = 8 ), parameter :: ti11 =  0.9510565162951536D+00
  real ( kind = 8 ), parameter :: ti12 =  0.5877852522924731D+00
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) tr5
  real ( kind = 8 ), parameter :: tr11 =  0.3090169943749474D+00
  real ( kind = 8 ), parameter :: tr12 = -0.8090169943749474D+00
  real ( kind = 8 ) wa(ido,4,2)

  if ( 1 < ido .or. na == 1 ) then

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
        ch(1,k,2,i) = wa(i,1,1)*dr2-wa(i,1,2)*di2
        ch(2,k,2,i) = wa(i,1,1)*di2+wa(i,1,2)*dr2
        ch(1,k,3,i) = wa(i,2,1)*dr3-wa(i,2,2)*di3
        ch(2,k,3,i) = wa(i,2,1)*di3+wa(i,2,2)*dr3
        ch(1,k,4,i) = wa(i,3,1)*dr4-wa(i,3,2)*di4
        ch(2,k,4,i) = wa(i,3,1)*di4+wa(i,3,2)*dr4
        ch(1,k,5,i) = wa(i,4,1)*dr5-wa(i,4,2)*di5
        ch(2,k,5,i) = wa(i,4,1)*di5+wa(i,4,2)*dr5
      end do
    end do

  else

    do k = 1, l1
      ti5 = cc(2,k,1,2)-cc(2,k,1,5)
      ti2 = cc(2,k,1,2)+cc(2,k,1,5)
      ti4 = cc(2,k,1,3)-cc(2,k,1,4)
      ti3 = cc(2,k,1,3)+cc(2,k,1,4)
      tr5 = cc(1,k,1,2)-cc(1,k,1,5)
      tr2 = cc(1,k,1,2)+cc(1,k,1,5)
      tr4 = cc(1,k,1,3)-cc(1,k,1,4)
      tr3 = cc(1,k,1,3)+cc(1,k,1,4)
      chold1 = cc(1,k,1,1)+tr2+tr3
      chold2 = cc(2,k,1,1)+ti2+ti3
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
      cc(1,k,1,2) = cr2-ci5
      cc(1,k,1,5) = cr2+ci5
      cc(2,k,1,2) = ci2+cr5
      cc(2,k,1,3) = ci3+cr4
      cc(1,k,1,3) = cr3-ci4
      cc(1,k,1,4) = cr3+ci4
      cc(2,k,1,4) = ci3-cr4
      cc(2,k,1,5) = ci2-cr5
    end do

  end if

  return
end
subroutine z1f5kf ( ido, l1, na, cc, in1, ch, in2, wa )

!*****************************************************************************80
!
!! Z1F5KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(in1,l1,ido,5)
  real ( kind = 8 ) ch(in2,l1,5,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) ci5
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  real ( kind = 8 ) cr5
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) di4
  real ( kind = 8 ) di5
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  real ( kind = 8 ) dr4
  real ( kind = 8 ) dr5
  integer ( kind = 4 ) i
  integer ( kind = 4 ) k
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) ti5
  real ( kind = 8 ), parameter :: ti11 = -0.9510565162951536D+00
  real ( kind = 8 ), parameter :: ti12 = -0.5877852522924731D+00
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) tr5
  real ( kind = 8 ), parameter :: tr11 =  0.3090169943749474D+00
  real ( kind = 8 ), parameter :: tr12 = -0.8090169943749474D+00
  real ( kind = 8 ) wa(ido,4,2)

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

    sn = 1.0D+00 / real ( 5 * l1, kind = 8 )

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

    sn = 1.0D+00 / real ( 5 * l1, kind = 8 )

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
subroutine z1fgkb ( ido, ip, l1, lid, na, cc, cc1, in1, ch, ch1, in2, wa )

!*****************************************************************************80
!
!! Z1FGKB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 8 ) cc(in1,l1,ip,ido)
  real ( kind = 8 ) cc1(in1,lid,ip)
  real ( kind = 8 ) ch(in2,l1,ido,ip)
  real ( kind = 8 ) ch1(in2,lid,ip)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
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
  real ( kind = 8 ) wa(ido,ip-1,2)
  real ( kind = 8 ) wai
  real ( kind = 8 ) war

  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2
  do ki = 1, lid
    ch1(1,ki,1) = cc1(1,ki,1)
    ch1(2,ki,1) = cc1(2,ki,1)
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      ch1(1,ki,j) =  cc1(1,ki,j) + cc1(1,ki,jc)
      ch1(1,ki,jc) = cc1(1,ki,j) - cc1(1,ki,jc)
      ch1(2,ki,j) =  cc1(2,ki,j) + cc1(2,ki,jc)
      ch1(2,ki,jc) = cc1(2,ki,j) - cc1(2,ki,jc)
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      cc1(1,ki,1) = cc1(1,ki,1)+ch1(1,ki,j)
      cc1(2,ki,1) = cc1(2,ki,1)+ch1(2,ki,j)
    end do
  end do

  do l = 2, ipph

     lc = ipp2 - l
     do ki = 1, lid
       cc1(1,ki,l) = ch1(1,ki,1)+wa(1,l-1,1)*ch1(1,ki,2)
       cc1(1,ki,lc) = wa(1,l-1,2)*ch1(1,ki,ip)
       cc1(2,ki,l) = ch1(2,ki,1)+wa(1,l-1,1)*ch1(2,ki,2)
       cc1(2,ki,lc) = wa(1,l-1,2)*ch1(2,ki,ip)
     end do

     do j = 3, ipph
       jc = ipp2 - j
       idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
       war = wa(1,idlj,1)
       wai = wa(1,idlj,2)
       do ki = 1, lid
         cc1(1,ki,l) = cc1(1,ki,l)+war*ch1(1,ki,j)
         cc1(1,ki,lc) = cc1(1,ki,lc)+wai*ch1(1,ki,jc)
         cc1(2,ki,l) = cc1(2,ki,l)+war*ch1(2,ki,j)
         cc1(2,ki,lc) = cc1(2,ki,lc)+wai*ch1(2,ki,jc)
       end do
     end do

  end do

  if ( 1 < ido .or. na == 1 ) then

    do ki = 1, lid
      ch1(1,ki,1) = cc1(1,ki,1)
      ch1(2,ki,1) = cc1(2,ki,1)
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        ch1(1,ki,j) = cc1(1,ki,j)-cc1(2,ki,jc)
        ch1(1,ki,jc) = cc1(1,ki,j)+cc1(2,ki,jc)
        ch1(2,ki,jc) = cc1(2,ki,j)-cc1(1,ki,jc)
        ch1(2,ki,j) = cc1(2,ki,j)+cc1(1,ki,jc)
      end do
    end do

    if ( ido == 1 ) then
      return
    end if

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
          cc(1,k,j,i) = wa(i,j-1,1)*ch(1,k,i,j) &
                       -wa(i,j-1,2)*ch(2,k,i,j)
          cc(2,k,j,i) = wa(i,j-1,1)*ch(2,k,i,j) &
                       +wa(i,j-1,2)*ch(1,k,i,j)
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        chold1 = cc1(1,ki,j)-cc1(2,ki,jc)
        chold2 = cc1(1,ki,j)+cc1(2,ki,jc)
        cc1(1,ki,j) = chold1
        cc1(2,ki,jc) = cc1(2,ki,j)-cc1(1,ki,jc)
        cc1(2,ki,j) = cc1(2,ki,j)+cc1(1,ki,jc)
        cc1(1,ki,jc) = chold2
      end do
    end do

  end if

  return
end
subroutine z1fgkf ( ido, ip, l1, lid, na, cc, cc1, in1, ch, ch1, in2, wa )

!*****************************************************************************80
!
!! Z1FGKF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 8 ) cc(in1,l1,ip,ido)
  real ( kind = 8 ) cc1(in1,lid,ip)
  real ( kind = 8 ) ch(in2,l1,ido,ip)
  real ( kind = 8 ) ch1(in2,lid,ip)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
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
  real ( kind = 8 ) sn
  real ( kind = 8 ) wa(ido,ip-1,2)
  real ( kind = 8 ) wai
  real ( kind = 8 ) war

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

    sn = 1.0D+00 / real ( ip * l1, kind = 8 )

    do ki = 1, lid
      ch1(1,ki,1) = sn * cc1(1,ki,1)
      ch1(2,ki,1) = sn * cc1(2,ki,1)
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        ch1(1,ki,j) = sn*(cc1(1,ki,j)-cc1(2,ki,jc))
        ch1(2,ki,j) = sn*(cc1(2,ki,j)+cc1(1,ki,jc))
        ch1(1,ki,jc) = sn*(cc1(1,ki,j)+cc1(2,ki,jc))
        ch1(2,ki,jc) = sn*(cc1(2,ki,j)-cc1(1,ki,jc))
      end do
    end do

  else

    sn = 1.0D+00 / real ( ip * l1, kind = 8 )

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
subroutine z1fm1b ( n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! Z1FM1B is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  complex ( kind = 8 ) c(*)
  real ( kind = 8 ) ch(*)
  real ( kind = 8 ) fac(*)
  real ( kind = 8 ) fnf
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
  real ( kind = 8 ) wa(*)

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
      call z1f2kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 2 ) then
      call z1f2kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 3 ) then
      call z1f3kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 4 ) then
      call z1f3kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 5 ) then
      call z1f4kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 6 ) then
      call z1f4kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 7 ) then
      call z1f5kb ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
    else if ( nbr == 8 ) then
      call z1f5kb ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
    else if ( nbr == 9 ) then
      call z1fgkb ( ido, ip, l1, lid, na, c, c, inc2, ch, ch, 2, wa(iw) )
    else if ( nbr == 10 ) then
      call z1fgkb ( ido, ip, l1, lid, na, ch, ch, 2, c, c, inc2, wa(iw) )
    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ( ido + ido )

    if ( ip <= 5 ) then
      na = 1 - na
    end if

  end do

  return
end
subroutine z1fm1f ( n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! Z1FM1F is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  complex ( kind = 8 ) c(*)
  real ( kind = 8 ) ch(*)
  real ( kind = 8 ) fac(*)
  real ( kind = 8 ) fnf
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
  real ( kind = 8 ) wa(*)

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
       call z1f2kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 2 ) then
       call z1f2kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 3 ) then
       call z1f3kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 4 ) then
       call z1f3kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 5 ) then
       call z1f4kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 6 ) then
       call z1f4kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 7 ) then
       call z1f5kf ( ido, l1, na, c, inc2, ch, 2, wa(iw) )
     else if ( nbr == 8 ) then
       call z1f5kf ( ido, l1, na, ch, 2, c, inc2, wa(iw) )
     else if ( nbr == 9 ) then
       call z1fgkf ( ido, ip, l1, lid, na, c, c, inc2, ch, ch, 1, wa(iw) )
     else if ( nbr == 10 ) then
       call z1fgkf ( ido, ip, l1, lid, na, ch, ch, 2, c, c, inc2, wa(iw) )
     end if

     l1 = l2
     iw = iw + ( ip - 1 ) * ( ido + ido )

     if ( ip <= 5 ) then
       na = 1 - na
     end if

  end do

  return
end
subroutine zfft1b ( n, inc, c, lenc, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! ZFFT1B: complex double precision backward fast Fourier transform, 1D.
!
!  Discussion:
!
!    ZFFT1B computes the one-dimensional Fourier transform of a single 
!    periodic sequence within a complex array.  This transform is referred 
!    to as the backward transform or Fourier synthesis, transforming the
!    sequence from spectral to physical space.
!
!    This transform is normalized since a call to ZFFT1B followed
!    by a call to ZFFT1F (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input/output, complex ( kind = 8 ) C(LENC) containing the sequence to 
!    be transformed.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least INC*(N-1) + 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to ZFFT1I before the first call to routine ZFFT1F 
!    or ZFFT1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to ZFFT1F and ZFFT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    1, input parameter LENC not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 8 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lenc < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'ZFFT1B', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'ZFFT1B', 8 )
    return
  end if

  if ( lenwrk < 2 * n ) then
    ier = 3
    call xerfft ( 'ZFFT1B', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call z1fm1b ( n, inc, c, work, wsave, wsave(iw1), wsave(iw1+1) )

  return
end
subroutine zfft1f ( n, inc, c, lenc, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! ZFFT1F: complex double precision forward fast Fourier transform, 1D.
!
!  Discussion:
!
!    ZFFT1F computes the one-dimensional Fourier transform of a single 
!    periodic sequence within a complex array.  This transform is referred 
!    to as the forward transform or Fourier analysis, transforming the 
!    sequence from physical to spectral space.
!
!    This transform is normalized since a call to ZFFT1F followed
!    by a call to ZFFT1B (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input/output, complex ( kind = 8 ) C(LENC) containing the sequence to 
!    be transformed.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least INC*(N-1) + 1.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to ZFFT1I before the first call to routine ZFFT1F
!    or ZFFT1B for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to ZFFT1F and ZFFT1B with the same N.
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK).
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
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 8 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lenc < inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'ZFFT1F', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'ZFFT1F', 8 )
    return
  end if

  if ( lenwrk < 2 * n ) then
    ier = 3
    call xerfft ( 'ZFFT1F', 10 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call z1fm1f ( n, inc, c, work, wsave, wsave(iw1), wsave(iw1+1) )

  return
end
subroutine zfft1i ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! ZFFT1I: initialization for ZFFT1B and ZFFT1F.
!
!  Discussion:
!
!    ZFFT1I initializes array WSAVE for use in its companion routines 
!    ZFFT1B and ZFFT1F.  Routine ZFFT1I must be called before the first 
!    call to ZFFT1B or ZFFT1F, and after whenever the value of integer 
!    N changes.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Output, real ( kind = 8 ) WSAVE(LENSAV), containing the prime factors 
!    of N and  also containing certain trigonometric values which will be used 
!    in routines ZFFT1B or ZFFT1F.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.

  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'ZFFT1I', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call r8_mcfti1 ( n, wsave, wsave(iw1), wsave(iw1+1) )

  return
end
subroutine zfft2b ( ldim, l, m, c, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! ZFFT2B: complex double precision backward fast Fourier transform, 2D.
!
!  Discussion:
!
!    ZFFT2B computes the two-dimensional discrete Fourier transform of a 
!    complex periodic array.  This transform is known as the backward 
!    transform or Fourier synthesis, transforming from spectral to 
!    physical space.  Routine ZFFT2B is normalized, in that a call to 
!    ZFFT2B followed by a call to ZFFT2F (or vice-versa) reproduces the 
!    original array within roundoff error. 
!
!    On 10 May 2010, this code was modified by changing the value
!    of an index into the WSAVE array.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    10 May 2010
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) LDIM, the first dimension of C. 
!
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension of the two-dimensional complex array C.  The value 
!    of L must be less than or equal to that of LDIM.  The transform is
!    most efficient when L is a product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension of the two-dimensional complex array C.  The 
!    transform is most efficient when M is a product of small primes. 
!
!    Input/output, complex ( kind = 8 ) C(LDIM,M), on intput, the array of two
!    dimensions containing the (L,M) subarray to be transformed.  On output, 
!    the transformed data.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV). WSAVE's contents must be 
!    initialized with a call to ZFFT2I before the first call to routine ZFFT2F
!    or ZFFT2B with transform lengths L and M.  WSAVE's contents may be
!    re-used for subsequent calls to ZFFT2F and ZFFT2B with the same 
!    transform lengths L and M. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array. 
!    LENSAV must be at least 2*(L+M) + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8. 
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*L*M. 
!
!    Output, integer ( kind = 4 ) IER, the error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    5, input parameter LDIM < L;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) ldim
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 8 ) c(ldim,m)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) l
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( ldim < l ) then
    ier = 5
    call xerfft ( 'ZFFT2B', -2 )
    return
  end if

  if ( lensav < 2 * l + int ( log ( real ( l, kind = 8 ) ) ) + &
                2 * m + int ( log ( real ( m, kind = 8 ) ) ) + 8 ) then
    ier = 2
    call xerfft ( 'ZFFT2B', 6 )
    return
  end if

  if ( lenwrk < 2 * l * m ) then
    ier = 3
    call xerfft ( 'ZFFT2B', 8 )
    return
  end if
!
!  Transform the X lines of the C array.
!
!  On 10 May 2010, the value of IW was modified.
!
  iw = 2 * l + int ( log ( real ( l, kind = 8 ) ) ) + 5

  call zfftmb ( l, 1, m, ldim, c, (l-1)+ldim*(m-1) +1, &
    wsave(iw), 2*m + int(log(real ( m, kind = 8 ))) + 4, work, 2*l*m, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'ZFFT2B', -5 )
    return
  end if
!
!  Transform the Y lines of the C array.
!
  iw = 1
  call zfftmb ( m, ldim, l, 1, c, (m-1)*ldim + l, wsave(iw), &
    2*l + int(log(real ( l, kind = 8 ))) + 4, work, 2*m*l, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'ZFFT2B', -5 )
    return
  end if

  return
end
subroutine zfft2f ( ldim, l, m, c, wsave, lensav, work, lenwrk, ier )

!*****************************************************************************80
!
!! ZFFT2F: complex double precision forward fast Fourier transform, 2D.
!
!  Discussion:
!
!    ZFFT2F computes the two-dimensional discrete Fourier transform of 
!    a complex periodic array. This transform is known as the forward 
!    transform or Fourier analysis, transforming from physical to 
!    spectral space. Routine ZFFT2F is normalized, in that a call to 
!    ZFFT2F followed by a call to ZFFT2B (or vice-versa) reproduces the 
!    original array within roundoff error. 
!
!    On 10 May 2010, this code was modified by changing the value
!    of an index into the WSAVE array.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    10 May 2010
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) LDIM, the first dimension of the array C. 
!
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension of the two-dimensional complex array C.  The value 
!    of L must be less than or equal to that of LDIM.  The transform is most 
!    efficient when L is a product of small primes. 
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension of the two-dimensional complex array C.  The 
!    transform is most efficient when M is a product of small primes. 
!
!    Input/output, complex ( kind = 8 ) C(LDIM,M), on input, the array of two 
!    dimensions containing the (L,M) subarray to be transformed.  On output, 
!    the transformed data.
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV). WSAVE's contents must be 
!    initialized with a call to ZFFT2I before the first call to routine ZFFT2F 
!    or ZFFT2B with transform lengths L and M.  WSAVE's contents may be re-used 
!    for subsequent calls to ZFFT2F and ZFFT2B having those same 
!    transform lengths. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*(L+M) + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8. 
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*L*M.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit; 
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    5, input parameter LDIM < L;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) ldim
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 8 ) c(ldim,m)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) l
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( ldim < l ) then
    ier = 5
    call xerfft ( 'ZFFT2F', -2 )
    return
  end if

  if ( lensav < 2 * l + int ( log ( real ( l, kind = 8 ) ) ) + &
                2 * m + int ( log ( real ( m, kind = 8 ) ) ) + 8 ) then
    ier = 2
    call xerfft ( 'ZFFT2F', 6 )
    return
  end if

  if ( lenwrk < 2 * l * m ) then
    ier = 3
    call xerfft ( 'ZFFT2F', 8 )
    return
  end if
!
!  Transform the X lines of the C array.
!
!  On 10 May 2010, the value of IW was modified.
!
  iw = 2 * l + int ( log ( real ( l, kind = 8 ) ) ) + 5

  call zfftmf ( l, 1, m, ldim, c, (l-1) + ldim*(m-1) +1, wsave(iw), &
    2*m + int(log(real ( m, kind = 8 ))) + 4, work, 2*l*m, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'ZFFT2F', -5 )
    return
  end if
!
!  Transform the Y lines of the C array.
!
  iw = 1

  call zfftmf ( m, ldim, l, 1, c, (m-1)*ldim + l, wsave(iw), &
    2*l + int(log(real ( l, kind = 8 ))) + 4, work, 2*m*l, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'ZFFT2F', -5 )
    return
  end if

  return
end
subroutine zfft2i ( l, m, wsave, lensav, ier )

!*****************************************************************************80
!
!! ZFFT2I: initialization for ZFFT2B and ZFFT2F.
!
!  Discussion:
!
!    ZFFT2I initializes real array WSAVE for use in its companion 
!    routines ZFFT2F and ZFFT2B for computing two-dimensional fast 
!    Fourier transforms of complex data.  Prime factorizations of L and M,
!    together with tabulations of the trigonometric functions, are 
!    computed and stored in array WSAVE.
!
!    On 10 May 2010, this code was modified by changing the value
!    of an index into the WSAVE array.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) L, the number of elements to be transformed 
!    in the first dimension.  The transform is most efficient when L is a 
!    product of small primes.
!
!    Input, integer ( kind = 4 ) M, the number of elements to be transformed 
!    in the second dimension.  The transform is most efficient when M is a 
!    product of small primes. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*(L+M) + INT(LOG(REAL(L))) 
!    + INT(LOG(REAL(M))) + 8.
!
!    Output, real ( kind = 8 ) WSAVE(LENSAV), contains the prime factors of L 
!    and M, and also certain trigonometric values which will be used in 
!    routines ZFFT2B or ZFFT2F. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) ier1
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) l
  integer ( kind = 4 ) m
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * l + int ( log ( real ( l, kind = 8 ) ) ) + &
                2 * m + int ( log ( real ( m, kind = 8 ) ) ) + 8 ) then
    ier = 2
    call xerfft ( 'ZFFT2I', 4 )
    return
  end if

  call zfftmi ( l, wsave(1), 2*l + int(log(real ( l, kind = 8 ))) + 4, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'ZFFT2I', -5 )
    return
  end if
!
!  On 10 May 2010, the value of IW was modified.
!
  iw = 2 * l + int ( log ( real ( l, kind = 8 ) ) ) + 5

  call zfftmi ( m, wsave(iw), 2*m + int(log(real ( m, kind = 8 ))) + 4, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'ZFFT2I', -5 )
    return
  end if

  return
end
subroutine zfftmb ( lot, jump, n, inc, c, lenc, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! ZFFTMB: complex double precision backward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    ZFFTMB computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a complex array.  This transform is referred
!    to as the backward transform or Fourier synthesis, transforming the
!    sequences from spectral to physical space.  This transform is 
!    normalized since a call to ZFFTMF followed by a call to ZFFTMB (or
!    vice-versa) reproduces the original array within roundoff error. 
!
!    The parameters INC, JUMP, N and LOT are consistent if equality 
!    I1*INC + J1*JUMP = I2*INC + J2*JUMP for I1,I2 < N and J1,J2 < LOT 
!    implies I1=I2 and J1=J2.  For multiple FFTs to execute correctly, 
!    input variables INC, JUMP, N and LOT must be consistent, otherwise 
!    at least one array element mistakenly is transformed more than once.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be transformed
!    within array C. 
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, in
!    array C, of the first elements of two consecutive sequences to 
!    be transformed. 
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array C, of two consecutive elements within the same sequence to be 
!    transformed. 
!
!    Input/output, complex ( kind = 8 ) C(LENC), an array containing LOT 
!    sequences, each having length N, to be transformed.  C can have any 
!    number of dimensions, but the total number of locations must be at least 
!    LENC.  On output, C contains the transformed sequences.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least (LOT-1)*JUMP + INC*(N-1) + 1. 
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to ZFFTMI before the first call to routine ZFFTMF 
!    or ZFFTMB for a given transform length N. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*LOT*N.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit
!    1, input parameter LENC not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC, JUMP, N, LOT are not consistent. 
!
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 8 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  logical              xercon

  ier = 0

  if ( lenc < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'ZFFTMB', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'ZFFTMB', 8 )
    return
  end if

  if ( lenwrk < 2 * lot * n ) then
    ier = 3
    call xerfft ( 'ZFFTMB', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'ZFFTMB', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call zmfm1b ( lot, jump, n, inc, c, work, wsave, wsave(iw1), &
    wsave(iw1+1) )

  return
end
subroutine zfftmf ( lot, jump, n, inc, c, lenc, wsave, lensav, work, &
  lenwrk, ier )

!*****************************************************************************80
!
!! ZFFTMF: complex double precision forward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    ZFFTMF computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a complex array. This transform is referred 
!    to as the forward transform or Fourier analysis, transforming the 
!    sequences from physical to spectral space. This transform is 
!    normalized since a call to ZFFTMF followed by a call to ZFFTMB 
!    (or vice-versa) reproduces the original array within roundoff error. 
!
!    The parameters integers INC, JUMP, N and LOT are consistent if equality
!    I1*INC + J1*JUMP = I2*INC + J2*JUMP for I1,I2 < N and J1,J2 < LOT 
!    implies I1=I2 and J1=J2. For multiple FFTs to execute correctly, 
!    input variables INC, JUMP, N and LOT must be consistent, otherwise 
!    at least one array element mistakenly is transformed more than once.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) LOT, the number of sequences to be 
!    transformed within array C. 
!
!    Input, integer ( kind = 4 ) JUMP, the increment between the locations, 
!    in array C, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes. 
!
!    Input, integer ( kind = 4 ) INC, the increment between the locations, in 
!    array C, of two consecutive elements within the same sequence to be 
!    transformed.
!
!    Input/output, complex ( kind = 8 ) C(LENC), array containing LOT sequences,
!    each having length N, to be transformed.  C can have any number of 
!    dimensions, but the total number of locations must be at least LENC.
!
!    Input, integer ( kind = 4 ) LENC, the dimension of the C array.  
!    LENC must be at least (LOT-1)*JUMP + INC*(N-1) + 1. 
!
!    Input, real ( kind = 8 ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to ZFFTMI before the first call to routine ZFFTMF 
!    or ZFFTMB for a given transform length N. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4. 
!
!    Workspace, real ( kind = 8 ) WORK(LENWRK). 
!
!    Input, integer ( kind = 4 ) LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*LOT*N. 
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0 successful exit;
!    1 input parameter LENC not big enough;
!    2 input parameter LENSAV not big enough;
!    3 input parameter LENWRK not big enough;
!    4 input parameters INC, JUMP, N, LOT are not consistent.
!
  implicit none

  integer ( kind = 4 ) lenc
  integer ( kind = 4 ) lensav
  integer ( kind = 4 ) lenwrk

  complex ( kind = 8 ) c(lenc)
  integer ( kind = 4 ) ier
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  real ( kind = 8 ) work(lenwrk)
  real ( kind = 8 ) wsave(lensav)
  logical              xercon

  ier = 0

  if ( lenc < ( lot - 1 ) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'ZFFTMF', 6 )
    return
  end if

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'ZFFTMF', 8 )
    return
  end if

  if ( lenwrk < 2 * lot * n ) then
    ier = 3
    call xerfft ( 'ZFFTMF', 10 )
    return
  end if

  if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'ZFFTMF', -1 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1

  call zmfm1f ( lot, jump, n, inc, c, work, wsave, wsave(iw1), &
    wsave(iw1+1) )

  return
end
subroutine zfftmi ( n, wsave, lensav, ier )

!*****************************************************************************80
!
!! ZFFTMI: initialization for ZFFTMB and ZFFTMF.
!
!  Discussion:
!
!    ZFFTMI initializes array WSAVE for use in its companion routines 
!    ZFFTMB and ZFFTMF.  ZFFTMI must be called before the first call 
!    to ZFFTMB or ZFFTMF, and after whenever the value of integer N changes. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
!    Input, integer ( kind = 4 ) N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes. 
!
!    Input, integer ( kind = 4 ) LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4. 
!
!    Output, real ( kind = 8 ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used in
!    routines ZFFTMB or ZFFTMF.
!
!    Output, integer ( kind = 4 ) IER, error flag.
!    0, successful exit; 
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer ( kind = 4 ) lensav

  integer ( kind = 4 ) ier
  integer ( kind = 4 ) iw1
  integer ( kind = 4 ) n
  real ( kind = 8 ) wsave(lensav)

  ier = 0

  if ( lensav < 2 * n + int ( log ( real ( n, kind = 8 ) ) ) + 4 ) then
    ier = 2
    call xerfft ( 'cfftmi ', 3 )
    return
  end if

  if ( n == 1 ) then
    return
  end if

  iw1 = n + n + 1
  call r8_mcfti1 ( n, wsave, wsave(iw1), wsave(iw1+1) )

  return
end
subroutine zmf2kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF2KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,2)
  real ( kind = 8 ) ch(2,in2,l1,2,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,1,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1) + cc(1,m1,k,1,2)
        ch(1,m2,k,2,1) = cc(1,m1,k,1,1) - cc(1,m1,k,1,2)
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1) + cc(2,m1,k,1,2)
        ch(2,m2,k,2,1) = cc(2,m1,k,1,1) - cc(2,m1,k,1,2)
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+cc(1,m1,k,i,2)
          tr2 = cc(1,m1,k,i,1)-cc(1,m1,k,i,2)
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+cc(2,m1,k,i,2)
          ti2 = cc(2,m1,k,i,1)-cc(2,m1,k,i,2)

          ch(2,m2,k,2,i) = wa(i,1,1) * ti2 + wa(i,1,2) * tr2
          ch(1,m2,k,2,i) = wa(i,1,1) * tr2 - wa(i,1,2) * ti2

        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1

        chold1         = cc(1,m1,k,1,1) + cc(1,m1,k,1,2)
        cc(1,m1,k,1,2) = cc(1,m1,k,1,1) - cc(1,m1,k,1,2)
        cc(1,m1,k,1,1) = chold1

        chold2         = cc(2,m1,k,1,1) + cc(2,m1,k,1,2)
        cc(2,m1,k,1,2) = cc(2,m1,k,1,1) - cc(2,m1,k,1,2)
        cc(2,m1,k,1,1) = chold2

      end do
    end do

  end if

  return
end
subroutine zmf2kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF2KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,2)
  real ( kind = 8 ) ch(2,in2,l1,2,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,1,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+cc(1,m1,k,1,2)
        ch(1,m2,k,2,1) = cc(1,m1,k,1,1)-cc(1,m1,k,1,2)
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+cc(2,m1,k,1,2)
        ch(2,m2,k,2,1) = cc(2,m1,k,1,1)-cc(2,m1,k,1,2)
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+cc(1,m1,k,i,2)
          tr2 = cc(1,m1,k,i,1)-cc(1,m1,k,i,2)
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+cc(2,m1,k,i,2)
          ti2 = cc(2,m1,k,i,1)-cc(2,m1,k,i,2)
          ch(2,m2,k,2,i) = wa(i,1,1)*ti2-wa(i,1,2)*tr2
          ch(1,m2,k,2,i) = wa(i,1,1)*tr2+wa(i,1,2)*ti2
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0D+00 / real ( 2 * l1, kind = 8 )

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch(1,m2,k,1,1) = sn * ( cc(1,m1,k,1,1) + cc(1,m1,k,1,2) )
        ch(1,m2,k,2,1) = sn * ( cc(1,m1,k,1,1) - cc(1,m1,k,1,2) )
        ch(2,m2,k,1,1) = sn * ( cc(2,m1,k,1,1) + cc(2,m1,k,1,2) )
        ch(2,m2,k,2,1) = sn * ( cc(2,m1,k,1,1) - cc(2,m1,k,1,2) )
      end do
    end do

  else

    sn = 1.0D+00 / real ( 2 * l1, kind = 8 )

    do k = 1, l1
      do m1 = 1, m1d, im1

        chold1         = sn * ( cc(1,m1,k,1,1) + cc(1,m1,k,1,2) )
        cc(1,m1,k,1,2) = sn * ( cc(1,m1,k,1,1) - cc(1,m1,k,1,2) )
        cc(1,m1,k,1,1) = chold1

        chold2         = sn * ( cc(2,m1,k,1,1) + cc(2,m1,k,1,2) )
        cc(2,m1,k,1,2) = sn * ( cc(2,m1,k,1,1) - cc(2,m1,k,1,2) )
        cc(2,m1,k,1,1) = chold2

      end do
    end do

  end if

  return
end
subroutine zmf3kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF3KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,3)
  real ( kind = 8 ) ch(2,in2,l1,3,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ), parameter :: taui =  0.866025403784439D+00
  real ( kind = 8 ), parameter :: taur = -0.5D+00
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,2,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1

        m2 = m2 + im2

        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2

        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2

        cr3 = taui * (cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui * (cc(2,m1,k,1,2)-cc(2,m1,k,1,3))

        ch(1,m2,k,2,1) = cr2-ci3
        ch(1,m2,k,3,1) = cr2+ci3
        ch(2,m2,k,2,1) = ci2+cr3
        ch(2,m2,k,3,1) = ci2-cr3

      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,3)
          cr2 = cc(1,m1,k,i,1)+taur*tr2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,3)
          ci2 = cc(2,m1,k,i,1)+taur*ti2
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2
          cr3 = taui*(cc(1,m1,k,i,2)-cc(1,m1,k,i,3))
          ci3 = taui*(cc(2,m1,k,i,2)-cc(2,m1,k,i,3))
          dr2 = cr2-ci3
          dr3 = cr2+ci3
          di2 = ci2+cr3
          di3 = ci2-cr3
          ch(2,m2,k,2,i) = wa(i,1,1)*di2+wa(i,1,2)*dr2
          ch(1,m2,k,2,i) = wa(i,1,1)*dr2-wa(i,1,2)*di2
          ch(2,m2,k,3,i) = wa(i,2,1)*di3+wa(i,2,2)*dr3
          ch(1,m2,k,3,i) = wa(i,2,1)*dr3-wa(i,2,2)*di3
        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        cc(1,m1,k,1,1) = cc(1,m1,k,1,1)+tr2
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        cc(2,m1,k,1,1) = cc(2,m1,k,1,1)+ti2
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        cc(1,m1,k,1,2) = cr2-ci3
        cc(1,m1,k,1,3) = cr2+ci3
        cc(2,m1,k,1,2) = ci2+cr3
        cc(2,m1,k,1,3) = ci2-cr3
      end do
    end do

  end if

  return
end
subroutine zmf3kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF3KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,3)
  real ( kind = 8 ) ch(2,in2,l1,3,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ), parameter :: taui = -0.866025403784439D+00
  real ( kind = 8 ), parameter :: taur = -0.5D+00
  real ( kind = 8 ) ti2
  real ( kind = 8 ) tr2
  real ( kind = 8 ) wa(ido,2,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        ch(1,m2,k,2,1) = cr2-ci3
        ch(1,m2,k,3,1) = cr2+ci3
        ch(2,m2,k,2,1) = ci2+cr3
        ch(2,m2,k,3,1) = ci2-cr3
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,3)
          cr2 = cc(1,m1,k,i,1)+taur*tr2
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,3)
          ci2 = cc(2,m1,k,i,1)+taur*ti2
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2
          cr3 = taui*(cc(1,m1,k,i,2)-cc(1,m1,k,i,3))
          ci3 = taui*(cc(2,m1,k,i,2)-cc(2,m1,k,i,3))
          dr2 = cr2-ci3
          dr3 = cr2+ci3
          di2 = ci2+cr3
          di3 = ci2-cr3
          ch(2,m2,k,2,i) = wa(i,1,1)*di2-wa(i,1,2)*dr2
          ch(1,m2,k,2,i) = wa(i,1,1)*dr2+wa(i,1,2)*di2
          ch(2,m2,k,3,i) = wa(i,2,1)*di3-wa(i,2,2)*dr3
          ch(1,m2,k,3,i) = wa(i,2,1)*dr3+wa(i,2,2)*di3
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0D+00 / real ( 3 * l1, kind = 8 )

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        ch(1,m2,k,1,1) = sn*(cc(1,m1,k,1,1)+tr2)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        ch(2,m2,k,1,1) = sn*(cc(2,m1,k,1,1)+ti2)
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        ch(1,m2,k,2,1) = sn*(cr2-ci3)
        ch(1,m2,k,3,1) = sn*(cr2+ci3)
        ch(2,m2,k,2,1) = sn*(ci2+cr3)
        ch(2,m2,k,3,1) = sn*(ci2-cr3)
      end do
    end do

  else

    sn = 1.0D+00 / real ( 3 * l1, kind = 8 )

    do k = 1, l1
      do m1 = 1, m1d, im1
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,3)
        cr2 = cc(1,m1,k,1,1)+taur*tr2
        cc(1,m1,k,1,1) = sn*(cc(1,m1,k,1,1)+tr2)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,3)
        ci2 = cc(2,m1,k,1,1)+taur*ti2
        cc(2,m1,k,1,1) = sn*(cc(2,m1,k,1,1)+ti2)
        cr3 = taui*(cc(1,m1,k,1,2)-cc(1,m1,k,1,3))
        ci3 = taui*(cc(2,m1,k,1,2)-cc(2,m1,k,1,3))
        cc(1,m1,k,1,2) = sn*(cr2-ci3)
        cc(1,m1,k,1,3) = sn*(cr2+ci3)
        cc(2,m1,k,1,2) = sn*(ci2+cr3)
        cc(2,m1,k,1,3) = sn*(ci2-cr3)
      end do
    end do

  end if

  return
end
subroutine zmf4kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF4KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,4)
  real ( kind = 8 ) ch(2,in2,l1,4,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) ti1
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) tr1
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) wa(ido,3,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,4)-cc(2,m1,k,1,2)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,2)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = tr2+tr3
        ch(1,m2,k,3,1) = tr2-tr3
        ch(2,m2,k,1,1) = ti2+ti3
        ch(2,m2,k,3,1) = ti2-ti3
        ch(1,m2,k,2,1) = tr1+tr4
        ch(1,m2,k,4,1) = tr1-tr4
        ch(2,m2,k,2,1) = ti1+ti4
        ch(2,m2,k,4,1) = ti1-ti4
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti1 = cc(2,m1,k,i,1)-cc(2,m1,k,i,3)
          ti2 = cc(2,m1,k,i,1)+cc(2,m1,k,i,3)
          ti3 = cc(2,m1,k,i,2)+cc(2,m1,k,i,4)
          tr4 = cc(2,m1,k,i,4)-cc(2,m1,k,i,2)
          tr1 = cc(1,m1,k,i,1)-cc(1,m1,k,i,3)
          tr2 = cc(1,m1,k,i,1)+cc(1,m1,k,i,3)
          ti4 = cc(1,m1,k,i,2)-cc(1,m1,k,i,4)
          tr3 = cc(1,m1,k,i,2)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = tr2+tr3
          cr3 = tr2-tr3
          ch(2,m2,k,1,i) = ti2+ti3
          ci3 = ti2-ti3
          cr2 = tr1+tr4
          cr4 = tr1-tr4
          ci2 = ti1+ti4
          ci4 = ti1-ti4
          ch(1,m2,k,2,i) = wa(i,1,1)*cr2-wa(i,1,2)*ci2
          ch(2,m2,k,2,i) = wa(i,1,1)*ci2+wa(i,1,2)*cr2
          ch(1,m2,k,3,i) = wa(i,2,1)*cr3-wa(i,2,2)*ci3
          ch(2,m2,k,3,i) = wa(i,2,1)*ci3+wa(i,2,2)*cr3
          ch(1,m2,k,4,i) = wa(i,3,1)*cr4-wa(i,3,2)*ci4
          ch(2,m2,k,4,i) = wa(i,3,1)*ci4+wa(i,3,2)*cr4
        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,4)-cc(2,m1,k,1,2)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,2)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        cc(1,m1,k,1,1) = tr2+tr3
        cc(1,m1,k,1,3) = tr2-tr3
        cc(2,m1,k,1,1) = ti2+ti3
        cc(2,m1,k,1,3) = ti2-ti3
        cc(1,m1,k,1,2) = tr1+tr4
        cc(1,m1,k,1,4) = tr1-tr4
        cc(2,m1,k,1,2) = ti1+ti4
        cc(2,m1,k,1,4) = ti1-ti4
      end do
    end do

  end if

  return
end
subroutine zmf4kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF4KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,4)
  real ( kind = 8 ) ch(2,in2,l1,4,ido)
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) ti1
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) tr1
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) wa(ido,3,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,2)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,4)-cc(1,m1,k,1,2)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = tr2+tr3
        ch(1,m2,k,3,1) = tr2-tr3
        ch(2,m2,k,1,1) = ti2+ti3
        ch(2,m2,k,3,1) = ti2-ti3
        ch(1,m2,k,2,1) = tr1+tr4
        ch(1,m2,k,4,1) = tr1-tr4
        ch(2,m2,k,2,1) = ti1+ti4
        ch(2,m2,k,4,1) = ti1-ti4
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti1 = cc(2,m1,k,i,1)-cc(2,m1,k,i,3)
          ti2 = cc(2,m1,k,i,1)+cc(2,m1,k,i,3)
          ti3 = cc(2,m1,k,i,2)+cc(2,m1,k,i,4)
          tr4 = cc(2,m1,k,i,2)-cc(2,m1,k,i,4)
          tr1 = cc(1,m1,k,i,1)-cc(1,m1,k,i,3)
          tr2 = cc(1,m1,k,i,1)+cc(1,m1,k,i,3)
          ti4 = cc(1,m1,k,i,4)-cc(1,m1,k,i,2)
          tr3 = cc(1,m1,k,i,2)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = tr2+tr3
          cr3 = tr2-tr3
          ch(2,m2,k,1,i) = ti2+ti3
          ci3 = ti2-ti3
          cr2 = tr1+tr4
          cr4 = tr1-tr4
          ci2 = ti1+ti4
          ci4 = ti1-ti4
          ch(1,m2,k,2,i) = wa(i,1,1)*cr2+wa(i,1,2)*ci2
          ch(2,m2,k,2,i) = wa(i,1,1)*ci2-wa(i,1,2)*cr2
          ch(1,m2,k,3,i) = wa(i,2,1)*cr3+wa(i,2,2)*ci3
          ch(2,m2,k,3,i) = wa(i,2,1)*ci3-wa(i,2,2)*cr3
          ch(1,m2,k,4,i) = wa(i,3,1)*cr4+wa(i,3,2)*ci4
          ch(2,m2,k,4,i) = wa(i,3,1)*ci4-wa(i,3,2)*cr4
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0D+00 / real ( 4 * l1, kind = 8 )

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,2)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,4)-cc(1,m1,k,1,2)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = sn*(tr2+tr3)
        ch(1,m2,k,3,1) = sn*(tr2-tr3)
        ch(2,m2,k,1,1) = sn*(ti2+ti3)
        ch(2,m2,k,3,1) = sn*(ti2-ti3)
        ch(1,m2,k,2,1) = sn*(tr1+tr4)
        ch(1,m2,k,4,1) = sn*(tr1-tr4)
        ch(2,m2,k,2,1) = sn*(ti1+ti4)
        ch(2,m2,k,4,1) = sn*(ti1-ti4)
      end do
    end do

  else

    sn = 1.0D+00 / real ( 4 * l1, kind = 8 )

    do k = 1, l1
      do m1 = 1, m1d, im1
        ti1 = cc(2,m1,k,1,1)-cc(2,m1,k,1,3)
        ti2 = cc(2,m1,k,1,1)+cc(2,m1,k,1,3)
        tr4 = cc(2,m1,k,1,2)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,2)+cc(2,m1,k,1,4)
        tr1 = cc(1,m1,k,1,1)-cc(1,m1,k,1,3)
        tr2 = cc(1,m1,k,1,1)+cc(1,m1,k,1,3)
        ti4 = cc(1,m1,k,1,4)-cc(1,m1,k,1,2)
        tr3 = cc(1,m1,k,1,2)+cc(1,m1,k,1,4)
        cc(1,m1,k,1,1) = sn*(tr2+tr3)
        cc(1,m1,k,1,3) = sn*(tr2-tr3)
        cc(2,m1,k,1,1) = sn*(ti2+ti3)
        cc(2,m1,k,1,3) = sn*(ti2-ti3)
        cc(1,m1,k,1,2) = sn*(tr1+tr4)
        cc(1,m1,k,1,4) = sn*(tr1-tr4)
        cc(2,m1,k,1,2) = sn*(ti1+ti4)
        cc(2,m1,k,1,4) = sn*(ti1-ti4)
      end do
    end do

  end if

  return
end
subroutine zmf5kb ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF5KB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,5)
  real ( kind = 8 ) ch(2,in2,l1,5,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) ci5
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  real ( kind = 8 ) cr5
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) di4
  real ( kind = 8 ) di5
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  real ( kind = 8 ) dr4
  real ( kind = 8 ) dr5
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) ti5
  real ( kind = 8 ), parameter :: ti11 =  0.9510565162951536D+00
  real ( kind = 8 ), parameter :: ti12 =  0.5877852522924731D+00
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) tr5
  real ( kind = 8 ), parameter :: tr11 =  0.3090169943749474D+00
  real ( kind = 8 ), parameter :: tr12 = -0.8090169943749474D+00
  real ( kind = 8 ) wa(ido,4,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido .or. na == 1 ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2+tr3
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2+ti3
        cr2 = cc(1,m1,k,1,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,m1,k,1,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,m1,k,1,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,m1,k,1,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        ch(1,m2,k,2,1) = cr2-ci5
        ch(1,m2,k,5,1) = cr2+ci5
        ch(2,m2,k,2,1) = ci2+cr5
        ch(2,m2,k,3,1) = ci3+cr4
        ch(1,m2,k,3,1) = cr3-ci4
        ch(1,m2,k,4,1) = cr3+ci4
        ch(2,m2,k,4,1) = ci3-cr4
        ch(2,m2,k,5,1) = ci2-cr5
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti5 = cc(2,m1,k,i,2)-cc(2,m1,k,i,5)
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,5)
          ti4 = cc(2,m1,k,i,3)-cc(2,m1,k,i,4)
          ti3 = cc(2,m1,k,i,3)+cc(2,m1,k,i,4)
          tr5 = cc(1,m1,k,i,2)-cc(1,m1,k,i,5)
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,5)
          tr4 = cc(1,m1,k,i,3)-cc(1,m1,k,i,4)
          tr3 = cc(1,m1,k,i,3)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2+tr3
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2+ti3
          cr2 = cc(1,m1,k,i,1)+tr11*tr2+tr12*tr3
          ci2 = cc(2,m1,k,i,1)+tr11*ti2+tr12*ti3
          cr3 = cc(1,m1,k,i,1)+tr12*tr2+tr11*tr3
          ci3 = cc(2,m1,k,i,1)+tr12*ti2+tr11*ti3
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
          ch(1,m2,k,2,i) = wa(i,1,1) * dr2 - wa(i,1,2) * di2
          ch(2,m2,k,2,i) = wa(i,1,1) * di2 + wa(i,1,2) * dr2
          ch(1,m2,k,3,i) = wa(i,2,1) * dr3 - wa(i,2,2) * di3
          ch(2,m2,k,3,i) = wa(i,2,1) * di3 + wa(i,2,2) * dr3
          ch(1,m2,k,4,i) = wa(i,3,1) * dr4 - wa(i,3,2) * di4
          ch(2,m2,k,4,i) = wa(i,3,1) * di4 + wa(i,3,2) * dr4
          ch(1,m2,k,5,i) = wa(i,4,1) * dr5 - wa(i,4,2) * di5
          ch(2,m2,k,5,i) = wa(i,4,1) * di5 + wa(i,4,2) * dr5
        end do
      end do
    end do

  else

    do k = 1, l1
      do m1 = 1, m1d, im1
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)

        chold1 = cc(1,m1,k,1,1) + tr2 + tr3
        chold2 = cc(2,m1,k,1,1) + ti2 + ti3

        cr2 = cc(1,m1,k,1,1) + tr11 * tr2 + tr12 * tr3
        ci2 = cc(2,m1,k,1,1) + tr11 * ti2 + tr12 * ti3
        cr3 = cc(1,m1,k,1,1) + tr12 * tr2 + tr11 * tr3
        ci3 = cc(2,m1,k,1,1) + tr12 * ti2 + tr11 * ti3

        cc(1,m1,k,1,1) = chold1
        cc(2,m1,k,1,1) = chold2

        cr5 = ti11*tr5 + ti12*tr4
        ci5 = ti11*ti5 + ti12*ti4
        cr4 = ti12*tr5 - ti11*tr4
        ci4 = ti12*ti5 - ti11*ti4
        cc(1,m1,k,1,2) = cr2-ci5
        cc(1,m1,k,1,5) = cr2+ci5
        cc(2,m1,k,1,2) = ci2+cr5
        cc(2,m1,k,1,3) = ci3+cr4
        cc(1,m1,k,1,3) = cr3-ci4
        cc(1,m1,k,1,4) = cr3+ci4
        cc(2,m1,k,1,4) = ci3-cr4
        cc(2,m1,k,1,5) = ci2-cr5
      end do
    end do

  end if

  return
end
subroutine zmf5kf ( lot, ido, l1, na, cc, im1, in1, ch, im2, in2, wa )

!*****************************************************************************80
!
!! ZMF5KF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) l1

  real ( kind = 8 ) cc(2,in1,l1,ido,5)
  real ( kind = 8 ) ch(2,in2,l1,5,ido)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  real ( kind = 8 ) ci2
  real ( kind = 8 ) ci3
  real ( kind = 8 ) ci4
  real ( kind = 8 ) ci5
  real ( kind = 8 ) cr2
  real ( kind = 8 ) cr3
  real ( kind = 8 ) cr4
  real ( kind = 8 ) cr5
  real ( kind = 8 ) di2
  real ( kind = 8 ) di3
  real ( kind = 8 ) di4
  real ( kind = 8 ) di5
  real ( kind = 8 ) dr2
  real ( kind = 8 ) dr3
  real ( kind = 8 ) dr4
  real ( kind = 8 ) dr5
  integer ( kind = 4 ) i
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) k
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) ti2
  real ( kind = 8 ) ti3
  real ( kind = 8 ) ti4
  real ( kind = 8 ) ti5
  real ( kind = 8 ), parameter :: ti11 = -0.9510565162951536D+00
  real ( kind = 8 ), parameter :: ti12 = -0.5877852522924731D+00
  real ( kind = 8 ) tr2
  real ( kind = 8 ) tr3
  real ( kind = 8 ) tr4
  real ( kind = 8 ) tr5
  real ( kind = 8 ), parameter :: tr11 =  0.3090169943749474D+00
  real ( kind = 8 ), parameter :: tr12 = -0.8090169943749474D+00
  real ( kind = 8 ) wa(ido,4,2)

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2

  if ( 1 < ido ) then

    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = cc(1,m1,k,1,1)+tr2+tr3
        ch(2,m2,k,1,1) = cc(2,m1,k,1,1)+ti2+ti3
        cr2 = cc(1,m1,k,1,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,m1,k,1,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,m1,k,1,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,m1,k,1,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        ch(1,m2,k,2,1) = cr2-ci5
        ch(1,m2,k,5,1) = cr2+ci5
        ch(2,m2,k,2,1) = ci2+cr5
        ch(2,m2,k,3,1) = ci3+cr4
        ch(1,m2,k,3,1) = cr3-ci4
        ch(1,m2,k,4,1) = cr3+ci4
        ch(2,m2,k,4,1) = ci3-cr4
        ch(2,m2,k,5,1) = ci2-cr5
      end do
    end do

    do i = 2, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ti5 = cc(2,m1,k,i,2)-cc(2,m1,k,i,5)
          ti2 = cc(2,m1,k,i,2)+cc(2,m1,k,i,5)
          ti4 = cc(2,m1,k,i,3)-cc(2,m1,k,i,4)
          ti3 = cc(2,m1,k,i,3)+cc(2,m1,k,i,4)
          tr5 = cc(1,m1,k,i,2)-cc(1,m1,k,i,5)
          tr2 = cc(1,m1,k,i,2)+cc(1,m1,k,i,5)
          tr4 = cc(1,m1,k,i,3)-cc(1,m1,k,i,4)
          tr3 = cc(1,m1,k,i,3)+cc(1,m1,k,i,4)
          ch(1,m2,k,1,i) = cc(1,m1,k,i,1)+tr2+tr3
          ch(2,m2,k,1,i) = cc(2,m1,k,i,1)+ti2+ti3
          cr2 = cc(1,m1,k,i,1)+tr11*tr2+tr12*tr3
          ci2 = cc(2,m1,k,i,1)+tr11*ti2+tr12*ti3
          cr3 = cc(1,m1,k,i,1)+tr12*tr2+tr11*tr3
          ci3 = cc(2,m1,k,i,1)+tr12*ti2+tr11*ti3
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
          ch(1,m2,k,2,i) = wa(i,1,1)*dr2+wa(i,1,2)*di2
          ch(2,m2,k,2,i) = wa(i,1,1)*di2-wa(i,1,2)*dr2
          ch(1,m2,k,3,i) = wa(i,2,1)*dr3+wa(i,2,2)*di3
          ch(2,m2,k,3,i) = wa(i,2,1)*di3-wa(i,2,2)*dr3
          ch(1,m2,k,4,i) = wa(i,3,1)*dr4+wa(i,3,2)*di4
          ch(2,m2,k,4,i) = wa(i,3,1)*di4-wa(i,3,2)*dr4
          ch(1,m2,k,5,i) = wa(i,4,1)*dr5+wa(i,4,2)*di5
          ch(2,m2,k,5,i) = wa(i,4,1)*di5-wa(i,4,2)*dr5
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0D+00 / real ( 5 * l1, kind = 8 )
 
    do k = 1, l1
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ti5 = cc(2,m1,k,1,2)-cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2)+cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3)-cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3)+cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2)-cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2)+cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3)-cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3)+cc(1,m1,k,1,4)
        ch(1,m2,k,1,1) = sn*(cc(1,m1,k,1,1)+tr2+tr3)
        ch(2,m2,k,1,1) = sn*(cc(2,m1,k,1,1)+ti2+ti3)
        cr2 = cc(1,m1,k,1,1)+tr11*tr2+tr12*tr3
        ci2 = cc(2,m1,k,1,1)+tr11*ti2+tr12*ti3
        cr3 = cc(1,m1,k,1,1)+tr12*tr2+tr11*tr3
        ci3 = cc(2,m1,k,1,1)+tr12*ti2+tr11*ti3
        cr5 = ti11*tr5+ti12*tr4
        ci5 = ti11*ti5+ti12*ti4
        cr4 = ti12*tr5-ti11*tr4
        ci4 = ti12*ti5-ti11*ti4
        ch(1,m2,k,2,1) = sn*(cr2-ci5)
        ch(1,m2,k,5,1) = sn*(cr2+ci5)
        ch(2,m2,k,2,1) = sn*(ci2+cr5)
        ch(2,m2,k,3,1) = sn*(ci3+cr4)
        ch(1,m2,k,3,1) = sn*(cr3-ci4)
        ch(1,m2,k,4,1) = sn*(cr3+ci4)
        ch(2,m2,k,4,1) = sn*(ci3-cr4)
        ch(2,m2,k,5,1) = sn*(ci2-cr5)
      end do
    end do

  else

    sn = 1.0D+00 / real ( 5 * l1, kind = 8 )

    do k = 1, l1
      do m1 = 1, m1d, im1

        ti5 = cc(2,m1,k,1,2) - cc(2,m1,k,1,5)
        ti2 = cc(2,m1,k,1,2) + cc(2,m1,k,1,5)
        ti4 = cc(2,m1,k,1,3) - cc(2,m1,k,1,4)
        ti3 = cc(2,m1,k,1,3) + cc(2,m1,k,1,4)
        tr5 = cc(1,m1,k,1,2) - cc(1,m1,k,1,5)
        tr2 = cc(1,m1,k,1,2) + cc(1,m1,k,1,5)
        tr4 = cc(1,m1,k,1,3) - cc(1,m1,k,1,4)
        tr3 = cc(1,m1,k,1,3) + cc(1,m1,k,1,4)

        chold1 = sn * ( cc(1,m1,k,1,1) + tr2 + tr3 )
        chold2 = sn * ( cc(2,m1,k,1,1) + ti2 + ti3 )

        cr2 = cc(1,m1,k,1,1) + tr11 * tr2 + tr12 * tr3
        ci2 = cc(2,m1,k,1,1) + tr11 * ti2 + tr12 * ti3
        cr3 = cc(1,m1,k,1,1) + tr12 * tr2 + tr11 * tr3
        ci3 = cc(2,m1,k,1,1) + tr12 * ti2 + tr11 * ti3

        cc(1,m1,k,1,1) = chold1
        cc(2,m1,k,1,1) = chold2

        cr5 = ti11 * tr5 + ti12 * tr4
        ci5 = ti11 * ti5 + ti12 * ti4
        cr4 = ti12 * tr5 - ti11 * tr4
        ci4 = ti12 * ti5 - ti11 * ti4

        cc(1,m1,k,1,2) = sn * ( cr2 - ci5 )
        cc(1,m1,k,1,5) = sn * ( cr2 + ci5 )
        cc(2,m1,k,1,2) = sn * ( ci2 + cr5 )
        cc(2,m1,k,1,3) = sn * ( ci3 + cr4 )
        cc(1,m1,k,1,3) = sn * ( cr3 - ci4 )
        cc(1,m1,k,1,4) = sn * ( cr3 + ci4 )
        cc(2,m1,k,1,4) = sn * ( ci3 - cr4 )
        cc(2,m1,k,1,5) = sn * ( ci2 - cr5 )

      end do
    end do

  end if

  return
end
subroutine zmfgkb ( lot, ido, ip, l1, lid, na, cc, cc1, im1, in1, &
  ch, ch1, im2, in2, wa )

!*****************************************************************************80
!
!! ZMFGKB is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 8 ) cc(2,in1,l1,ip,ido)
  real ( kind = 8 ) cc1(2,in1,lid,ip)
  real ( kind = 8 ) ch(2,in2,l1,ido,ip)
  real ( kind = 8 ) ch1(2,in2,lid,ip)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idlj
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) j
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) ki
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) wa(ido,ip-1,2)
  real ( kind = 8 ) wai
  real ( kind = 8 ) war

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  do ki = 1, lid
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
      ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,j) =  cc1(1,m1,ki,j) + cc1(1,m1,ki,jc)
        ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) - cc1(1,m1,ki,jc)
        ch1(2,m2,ki,j) =  cc1(2,m1,ki,j) + cc1(2,m1,ki,jc)
        ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(2,m1,ki,jc)
      end do
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,1) = cc1(1,m1,ki,1) + ch1(1,m2,ki,j)
        cc1(2,m1,ki,1) = cc1(2,m1,ki,1) + ch1(2,m2,ki,j)
      end do
    end do
  end do

  do l = 2, ipph

    lc = ipp2 - l
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2

        cc1(1,m1,ki,l)  = ch1(1,m2,ki,1) + wa(1,l-1,1) * ch1(1,m2,ki,2)
        cc1(1,m1,ki,lc) =                  wa(1,l-1,2) * ch1(1,m2,ki,ip)
        cc1(2,m1,ki,l)  = ch1(2,m2,ki,1) + wa(1,l-1,1) * ch1(2,m2,ki,2)
        cc1(2,m1,ki,lc) =                  wa(1,l-1,2) * ch1(2,m2,ki,ip)

      end do
    end do

    do j = 3, ipph
      jc = ipp2 - j
      idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
      war = wa(1,idlj,1)
      wai = wa(1,idlj,2)
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc1(1,m1,ki,l)  = cc1(1,m1,ki,l)  + war * ch1(1,m2,ki,j)
          cc1(1,m1,ki,lc) = cc1(1,m1,ki,lc) + wai * ch1(1,m2,ki,jc)
          cc1(2,m1,ki,l)  = cc1(2,m1,ki,l)  + war * ch1(2,m2,ki,j)
          cc1(2,m1,ki,lc) = cc1(2,m1,ki,lc) + wai * ch1(2,m2,ki,jc)
        end do
      end do
    end do

  end do

  if( 1 < ido .or. na == 1 ) then

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
        ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch1(1,m2,ki,j)  = cc1(1,m1,ki,j) - cc1(2,m1,ki,jc)
          ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) + cc1(2,m1,ki,jc)
          ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(1,m1,ki,jc)
          ch1(2,m2,ki,j)  = cc1(2,m1,ki,j) + cc1(1,m1,ki,jc)
        end do
      end do
    end do

    if ( ido == 1 ) then
      return
    end if

    do i = 1, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,1,i) = ch(1,m2,k,i,1)
          cc(2,m1,k,1,i) = ch(2,m2,k,i,1)
        end do
      end do
    end do

    do j = 2, ip
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,j,1) = ch(1,m2,k,1,j)
          cc(2,m1,k,j,1) = ch(2,m2,k,1,j)
        end do
      end do
    end do

    do j = 2, ip
      do i = 2, ido
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            cc(1,m1,k,j,i) = wa(i,j-1,1) * ch(1,m2,k,i,j) &
                           - wa(i,j-1,2) * ch(2,m2,k,i,j)
            cc(2,m1,k,j,i) = wa(i,j-1,1) * ch(2,m2,k,i,j) &
                           + wa(i,j-1,2) * ch(1,m2,k,i,j)
          end do
        end do
      end do
    end do

  else

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        do m1 = 1, m1d, im1

          chold1         = cc1(1,m1,ki,j) - cc1(2,m1,ki,jc)
          chold2         = cc1(1,m1,ki,j) + cc1(2,m1,ki,jc)
          cc1(1,m1,ki,j) = chold1

          cc1(2,m1,ki,jc) = cc1(2,m1,ki,j) - cc1(1,m1,ki,jc)
          cc1(2,m1,ki,j)  = cc1(2,m1,ki,j) + cc1(1,m1,ki,jc)
          cc1(1,m1,ki,jc) = chold2

        end do
      end do
    end do

  end if

  return
end
subroutine zmfgkf ( lot, ido, ip, l1, lid, na, cc, cc1, im1, in1, &
  ch, ch1, im2, in2, wa )

!*****************************************************************************80
!
!! ZMFGKF is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  integer ( kind = 4 ) ido
  integer ( kind = 4 ) in1
  integer ( kind = 4 ) in2
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) lid

  real ( kind = 8 ) cc(2,in1,l1,ip,ido)
  real ( kind = 8 ) cc1(2,in1,lid,ip)
  real ( kind = 8 ) ch(2,in2,l1,ido,ip)
  real ( kind = 8 ) ch1(2,in2,lid,ip)
  real ( kind = 8 ) chold1
  real ( kind = 8 ) chold2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) idlj
  integer ( kind = 4 ) im1
  integer ( kind = 4 ) im2
  integer ( kind = 4 ) ipp2
  integer ( kind = 4 ) ipph
  integer ( kind = 4 ) j 
  integer ( kind = 4 ) jc
  integer ( kind = 4 ) k
  integer ( kind = 4 ) ki
  integer ( kind = 4 ) l
  integer ( kind = 4 ) lc
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) m1
  integer ( kind = 4 ) m1d
  integer ( kind = 4 ) m2
  integer ( kind = 4 ) m2s
  integer ( kind = 4 ) na
  real ( kind = 8 ) sn
  real ( kind = 8 ) wa(ido,ip-1,2)
  real ( kind = 8 ) wai
  real ( kind = 8 ) war

  m1d = ( lot - 1 ) * im1 + 1
  m2s = 1 - im2
  ipp2 = ip + 2
  ipph = ( ip + 1 ) / 2

  do ki = 1, lid
    m2 = m2s
    do m1 = 1, m1d, im1
      m2 = m2 + im2
      ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
      ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
    end do
  end do

  do j = 2, ipph
    jc = ipp2 - j
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,j) =  cc1(1,m1,ki,j) + cc1(1,m1,ki,jc)
        ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) - cc1(1,m1,ki,jc)
        ch1(2,m2,ki,j) =  cc1(2,m1,ki,j) + cc1(2,m1,ki,jc)
        ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(2,m1,ki,jc)
      end do
    end do
  end do

  do j = 2, ipph
    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,1) = cc1(1,m1,ki,1) + ch1(1,m2,ki,j)
        cc1(2,m1,ki,1) = cc1(2,m1,ki,1) + ch1(2,m2,ki,j)
      end do
    end do
  end do

  do l = 2, ipph

    lc = ipp2 - l

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,l)  = ch1(1,m2,ki,1) + wa(1,l-1,1) * ch1(1,m2,ki,2)
        cc1(1,m1,ki,lc) =                - wa(1,l-1,2) * ch1(1,m2,ki,ip)
        cc1(2,m1,ki,l)  = ch1(2,m2,ki,1) + wa(1,l-1,1) * ch1(2,m2,ki,2)
        cc1(2,m1,ki,lc) =                - wa(1,l-1,2) * ch1(2,m2,ki,ip)
      end do
    end do

    do j = 3, ipph
      jc = ipp2 - j
      idlj = mod ( ( l - 1 ) * ( j - 1 ), ip )
      war = wa(1,idlj,1)
      wai = -wa(1,idlj,2)
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc1(1,m1,ki,l)  = cc1(1,m1,ki,l)  + war * ch1(1,m2,ki,j)
          cc1(1,m1,ki,lc) = cc1(1,m1,ki,lc) + wai * ch1(1,m2,ki,jc)
          cc1(2,m1,ki,l)  = cc1(2,m1,ki,l)  + war * ch1(2,m2,ki,j)
          cc1(2,m1,ki,lc) = cc1(2,m1,ki,lc) + wai * ch1(2,m2,ki,jc)
        end do
      end do
    end do

  end do

  if ( 1 < ido ) then

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,1) = cc1(1,m1,ki,1)
        ch1(2,m2,ki,1) = cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch1(1,m2,ki,j)  = cc1(1,m1,ki,j) - cc1(2,m1,ki,jc)
          ch1(2,m2,ki,j)  = cc1(2,m1,ki,j) + cc1(1,m1,ki,jc)
          ch1(1,m2,ki,jc) = cc1(1,m1,ki,j) + cc1(2,m1,ki,jc)
          ch1(2,m2,ki,jc) = cc1(2,m1,ki,j) - cc1(1,m1,ki,jc)
        end do
      end do
    end do

    do i = 1, ido
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,1,i) = ch(1,m2,k,i,1)
          cc(2,m1,k,1,i) = ch(2,m2,k,i,1)
        end do
      end do
    end do

    do j = 2, ip
      do k = 1, l1
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          cc(1,m1,k,j,1) = ch(1,m2,k,1,j)
          cc(2,m1,k,j,1) = ch(2,m2,k,1,j)
        end do
      end do
    end do

    do j = 2, ip
      do i = 2, ido
        do k = 1, l1
          m2 = m2s
          do m1 = 1, m1d, im1
            m2 = m2 + im2
            cc(1,m1,k,j,i) = wa(i,j-1,1) * ch(1,m2,k,i,j) &
                           + wa(i,j-1,2) * ch(2,m2,k,i,j)
            cc(2,m1,k,j,i) = wa(i,j-1,1) * ch(2,m2,k,i,j) &
                           - wa(i,j-1,2) * ch(1,m2,k,i,j)
          end do
        end do
      end do
    end do

  else if ( na == 1 ) then

    sn = 1.0D+00 / real ( ip * l1, kind = 8 )

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        ch1(1,m2,ki,1) = sn * cc1(1,m1,ki,1)
        ch1(2,m2,ki,1) = sn * cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        m2 = m2s
        do m1 = 1, m1d, im1
          m2 = m2 + im2
          ch1(1,m2,ki,j)  = sn * ( cc1(1,m1,ki,j) - cc1(2,m1,ki,jc) )
          ch1(2,m2,ki,j)  = sn * ( cc1(2,m1,ki,j) + cc1(1,m1,ki,jc) )
          ch1(1,m2,ki,jc) = sn * ( cc1(1,m1,ki,j) + cc1(2,m1,ki,jc) )
          ch1(2,m2,ki,jc) = sn * ( cc1(2,m1,ki,j) - cc1(1,m1,ki,jc) )
        end do
      end do
    end do

  else

    sn = 1.0D+00 / real ( ip * l1, kind = 8 )

    do ki = 1, lid
      m2 = m2s
      do m1 = 1, m1d, im1
        m2 = m2 + im2
        cc1(1,m1,ki,1) = sn * cc1(1,m1,ki,1)
        cc1(2,m1,ki,1) = sn * cc1(2,m1,ki,1)
      end do
    end do

    do j = 2, ipph
      jc = ipp2 - j
      do ki = 1, lid
        do m1 = 1, m1d, im1
          chold1 = sn * ( cc1(1,m1,ki,j) - cc1(2,m1,ki,jc) )
          chold2 = sn * ( cc1(1,m1,ki,j) + cc1(2,m1,ki,jc) )
          cc1(1,m1,ki,j) = chold1
          cc1(2,m1,ki,jc) = sn * ( cc1(2,m1,ki,j) - cc1(1,m1,ki,jc) )
          cc1(2,m1,ki,j)  = sn * ( cc1(2,m1,ki,j) + cc1(1,m1,ki,jc) )
          cc1(1,m1,ki,jc) = chold2
        end do
      end do
    end do

  end if

  return
end
subroutine zmfm1b ( lot, jump, n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! ZMFM1B is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  complex ( kind = 8 ) c(*)
  real ( kind = 8 ) ch(*)
  real ( kind = 8 ) fac(*)
  real ( kind = 8 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) lid
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nbr
  integer ( kind = 4 ) nf
  real ( kind = 8 ) wa(*)

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
      call zmf2kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 2 ) then
      call zmf2kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 3 ) then
      call zmf3kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 4 ) then
      call zmf3kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 5 ) then
      call zmf4kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 6 ) then
      call zmf4kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 7 ) then
      call zmf5kb ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 8 ) then
      call zmf5kb ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 9 ) then
      call zmfgkb ( lot, ido, ip, l1, lid, na, c, c, jump, inc, ch, ch, &
        1, lot, wa(iw) )
    else if ( nbr == 10 ) then
      call zmfgkb ( lot, ido, ip, l1, lid, na, ch, ch, 1, lot, c, c, &
        jump, inc, wa(iw) )
    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ( ido + ido )

    if ( ip <= 5 ) then
      na = 1 - na
    end if

  end do

  return
end
subroutine zmfm1f ( lot, jump, n, inc, c, ch, wa, fnf, fac )

!*****************************************************************************80
!
!! ZMFM1F is an FFTPACK5 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    26 Ausust 2009
!
!  Author:
!
!    Original complex single precision by Paul Swarztrauber, Richard Valent.
!    Complex double precision version by John Burkardt.
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
  implicit none

  complex ( kind = 8 ) c(*)
  real ( kind = 8 ) ch(*)
  real ( kind = 8 ) fac(*)
  real ( kind = 8 ) fnf
  integer ( kind = 4 ) ido
  integer ( kind = 4 ) inc
  integer ( kind = 4 ) ip
  integer ( kind = 4 ) iw
  integer ( kind = 4 ) jump
  integer ( kind = 4 ) k1
  integer ( kind = 4 ) l1
  integer ( kind = 4 ) l2
  integer ( kind = 4 ) lid
  integer ( kind = 4 ) lot
  integer ( kind = 4 ) n
  integer ( kind = 4 ) na
  integer ( kind = 4 ) nbr
  integer ( kind = 4 ) nf
  real ( kind = 8 ) wa(*)

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
      call zmf2kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 2 ) then
      call zmf2kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 3 ) then
      call zmf3kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 4 ) then
      call zmf3kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 5 ) then
      call zmf4kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 6 ) then
      call zmf4kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 7 ) then
      call zmf5kf ( lot, ido, l1, na, c, jump, inc, ch, 1, lot, wa(iw) )
    else if ( nbr == 8 ) then
      call zmf5kf ( lot, ido, l1, na, ch, 1, lot, c, jump, inc, wa(iw) )
    else if ( nbr == 9 ) then
      call zmfgkf ( lot, ido, ip, l1, lid, na, c, c, jump, inc, ch, ch, &
        1, lot, wa(iw) )
    else if ( nbr == 10 ) then
      call zmfgkf ( lot, ido, ip, l1, lid, na, ch, ch, 1, lot, c, c, &
        jump, inc, wa(iw) )
    end if

    l1 = l2
    iw = iw + ( ip - 1 ) * ( ido + ido )

    if ( ip <= 5 ) then
      na = 1 - na
    end if

  end do

  return
end