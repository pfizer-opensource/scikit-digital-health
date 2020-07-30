! -*- f90 -*-
include "complex/qpssf.f90"
include "complex/qpssf2.f90"
include "complex/qpssf3.f90"
include "complex/qpssf4.f90"
include "complex/qpssf5.f90"

subroutine qcftf1(n, c, ch, wa, ifac)
    use iso_fortran_env
    integer(int64) :: n, ifac(1)
    real(real64) :: c(1), ch(1), wa(1)
    ! local
    integer(int64) :: nf, na, l1, iw, k1, ip, l2, ido, idot, idl1
    integer(int64) :: ix3, ix2
    
    nf = ifac(2)
    na = 0
    l1 = 1
    iw = 1
    
    do k1 = 1, nf
        ip = ifac(k1 + 2)
        l2 = ip * l1
        ido = n / l2
        idot = ido + ido
        idl1 = idot * l1
        
        if (ip .ne. 4) then
            if (ip .ne. 2) then
                if (ip .ne. 3) then
                    if (ip .ne. 5) then
                        if (na .ne. 0) then
                            call qpssf(nac, idot, ip, l1, idl1, ch, ch, ch, c, c, wa(iw))
                        else
                            call qpssf(nac, idot, ip, l1, idl1, c, c, c, ch, ch, wa(iw))
                        end if
                        if (nac .ne. 0) na = 1 - na
                    else
                        ix2 = iw + idot
                        ix3 = ix2 + idot
                        ix4 = ix3 + idot
                        if (na .ne. 0) then
                            call qpssf5(idot, l1, ch, c, wa(iw), wa(ix2), wa(ix3), wa(ix4))
                        else
                            call qpssf5(idot, l1, c, ch, wa(iw), wa(ix2), wa(ix3), wa(ix4))
                        end if
                    end if
                else
                    ix2 = iw + idot
                    if (na .ne. 0) then
                        call qpssf3(idot, la, ch, c, wa(iw), wa(ix2))
                    else
                        call qpssf3(idot, l1, c, ch, wa(iw), wa(ix2))
                    end if
                end if
            else if (na .ne. 0) then
                call qpssf2(idot, l1, ch, c, wa(iw))
            else
                call qpssf2(idot, l1, c, ch, wa(iw))
            end if        
            
            na = 1 - na
            
        else
            ix2 = iw + idot
            ix3 = ix2 + idot
            
            if (na .ne. 0) then
                call qpssf4(idot, l1, ch, c, wa(iw), wa(ix2), wa(ix3))
            else
                call qpssf4(idot, l1, c, ch, wa(iw), wa(ix2), wa(ix3))
            end if
            na = 1 - na
        end if
        l1 = l2
        iw = iw + (ip - 1) * idot
    end do
            
        