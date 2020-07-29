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
        
        
        
    end do
            
        