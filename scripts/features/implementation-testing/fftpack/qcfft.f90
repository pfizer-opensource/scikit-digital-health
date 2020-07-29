! -*- f90 -*-
include "complex/qcfti1_2.f90"
include "complex/qcftf1_2.f90"

subroutine qcffti(n, wsave)
    use iso_fortran_env
    integer(int64) :: wsave, iw1, iw2
    real(real64) :: wsave(1)
    
    if (n .eq. 1) return
    
    iw1 = n + n + 1
    iw2 = iw1 + n + n
    
    call qcfti1(n, wsave(iw1), wsave(iw2))
    
    return
end subroutine


subroutine qcfftf(n, c, wsave)
    use iso_fortran_env
    integer(int64) :: n, iw1, iw2
    real(real64) :: c(1), wsave(1)
    
    if (n .eq. 1) return
    
    iw1 = n + n + 1
    iw2 = iw1 + n + n
    
    call qcftf1(n, c, wsave, wsave(iw1), wsave(iw2))
    
    return
end subroutine
    
    
    

