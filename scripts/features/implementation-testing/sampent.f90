! -*- f95 -*-

! --------------------------------------------------------------------
! SUBROUTINE  sampleEntropy
!     Compute the sample entropy of a signal
! 
!     Input
!     m          : integer(8), signal dimension
!     n          : integer(8), axis dimension
!     p          : integer(8), window dimension
!     x(m, n, p) : real(8), array to compute sample entropy on
!     L          : integer(8), length of sets to compare
!     r          : real(8), maximum set distance
! 
!     Output
!     sampEnt(n, p) : real(8), sample entropy
! --------------------------------------------------------------------
subroutine sampleEntropy(m, n, p, x, L, r, sampEnt)
    implicit none
    integer(8), intent(in) :: m, n, p, L
    real(8), intent(in) :: x(m, n, p), r
    real(8), intent(out) :: sampEnt(n, p)
!f2py intent(hide) :: m, n, p
    real(8) :: x1, A, B
    integer(8) :: i, j, k, ii, i2, run(m)
    
    do k=1, p
        do j=1, n
            A = 0._8
            B = 0._8
            run = 0_8
            do i=1, m-1
                x1 = x(i, j, k)
                do ii=1, m - i
                    i2 = ii + i
                    if (abs(x(i2, j, k) - x1) < r) then
                        run(ii) = run(ii) + 1
                        
                        if (run(ii) >= L) then
                            A = A + 1._8
                        end if
                        if (run(ii) >= (L - 1)) then
                            if (i2 < m) then
                                B = B + 1._8
                            end if
                        end if
                    else
                        run(ii) = 0_8
                    end if
                end do
            end do
            
            if (L == 1) then
                sampEnt(j, k) = -log(A / (m * (m - 1) / 2))
            else
                sampEnt(j, k) = -log(A / B)
            end if
        end do
    end do              
end subroutine


subroutine sampen2(n, x, L, r, se)
    implicit none
    integer(8), intent(in) :: n, L
    real(8), intent(in) :: x(n), r
    real(8), intent(out) :: se
!f2py intent(hide) :: n
    integer(8) :: run(n), i, ii, i2
    real(8) :: A, B, x1
    
    A = 0._8
    B = 0._8
    run = 0_8
    
    do i=1, n-1
        x1 = x(i)
        do ii=1, n - i
            i2 = ii + i
            if (abs(x(i2) - x1) < r) then
                run(ii) = run(ii) + 1
                
                if (run(ii) .GE. L) then
                    A = A + 1
                end if
                if (run(ii) .GE. (L - 1)) then
                    if (i2 < n) then
                        B = B + 1
                    end if
                end if
            else
                run(ii) = 0_8
            end if
        end do
    end do
    
    if (L == 1) then
        se = -log(A / (n * (n - 1) / 2))
    else
        se = -log(A / B)
    end if
end subroutine
            
            
            
    