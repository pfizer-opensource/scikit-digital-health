! -*- f95 -*-


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    lag : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-lag)/skip + 1)
!         Computed moving mean
subroutine rolling_moments_1(n, x, lag, skip, mean) bind(C, name="rolling_moments_1")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, lag, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-lag)/skip+1)
    ! local
    integer(c_long) :: i, j
    real(c_double) :: m1(n)
    
    m1(1) = x(1)

    do i=2, n
        m1(i) = m1(i - 1) + x(i)
    end do

    j = 2_c_long
    mean(1) = m1(lag)

    do i=lag+skip, n, skip
        mean(j) = m1(i) - m1(i-lag)
    end do

    mean = mean / lag
end subroutine


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    lag : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-lag)/skip + 1)
!         Computed moving mean
!    sd : array((n-lag)/skip + 1)
!         Computed moving standard deviation
subroutine rolling_moments_2(n, x, lag, skip, mean, sd) bind(C, name="rolling_moments_2")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, lag, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-lag)/skip+1)
    real(c_double), intent(out) :: sd((n-lag)/skip+1)
    ! local
    integer(c_long) :: i, j
    real(c_double) :: m1(n), m2(n)
    real(c_double) :: delta, delta_n, delta_n2, term1

    m1(1) = x(1)
    m2(1) = 0._c_double

    do i=2, n
        delta = x(i) - m1(i-1) / (i-1)
        delta_n = delta / i
        delta_n2 = delta_n**2
        term1 = delta * delta_n * (i-1)
        
        m1(i) = m1(i-1) + x(i)
        m2(i) = m2(i-1) + term1
    end do

    j = 2_c_long
    mean(1) = m1(lag)
    sd(1) = m2(lag)

    do i=lag+skip, n, skip
        delta = m1(i-lag) / (i-lag) - (m1(i) - m1(i-lag)) / lag
        
        mean(j) = m1(i) - m1(i-lag)
        sd(j) = m2(i) - m2(i-lag) - delta**2 * lag * (i-lag) / i
        
        j = j + 1
    end do

    mean = mean / lag
    sd = sqrt(sd / (lag - 1))

end subroutine


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    lag : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-lag)/skip + 1)
!         Computed moving mean
!    sd : array((n-lag)/skip + 1)
!         Computed moving standard deviation
!    skew : array((n-lag)/skip + 1)
!         Computed moving skewness
subroutine rolling_moments_3(n, x, lag, skip, mean, sd, skew) bind(C, name="rolling_moments_3")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, lag, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-lag)/skip+1)
    real(c_double), intent(out) :: sd((n-lag)/skip+1)
    real(c_double), intent(out) :: skew((n-lag)/skip+1)
    ! local
    integer(c_long) :: i, j
    real(c_double) :: m1(n), m2(n), m3(n)
    real(c_double) :: delta, delta_n, delta_n2, term1
    integer(c_long) :: na, nb

    m1(1) = x(1)
    m2(1) = 0._c_double
    m3(1) = 0._c_double

    do i=2, n
        delta = x(i) - m1(i-1) / (i-1)
        delta_n = delta / i
        delta_n2 = delta_n**2
        term1 = delta * delta_n * (i-1)
        
        m1(i) = m1(i-1) + x(i)
        m2(i) = m2(i-1) + term1
        m3(i) = m3(i-1) + term1 * delta_n * (i-2) - 3 * delta_n * m2(i-1)
    end do

    j = 2_c_long
    mean(1) = m1(lag)
    sd(1) = m2(lag)
    skew(1) = m3(lag)

    do i=lag+skip, n, skip
        na = lag
        nb = i-lag
        
        delta = m1(nb) / nb - (m1(i) - m1(nb)) / lag
        
        mean(j) = m1(i) - m1(nb)
        sd(j) = m2(i) - m2(nb) - delta**2 * na * nb / i
        
        skew(j) = m3(i) - m3(nb) - delta**3 * na * nb * (2 * na - i) / i**2 - 3 * delta * (na * m2(nb) - nb * sd(j)) / i
        
        j = j + 1
    end do

    ! NOTE: currently, sd = M2, skew = M3, kurt = M4, so this order of computation matters
    mean = mean / lag
    skew = sqrt(real(lag)) * skew / sd**(3._c_double / 2._c_double)
    sd = sqrt(sd / (lag - 1))

end subroutine


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    lag : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-lag)/skip + 1)
!         Computed moving mean
!    sd : array((n-lag)/skip + 1)
!         Computed moving standard deviation
!    skew : array((n-lag)/skip + 1)
!         Computed moving skewness
!    kurt : array((n-lag)/skip + 1)
!         Computed moving kurtosis
subroutine rolling_moments_4(n, x, lag, skip, mean, sd, skew, kurt) bind(C, name="rolling_moments_4")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, lag, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-lag)/skip+1)
    real(c_double), intent(out) :: sd((n-lag)/skip+1)
    real(c_double), intent(out) :: skew((n-lag)/skip+1)
    real(c_double), intent(out) :: kurt((n-lag)/skip+1)
    ! local
    integer(c_long) :: i, j
    real(c_double) :: m1(n), m2(n), m3(n), m4(n)
    real(c_double) :: delta, delta_n, delta_n2, term1
    integer(c_long) :: na, nb

    m1(1) = x(1)
    m2(1) = 0._c_double
    m3(1) = 0._c_double
    m4(1) = 0._c_double

    do i=2, n
        delta = x(i) - m1(i-1) / (i-1)
        delta_n = delta / i
        delta_n2 = delta_n**2
        term1 = delta * delta_n * (i-1)
        
        m1(i) = m1(i-1) + x(i)
        m2(i) = m2(i-1) + term1
        m3(i) = m3(i-1) + term1 * delta_n * (i-2) - 3 * delta_n * m2(i-1)
        m4(i) = m4(i-1) + term1 * delta_n2 * (i*i - 3*i + 3) + 6 * delta_n2 * m2(i-1) - 4 * delta_n * m3(i-1)
    end do

    j = 2_c_long
    mean(1) = m1(lag)
    sd(1) = m2(lag)
    skew(1) = m3(lag)
    kurt(1) = m4(lag)

    do i=lag+skip, n, skip
        na = lag
        nb = i-lag
        
        delta = m1(nb) / nb - (m1(i) - m1(nb)) / lag
        
        mean(j) = m1(i) - m1(nb)
        sd(j) = m2(i) - m2(nb) - delta**2 * na * nb / i
        
        skew(j) = m3(i) - m3(nb) - delta**3 * na * nb * (2 * na - i) / i**2 - 3 * delta * (na * m2(nb) - nb * sd(j)) / i
        
        kurt(j) = m4(i) - m4(nb) - delta**4 * na * nb * (na**2 - na*nb + nb**2) / i**3 - 6 * delta**2 * (na**2 * m2(i-lag) &
        + nb**2 * sd(j)) / i**2 - 4 * delta * (na * m3(i-lag) - nb * skew(j)) / i
        
        j = j + 1
    end do

    ! NOTE: currently, sd = M2, skew = M3, kurt = M4, so this order of computation matters
    mean = mean / lag
    skew = sqrt(real(lag)) * skew / sd**(3._c_double / 2._c_double)
    kurt = lag * kurt / sd**2 - 3
    sd = sqrt(sd / (lag - 1))

end subroutine
