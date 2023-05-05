! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.


subroutine mov_moments_1(n, x, wlen, skip, mean) bind(C, name="mov_moments_1")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-wlen)/skip+1)
    ! local
    integer(c_long) :: i, j
    real(c_double) :: m1(n)

    m1(1) = x(1)

    do i=2, n
        m1(i) = m1(i-1) + x(i)
    end do

    j = 2_c_long
    mean(1) = m1(wlen)

    do i=wlen+skip, n, skip
        mean(j) = m1(i) - m1(i-wlen)
        j = j + 1
    end do

    mean = mean / wlen
end subroutine


subroutine mov_moments_2(n, x, wlen, skip, mean, sd) bind(C, name="mov_moments_2")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-wlen)/skip+1)
    real(c_double), intent(out) :: sd((n-wlen)/skip+1)
    ! local
    integer(c_long) :: i, j
    real(c_double) :: m1(n), m2(n)
    real(c_double) :: delta, delta_n, term1
    integer(c_long) :: na, nb

    m1(1) = x(1)
    m2(1) = 0._c_double

    do i=2, n
        delta = x(i) - m1(i-1) / (i-1)
        delta_n = delta / i
        term1 = delta * delta_n * (i-1)

        m1(i) = m1(i-1) + x(i)
        m2(i) = m2(i-1) + term1
    end do

    j = 2_c_long
    mean(1) = m1(wlen)
    sd(1) = m2(wlen)

    do i=wlen+skip, n, skip
        na = wlen
        nb = i-wlen

        delta = m1(nb) / nb - (m1(i) - m1(nb)) / wlen

        mean(j) = m1(i) - m1(nb)
        sd(j) = m2(i) - m2(nb) - delta**2 * na * nb / i

        j = j + 1
    end do

    where ((sd > -epsilon(sd(1))) .and. (sd < 0.0))
        sd = -1.0 * sd
    end where

    ! NOTE: currently, sd = M2, skew = M3, kurt = M4, so this order of computation matters
    mean = mean / wlen
    sd = sqrt(sd / real(wlen - 1, c_double))

end subroutine

! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    wlen : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-wlen)/skip + 1)
!         Computed moving mean
subroutine moving_moments_1(n, x, wlen, skip, mean) bind(C, name="moving_moments_1")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-wlen)/skip+1)
    ! local
    integer(c_long) :: i, j, k
    real(c_double) :: m1(0:wlen)
    
    if ((skip < wlen) .AND. (mod(wlen, skip) == 0)) then
        m1(0) = 0._c_double
        do i = 1, wlen
            m1(i) = m1(i - 1) + x(i)
        end do

        mean(1) = m1(wlen)
        do k = 2, (n - wlen) / skip + 1
            i = wlen + 1 + (k - 2) * skip
            mean(k) = mean(k - 1) + sum(x(i:i + skip - 1))
        end do

        mean(2 + (wlen - skip) / skip + 1:) = mean(2 + (wlen - skip) / skip + 1:) - mean(2:size(mean) - (wlen - skip) / skip - 1)
        mean(2:2 + (wlen - skip) / skip) = mean(2:2 + (wlen - skip) / skip) - m1(skip:wlen:skip)

    else if (skip < wlen) then
        m1(0) = 0._c_double
        do i = 1, wlen
            m1(i) = m1(i - 1) + x(i)
        end do

        mean(1) = m1(wlen)
        k = 2_c_long

        do i = wlen + skip, n, skip
!            m1(0:wlen - skip) = m1(skip:wlen)
            m1 = eoshift(m1, shift=skip)

            do j = wlen - skip + 1, wlen
                m1(j) = m1(j - 1) + x(i - wlen + j)
            end do
            mean(k) = m1(wlen) - m1(0)
            k = k + 1
        end do
    else
        do k = 1, (n - wlen) / skip + 1
            mean(k) = x((k - 1) * skip + 1)

            do j = 2, wlen
                mean(k) = mean(k) + x((k - 1) * skip + j)
            end do
        end do
    end if

    mean = mean / wlen
end subroutine


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    wlen : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-wlen)/skip + 1)
!         Computed moving mean
!    sd : array((n-wlen)/skip + 1)
!         Computed moving standard deviation
subroutine moving_moments_2(n, x, wlen, skip, mean, sd) bind(C, name="moving_moments_2")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-wlen)/skip+1)
    real(c_double), intent(out) :: sd((n-wlen)/skip+1)
    ! local
    integer(c_long) :: i, j, k
    real(c_double) :: m1(0:n), m2(0:n)
    real(c_double) :: delta, delta_n, term1, tm1

    if ((skip < wlen) .AND. (mod(wlen, skip) == 0)) then
        m1(1) = x(1)
        m2(1) = 0._c_double

        do i=2, wlen
            delta = x(i) - m1(i - 1) / (i - 1)
            delta_n = delta / i
            term1 = delta * delta_n * (i - 1)

            m1(i) = m1(i - 1) + x(i)
            m2(i) = m2(i - 1) + term1
        end do

        mean(1) = m1(wlen)
        sd(1) = m2(wlen)
        do k = 2, (n - wlen) / skip + 1
            i = wlen + 1 + (k - 2) * skip
            mean(k) = mean(k - 1) + sum(x(i:i + skip - 1))
            sd(k) = sd(k - 1)
            ! this accomplishes same as the sum function for mean
            tm1 = mean(k - 1)
            do j = i, i + skip - 1
                delta = x(j) - tm1 / (j - 1)
                delta_n = delta / j
                term1 = delta * delta_n * (j - 1)

                sd(k) = sd(k) + term1
                tm1 = tm1 + x(j)
            end do
        end do

        ! this loop has to go backwards so we don't modify values we need later
        j = size(mean) - (wlen - skip) / skip - 1
        do i = size(mean), 2 + (wlen - skip) / skip + 1, -1
            delta = mean(j) / ((i - 1) * skip) - (mean(i) - mean(j)) / wlen

            sd(i) = sd(i) - sd(j) - delta**2 * wlen * ((i - 1) * skip) / ((i - 1) * skip + wlen)
            mean(i) = mean(i) - mean(j)
            j = j - 1
        end do
        j = skip
        do i = 2, 2 + (wlen - skip) / skip
            delta = m1(j) / ((i - 1) * skip) - (mean(i) - m1(j)) / wlen

            sd(i) = sd(i) - m2(j) - delta**2 * wlen * ((i - 1) * skip) / ((i - 1) * skip + wlen)
            mean(i) = mean(i) - m1(j)
            j = j + skip
        end do

    else if (skip < wlen) then
        m1(1) = x(1)
        m2(0:1) = 0._c_double
        do i=2, wlen
            delta = x(i) - m1(i - 1) / (i - 1)
            delta_n = delta / i
            term1 = delta * delta_n * (i - 1)

            m1(i) = m1(i - 1) + x(i)
            m2(i) = m2(i - 1) + term1
        end do

        mean(1) = m1(wlen)
        sd(1) = m2(wlen)
        k = 2_c_long

        do i = wlen + skip, n, skip
            m1(0:wlen - skip) = m1(skip:wlen)
            m2(0:wlen - skip) = m2(skip:wlen)

            do j = wlen - skip + 1, wlen
                delta = x(i - wlen + j) - m1(j - 1) / (i - wlen + j - 1)
                delta_n = delta / (i - wlen + j)
                term1 = delta * delta_n * (i - wlen + j - 1)

                m1(j) = m1(j - 1) + x(i - wlen + j)
                m2(j) = m2(j - 1) + term1
            end do
            delta = m1(0) / (i - wlen) - (m1(wlen) - m1(0)) / wlen

            mean(k) = m1(wlen) - m1(0)
            sd(k) = m2(wlen) - m2(0) - delta**2 * wlen * (i - wlen) / i
            k = k + 1
        end do
    else
        do k = 1, (n - wlen) / skip + 1
            mean(k) = x((k - 1) * skip + 1)
            sd(k) = 0._c_double

            do j = 2, wlen
                delta = x((k - 1) * skip + j) - mean(k) / (j - 1)
                delta_n = delta / j
                term1 = delta * delta_n * (j - 1)

                mean(k) = mean(k) + x((k - 1) * skip + j)
                sd(k) = sd(k) + term1
            end do
        end do
    end if

    mean = mean / wlen
    sd = sqrt(sd / (wlen - 1))
end subroutine


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    wlen : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-wlen)/skip + 1)
!         Computed moving mean
!    sd : array((n-wlen)/skip + 1)
!         Computed moving standard deviation
!    skew : array((n-wlen)/skip + 1)
!         Computed moving skewness
subroutine moving_moments_3(n, x, wlen, skip, mean, sd, skew) bind(C, name="moving_moments_3")
    use, intrinsic :: ieee_arithmetic, only: IEEE_Value, IEEE_QUIET_NAN
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-wlen)/skip+1)
    real(c_double), intent(out) :: sd((n-wlen)/skip+1)
    real(c_double), intent(out) :: skew((n-wlen)/skip+1)
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
    mean(1) = m1(wlen)
    sd(1) = m2(wlen)
    skew(1) = m3(wlen)

    do i=wlen+skip, n, skip
        na = wlen
        nb = i-wlen
        
        delta = m1(nb) / nb - (m1(i) - m1(nb)) / wlen
        
        mean(j) = m1(i) - m1(nb)
        sd(j) = m2(i) - m2(nb) - delta**2 * na * nb / i
        
        skew(j) = m3(i) - m3(nb) - delta**3 * na * nb * (2 * na - i) / i**2 - 3 * delta * (na * m2(nb) - nb * sd(j)) / i
        
        j = j + 1
    end do

    where ((sd > -epsilon(sd(1))) .and. (sd < 0.0))
        sd = -1.0 * sd
    end where
    where ((skew > -epsilon(sd(1))) .and. (skew < 0.0))
        skew = -1.0 * skew
    end where

    ! NOTE: currently, sd = M2, skew = M3, kurt = M4, so this order of computation matters
    mean = mean / wlen
    skew = sqrt(real(wlen)) * skew / sd**(3._c_double / 2._c_double)
    ! set to NaN where we would be dividing by zero
    where (sd < epsilon(sd(1)))
        skew = IEEE_Value(skew(1), IEEE_QUIET_NAN)
    end where
    sd = sqrt(sd / (wlen - 1))

end subroutine


! =======================================================
! computation of moving statistical moments
!
! Inputs
!    n : int
!         Number of samples in x
!    x : array
!         1D array of samples to compute the moments for
!    wlen : int
!         Number of samples in each window
!    skip : int     
!         Number of samples to skip for the start of each window
!
! Outputs
!    mean : array((n-wlen)/skip + 1)
!         Computed moving mean
!    sd : array((n-wlen)/skip + 1)
!         Computed moving standard deviation
!    skew : array((n-wlen)/skip + 1)
!         Computed moving skewness
!    kurt : array((n-wlen)/skip + 1)
!         Computed moving kurtosis
subroutine moving_moments_4(n, x, wlen, skip, mean, sd, skew, kurt) bind(C, name="moving_moments_4")
    use, intrinsic :: ieee_arithmetic, only: IEEE_Value, IEEE_QUIET_NAN
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_long), intent(in) :: n, wlen, skip
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: mean((n-wlen)/skip+1)
    real(c_double), intent(out) :: sd((n-wlen)/skip+1)
    real(c_double), intent(out) :: skew((n-wlen)/skip+1)
    real(c_double), intent(out) :: kurt((n-wlen)/skip+1)
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
    mean(1) = m1(wlen)
    sd(1) = m2(wlen)
    skew(1) = m3(wlen)
    kurt(1) = m4(wlen)

    do i=wlen+skip, n, skip
        na = wlen
        nb = i-wlen
        
        delta = m1(nb) / nb - (m1(i) - m1(nb)) / wlen
        
        mean(j) = m1(i) - m1(nb)
        sd(j) = m2(i) - m2(nb) - delta**2 * na * nb / i
        
        skew(j) = m3(i) - m3(nb) - delta**3 * na * nb * (2 * na - i) / i**2 - 3 * delta * (na * m2(nb) - nb * sd(j)) / i
        
        kurt(j) = m4(i) - m4(nb) - delta**4 * na * nb * (na**2 - na*nb + nb**2) / i**3 - 6 * delta**2 * (na**2 * m2(i-wlen) &
        + nb**2 * sd(j)) / i**2 - 4 * delta * (na * m3(i-wlen) - nb * skew(j)) / i
        
        j = j + 1
    end do

    where ((sd > -epsilon(sd(1))) .and. (sd < 0.0))
        sd = -1.0 * sd
    end where

    ! NOTE: currently, sd = M2, skew = M3, kurt = M4, so this order of computation matters
    mean = mean / wlen
    skew = sqrt(real(wlen)) * skew / sd**(3._c_double / 2._c_double)
    kurt = wlen * kurt / sd**2 - 3
    ! set to NaN where we would be dividing by zero
    where (sd < epsilon(sd(1)))
        skew = IEEE_Value(skew(1), IEEE_QUIET_NAN)
        kurt = IEEE_Value(kurt(1), IEEE_QUIET_NAN)
    end where
    sd = sqrt(sd / (wlen - 1))

end subroutine
