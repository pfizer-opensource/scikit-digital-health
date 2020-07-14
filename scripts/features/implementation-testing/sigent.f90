! -*- f90 -*-

subroutine fmean_sd_1d(n, x, mn, sd)
    implicit none
    integer(8), intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: mn, sd
!f2py intent(hide) :: n, mn, sd
    real(8) :: k, Ex, Ex2
    integer(8) :: i
    
    Ex = 0.0_8
    Ex2 = 0.0_8
    mn = 0.0_8
    k = x(1)
    
    do i=1, n
        mn = mn + x(i)
        Ex = Ex + x(i)
        Ex2 = Ex2 + (x(i) - k)**2
    end do
    Ex = Ex - (n * k)  ! account for not subtracting in the loop
    
    sd = sqrt((Ex2 - (Ex**2 / n)) / (n - 1))
    mn = mn / n
end subroutine

subroutine fhist(n, signal, ncells, min_val, max_val, counts)
    implicit none
    integer(8), intent(in) :: n, ncells
    real(8), intent(in) :: signal(n), min_val, max_val
    integer(8), intent(out) :: counts(ncells)
!f2py intent(hide) :: n
    integer(8) :: i, idx
    real(8) :: bin_width

    counts = 0_8
    
    bin_width = (max_val - min_val) / ncells
            
    if (bin_width .EQ. 0.0_8) then
        bin_width = 1.0_8  ! prevent 0 division
    end if
    
    do i=1, n
        if (.NOT. isnan(signal(i))) then
            idx = int((signal(i) - min_val) / bin_width, 8) + 1_8
            if (idx > ncells) then
                idx = ncells
            end if
            
            counts(idx) = counts(idx) + 1
        end if
    end do
end subroutine

subroutine fhistogram(n, k, signal, descriptor, counts)
    ! k is ceiling(sqrt(n))
    implicit none
    integer(8), intent(in) :: n, k
    real(8), intent(in) :: signal(n)
    real(8), intent(inout) :: descriptor(3)
    integer(8), intent(out) :: counts(k)
!f2py intent(hide) :: n
    real(8) :: min_val, max_val, delta
    
    min_val = minval(signal)
    max_val = maxval(signal)
    delta = (max_val - min_val) / (n - 1)
    
    descriptor(1) = min_val - delta / 2
    descriptor(2) = max_val + delta / 2
    descriptor(3) = real(k, 8)
    
    call fhist(n, signal, k, min_val, max_val, counts)
end subroutine
    

subroutine fsignalEntropy(p, n, m, signal, sigEnt)
    implicit none
    integer(8), intent(in) :: p, n, m
    real(8), intent(in) :: signal(p, n, m)
    real(8), intent(out) :: sigEnt(p, m)
!fp2y intent(hide) :: p, n, m
    real(8) :: d(3), data_norm(n), logf, nbias, count
    real(8) :: estimate, std, mean
    integer(8) :: i, j, k, sqn
    integer(8) :: h(ceiling(sqrt(real(n))))
            
    sqn = ceiling(sqrt(real(n)))
    
    do k=1, m
        do i=1, p
            std = 0._8
            mean = 0._8
            call fmean_sd_1d(n, signal(i, :, k), mean, std)
            
            if (std == 0._8) then
                std = 1._8  ! ensure no division by 0
            end if
            
            data_norm = signal(i, :, k) / std
            
            call fhistogram(n, sqn, data_norm, d, h)
            
            if (d(1) == d(2)) then
                sigEnt(i, k) = 0._8
            else
                count = 0._8
                estimate = 0._8

                do j=1, int(d(3), 8)
                    if (h(j) > 0) then
                        logf = log(real(h(j), 8))
                    else
                        logf = 0._8
                    end if
                    print *, logf, h(j)

                    count = count + h(j)
                    estimate = estimate - h(j) * logf
                end do
                print *, ''

                nbias = -(d(3) - 1) / (2 * count)
                estimate = estimate / count + log(count) + log((d(2) - d(1)) / d(3)) - nbias
                sigEnt(i, k) = exp(estimate**2) - 2
            end if
        end do
    end do
end subroutine


subroutine fsignalEntropy2(m, n, p, signal, sigEnt)
    ! # m is the signal dimension
    ! # n is the axis dimension
    ! # p is the window dimension
    implicit none
    integer(8), intent(in) :: m, n, p
    real(8), intent(in) :: signal(m, n, p)
    real(8), intent(out) :: sigEnt(n, p)
!fp2y intent(hide) :: m, n, p
    real(8) :: d(3), data_norm(m), nbias, count
    real(8) :: estimate, std, mean
    real(8) :: logf(ceiling(sqrt(real(m))))
    integer(8) :: j, k, sqn
    integer(8) :: h(ceiling(sqrt(real(m))))
            
    sqn = ceiling(sqrt(real(m)))
    
    do k=1, p
        do j=1, n
            call fmean_sd_1d(m, signal(:, j, k), mean, std)

            print *, mean, std
            
            if (std == 0._8) then
                std = 1._8  ! ensure no division by 0
            end if
            
            data_norm = signal(:, j, k) / std
            
            call fhistogram(m, sqn, data_norm, d, h)
            
            if (d(1) == d(2)) then
                sigEnt(j, k) = 0._8
            else
                count = 0._8
                estimate = 0._8
                
                logf = 0._8
                where (h > 0)
                    logf = log(real(h, 8))
                end where

                print *, logf
                print *, h
                print *, ''
                
                count = sum(h)
                estimate = -sum(h * logf)

                nbias = -(d(3) - 1) / (2 * count)
                estimate = estimate / count + log(count) + log((d(2) - d(1)) / d(3)) - nbias
                sigEnt(j, k) = exp(estimate**2) - 2
            end if
        end do
    end do
end subroutine
    