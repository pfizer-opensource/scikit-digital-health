! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.

subroutine f_rfft(n, x, nfft, F) bind(C, name="f_rfft")
    use, intrinsic :: iso_c_binding
    use real_fft, only : execute_real_forward
    implicit none
    integer(c_long), intent(in) :: n, nfft
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: F(2 * nfft + 2)
    ! local
    integer(c_long) :: ier
    real(c_double) :: y(2 * nfft)

    y = 0._c_double
    y(:n) = x
    F = 0._c_double
    call execute_real_forward(2 * nfft, y, 1.0_c_double, F, ier)
end subroutine f_rfft