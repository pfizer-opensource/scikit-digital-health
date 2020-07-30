      include "fftpack/dcfftf.f"
      include "fftpack/dcffti.f"
            

      subroutine domFreq(m, n, p, x, nfft, fs, low_cut, hi_cut, dFreq)
        implicit none
        integer(8), intent(in) :: m, n, p, nfft
        real(8), intent(in) :: x(m, n, p), low_cut, hi_cut, fs
        real(8), intent(out) :: dFreq(n, p)
!f2py   intent(hide) :: n
        real(8), parameter :: log2 = log(2._8)
        integer(8) :: i, ihcut, ilcut, imax, j, k
        real(8) :: freq(nfft), sp_norm(2 * nfft)
        real(8) :: log2_idx_cut
        complex(8) :: sp_hat(2 * nfft), ws(2 * 4 * nfft + 15)
        
        ! compute the resulting frequency array
        freq = (/ (fs * (i - 1) / nfft, i=1, nfft) /) / 2._8

        ! find the cutoffs for the high and low cutoffs
        ihcut = floor(hi_cut / (fs / 2) * (nfft - 1) + 1, 8)
        ilcut = ceiling(low_cut / (fs / 2) * (nfft - 1) + 1, 8)

        if (ihcut > nfft) then
            ihcut = int8(nfft)
        end if
        log2_idx_cut = log(real(ihcut - ilcut, 8)) / log2

        ! compute the FFT
        call dcffti(2 * nfft, ws)  ! initialize
        
        do k=1, p
            do j=1, n
                sp_hat = complex(0._8, 0._8)
                sp_hat(:m) = cmplx(x(:, j, k), kind=8)
                call dcfftf(2 * nfft, sp_hat, ws)  ! compute

                ! compute the power by multiplying by conjugate
                sp_norm = real(sp_hat * conjg(sp_hat), 8)
                sp_norm = sp_norm / sum(sp_norm(ilcut:ihcut))

                ! find the maximum index
                imax = maxloc(sp_norm(ilcut:ihcut), dim=1) + ilcut - 1

                dFreq(j, k) = freq(imax)
            end do
        end do
      end subroutine
    
    
    
    
    
    
    

