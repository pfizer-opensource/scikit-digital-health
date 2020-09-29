! -*- f95 -*-
include "sort.f90"
include "common.f90"


! --------------------------------------------------------------------
! SUBROUTINE  embed_sort
!     Create embedding vectors from the array x, and get the indices of the sorted embedding vectors
! 
!     In
!     n, m   : integer(8)
!     x(n)   : real(8), array of values of length n
!     order  : integer(8), order (number of values) of each embedding vector
!     delay  : integer(8), number of samples to skip when extracting each embedding vector
! 
!     Out
!     res(m, order)  : integer(8), sorted embedding vectors
! --------------------------------------------------------------------
subroutine embed_sort(n, m, x, order, delay, res)
    implicit none
    integer(8), intent(in) :: n, m, order, delay
    real(8), intent(in) :: x(n)
    integer(8), intent(out) :: res(m, order)
!f2py intent(hide) :: n
    integer(8) :: j
    real(8) :: temp(m, order)
    
    do j=0, order-1
        res(:, j+1) = j
    end do
    
    do j=0, order-1
        temp(:, j+1) = x(j * delay + 1:j * delay + 1 + m)
    end do
    
    call dpqsort_2d(m, order, temp, res)
    
end subroutine


! --------------------------------------------------------------------
! SUBROUTINE  permutationEntropy
!     Compute the permutation entropy of a 3d array
! 
!     In
!     m, n, p     : integer(8)
!     x(m, n, p)  : real(8), array of values of length n
!     order       : integer(8), order (number of values) of each embedding vector
!     delay       : integer(8), number of samples to skip when extracting each embedding vector
!     normalize   : logical, normalize the permutation entropy
! 
!     Out
!     pe(n, p)  : real(8), computed permutation entropy
! --------------------------------------------------------------------
subroutine permutationEntropy(m, n, p, x, order, delay, normalize, pe)
    implicit none
    integer(8), intent(in) :: m, n, p, order, delay
    real(8), intent(in) :: x(m, n, p)
    logical, intent(in) :: normalize
    real(8), intent(out) :: pe(n, p)
!f2py intent(hide) :: m, n, p
    integer(8) :: i, j, k, msi, n_unq
    integer(8) :: sorted_idx(m - ( order - 1) * delay, order)
    real(8) :: hashval(m - ( order - 1) * delay)
    real(8) :: unq_vals(m - ( order - 1) * delay)
    real(8) :: unq_counts(m - ( order - 1) * delay)
    real(8), parameter :: log2 = dlog(2._8)
    
    msi = m - ( order - 1) * delay
    
    do k=1, p
        do j=1, n
            call embed_sort(m, msi, x(:, j, k), order, delay, sorted_idx)
            
            hashval = 0._8
            do i=1, order
                hashval = hashval + order**(i-1) * sorted_idx(:, i)
            end do
            
            unq_counts = 0._8
            call unique(msi, hashval, unq_vals, unq_counts, n_unq)
            
            unq_counts = unq_counts / sum(unq_counts)
            
            pe(j, k) = -sum(unq_counts(:n_unq) * dlog(unq_counts(:n_unq)) / log2)
        end do
    end do
    
    if (normalize) then
        pe = pe / (dlog(product((/(real(j, 8), j=1, order)/))) / log2)
    end if
end subroutine
