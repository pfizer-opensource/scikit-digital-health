! -*- f95 -*-

module median_heap
    use sort, only: quick_argsort_, quick_argsort_long_
    use, intrinsic :: iso_c_binding
    implicit none

    ! the workspace for the heap
    real(c_double), dimension(:), allocatable :: heap  ! actual heap data values
    integer(c_long), dimension(:), allocatable :: oldest  ! keeps track of which element is oldest
    integer(c_long), dimension(:), allocatable :: pos  ! intermediate step to maintain oldest

    ! private local attributes to keep track of
    integer(c_long), private :: state  ! keeps track of where in `oldest` we are
    integer(c_long), private :: N  ! number of elements in the heap
    integer(c_long), private :: n_max_heap  ! number of elements in the max heap
    integer(c_long), private :: n_min_heap  ! number of elements in the min heap
    integer, private :: is_even  ! keep track of if the median is an avg of 2 values

    ! label some of the methods as private
    private :: min_sift_away
    private :: min_sift_towards
    private :: max_sift_away
    private :: max_sift_towards

contains
    ! Subroutine to handle the full moving median on a 1D array
    subroutine fmoving_median(k, x, wlen, skip, res) bind(C, name="fmoving_median")
        integer(c_long), intent(in) :: k, wlen, skip
        real(c_double), intent(in) :: x(k)
        real(c_double), intent(out) :: res((k - wlen) / skip + 1)
        ! local
        integer(c_long) :: i, ii, j

        ! first allocate the variables for the heap
        call allocate_heap(wlen)
        ! initialize the heap values
        call initialize_heap(x(1:wlen))
        ! keep track of the last element (+1) inserted into the heap
        ii = wlen + 1

        ! get the first median value
        res(1) = get_median()
        j = 2  ! keep track of where we are in the result array

        ! iterate over each window starting spot
        do i = skip + 1, k - wlen + 1, skip
            ! replace/insert multiple elements at once
            ! note the max(ii, i) here so that if we are skipping values
            ! we dont need to bother with passing them through the heap
            call insert_elements(x(max(ii, i):i + wlen - 1))

            ! get the resulting median value
            res(j) = get_median()
            j = j + 1
            ! update the next element to pull from the input array
            ii = i + wlen
        end do

        ! cleanup the heap, deallocating all the workspaces
        call cleanup_heap()
    end subroutine fmoving_median

    ! Subroutine to allocate the heap workspace
    subroutine allocate_heap(k)
        ! k : number of elements in the heap. equivalent to window length
        integer(c_long), intent(in) :: k

        ! set the # of elements
        N = k

        ! compute the number of elements in each part of the min/max heap
        n_min_heap = k / 2_c_long
        n_max_heap = n_min_heap + mod(k, 2_c_long)  ! 1 longer if odd # of elements

        ! transfer logical response to an integer (0/1)
        is_even = transfer(n_min_heap == n_max_heap, 1)

        ! make sure the heap is cleaned up/ready to be allocated
        call cleanup_heap()

        ! allocate the heap workspaces
        allocate(heap(-n_max_heap + 1:n_min_heap))
        allocate(pos(-n_max_heap + 1:n_min_heap))
        allocate(oldest(0:k-1))  ! different bounds so that it works easily with `state`
    end subroutine allocate_heap

    ! Subroutine to initialize the heap workspace values. This is split from
    ! `allocate_heap` because it can be re-used in the cases where we have no
    ! window overlap
    subroutine initialize_heap(vals)
        ! values to compute the median for using the max/min heap
        ! must match the number of elements provided in `allocate_heap`
        real(c_double), intent(in) :: vals(N)
        ! local variables
        integer(c_long) :: i
        integer(c_long) :: itemp(N)  ! temporary storage so that we dont lose the sorted position

        ! set state to start at the first element
        state = 0_c_long
        ! set the temporary values for the position tracking that will be part of argsort
        itemp = (/ (i, i=-n_max_heap + 1, n_min_heap) /)
        oldest = itemp  ! same values

        ! set the heap data values
        heap = vals

        ! sort the heap, with the temporary position sorting storage
        call quick_argsort_(N, heap, itemp)
        ! save the sorted array since sorting itemp will revert it to its original values
        pos = itemp
        ! sort the sorted index to get the corresponding order of oldest elements
        call quick_argsort_long_(N, itemp, oldest)
    end subroutine initialize_heap

    ! subroutine to quickly cleanup the heap workspace
    subroutine cleanup_heap()
        if (allocated(heap)) then
            deallocate(heap)
            deallocate(pos)
            deallocate(oldest)
        end if
    end subroutine cleanup_heap

    ! utility function to get the median from the max/min heap
    function get_median()
        real(c_double) :: get_median

        ! branchless version checking if we need to take an average of 2 values
        ! is_even = FALSE:
        ! = heap(0) * (1 - 0.5 * 0) + 0.5 * heap(1) * 0 = heap(0) * 1 + 0 = heap(0)
        !
        ! is_even = TRUE
        ! = heap(0) * (1 - 0.5 * 1) + 0.5 * heap(1) * 1
        ! = heap(0) * 0.5 + 0.5 * heap(1)
        ! = (heap(0) + heap(1)) / 2
        get_median = heap(0) * (1.0_c_double - (0.5_c_double * is_even)) + 0.5_c_double * heap(1) * is_even
    end function get_median

    ! subroutine to replace multiple elements from the heap at once
    subroutine insert_elements(vals)
        real(c_double), intent(in) :: vals(:)
        ! local
        integer(c_long) :: nn, i

        nn = size(vals)

        if (nn == N) then ! replacing the whole heap.
            ! just reset the whole heap, and sort again instead of
            ! sifting through the min/max heap N times
            call initialize_heap(vals)
        else
            do i=1, nn
                call insert_element(vals(i))
            end do
        end if
    end subroutine insert_elements

    ! subroutien to replace a single element from the heap
    subroutine insert_element(val)
        real(c_double), intent(in) :: val
        ! local
        integer(c_long) :: i

        ! get the oldest element's position
        i = oldest(state)
        ! update the state
        state = mod(state + 1, N)
        ! replace/insert the oldest value with the new value
        heap(i) = val

        ! now make sure that the heap is valid
        if (i > 0) then  ! we are in the min heap
            ! NOTE the 2i call here so that it is an even index. will modify index i if it needs to
            call min_sift_away(2 * i)  ! Try sorting away from min heap root node
            call min_sift_towards(i)  ! try sorting towards the min heap root node
        else
            ! NOTE the 2i-1 call here so that it is an odd index. will modify index i if it needs to
            call max_sift_away(2 * i - 1)  ! try sorting away from the max heap root node
            call max_sift_towards(i)  ! try sorting towards the max heap root node
        end if
    end subroutine insert_element

    ! subroutine to swap 2 elements in the heap workspace
    subroutine swap(i1, i2)
        integer(c_long), intent(in) :: i1, i2
        ! local
        real(c_double) :: temp
        integer(c_long) :: itemp

        temp = heap(i1)
        heap(i1) = heap(i2)
        heap(i2) = temp
        ! swap the sorted position
        itemp = pos(i1)
        pos(i1) = pos(i2)
        pos(i2) = itemp
        ! oldest list - need to modify index here since it uses a different index range
        oldest(pos(i1) + n_max_heap - 1) = i1
        oldest(pos(i2) + n_max_heap - 1) = i2
    end subroutine swap

    ! Subroutine to sift elements away from the root node in a min heap
    ! NOTE: should always be called with an EVEN index, which corresponds with the
    ! left child node, and allows it to easily find the right node
    subroutine min_sift_away(index)
        integer(c_long), intent(in) :: index
        ! local
        integer(c_long) :: i

        i = index  ! so we dont modify index

        ! 4 5  6 7
        ! 2    3
        ! 1

        do while (i <= n_min_heap)
            ! get the larger of the left/right child nodes
            ! because of the calling with an even #, the right node is i + 1
            ! if ((i > 1) .and. (i < n_min_heap) .and. (heap(i + 1) < heap(i))) then
            !     i = i + 1
            ! end if

            ! this is a branchless version of the above if statement
            ! adding the heap(min(i, j)) so that if a compiler does not support short-circuiting we
            ! dont read a value out of bounds
            i = i + transfer((i > 1) .and. (i < n_min_heap) .and. (heap(min(i + 1, n_min_heap)) < heap(i)), 1)
            ! if the heap is not correct
            if (heap(i) < heap(i / 2)) then
                call swap(i, i / 2)
            else
                exit  ! the heap is correct through here so we can stop checking farther away
            end if
            ! go to the next nodes
            i = 2 * i
        end do
    end subroutine min_sift_away

    ! Subroutine to sift elements away from the root node in the max heap
    ! NOTE: should always be called with an ODD index (negative), which will correspond to the
    ! left child node, and allows it to easily find the right node
    subroutine max_sift_away(index)
        integer(c_long), intent(in) :: index
        ! local
        integer(c_long) :: i

        i = index

        !       0
        !   -1     -2
        ! -3 -4   -5 -6

        do while (i > -n_max_heap)
            ! get the larger of the left/right child nodes
            ! because of the calling with an odd #, the left node is i - 1
            ! if ((i < 0) .and. (i > (-n_max_heap + 1)) .and. (heap(i - 1) > heap(i))) then
            !     i = i - 1
            ! end if

            ! this is a branchless version of the above if statement
            ! adding the heap(max(i, j)) in case a compiler does not support short-circuiting
            i = i - transfer((i < 0) .and. (i > (-n_max_heap + 1)) .and. (heap(max(i - 1, -n_max_heap + 1)) > heap(i)), 1)
            ! if the heap is not correct.  Need the `i+1` correction so that we check the correct
            ! parent node. ie (-2 + 1) / 2 -> 0, (-1 + 1) / 2 -> 0  (-6 + 1) / 2 -> -2
            if (heap(i) > heap((i + 1) / 2)) then
                call swap(i, (i + 1) / 2)
            else
                exit  ! the heap is correct through here, so we can stop checking
            end if
            i = 2 * i - 1  ! need to subtract 1 so that we get to the proper child node
        end do
    end subroutine max_sift_away

    subroutine min_sift_towards(index)
        integer(c_long), intent(in) :: index
        ! local
        integer(c_long) :: i

        i = index

        do while ((i > 0) .and. (heap(i) < heap(i / 2)))
            call swap(i, i / 2)
            i = i / 2
        end do
        ! handle crossing into the max heap
        if (i == 0_c_long) then
            call max_sift_away(-1_c_long)  ! set to odd node below the root
        end if
    end subroutine min_sift_towards

    subroutine max_sift_towards(index)
        integer(c_long), intent(in) :: index
        ! local
        integer(c_long) :: i

        i = index

        do while ((i < 0) .and. (heap(i) > heap((i + 1) / 2)))
            call swap(i, (i + 1) / 2)
            i = (i + 1) / 2
        end do
        ! handle crossing into the min heap
        if ((i == 0) .and. (heap(0) > heap(1))) then
            call swap(0_c_long, 1_c_long)
            call min_sift_away(2_c_long)  ! set to even node below the root
        end if
    end subroutine max_sift_towards
end module median_heap