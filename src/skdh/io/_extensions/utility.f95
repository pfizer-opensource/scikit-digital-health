! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.


module custom_time
    use, intrinsic :: iso_c_binding
    implicit none

    type, bind(C) :: time_t
        integer(c_long) :: hour
        integer(c_long) :: min
        integer(c_long) :: sec
        integer(c_long) :: msec  ! NOTE that this is a integer!
    end type time_t

    real(c_double), parameter :: SEC_HOUR = 3600.0
    real(c_double), parameter :: SEC_MIN = 60.0
    real(c_double), parameter :: DAY_SEC = 86400.0

contains

    ! =============================================================================================
    ! get_day_indexing
    ! 
    ! Sets the index into a time series any of the window ends fall into the provide data block 
    ! time frame.
    ! =============================================================================================
    subroutine get_day_indexing(fs, dtime, block_t_delta, mxd, n, bases, periods, block_n, max_n, &
            block_samples, starts, i_starts, stops, i_stops) bind(C, name="get_day_indexing")
        real(c_double), intent(in) :: fs    ! sampling frequency in Hz
        type(time_t), intent(in) :: dtime   ! storage structure for HH:MM:SS & msec data
        real(c_double), intent(in) :: block_t_delta  ! time delta across the block
        integer(c_long), intent(in) :: mxd  ! max days/possible windows. Dimension of starts & stops
        integer(c_long), intent(in) :: n    ! number of windows (bases & periods)
        integer(c_long), intent(in) :: bases(n)  ! base (start) window hour, 24 hr notation
        integer(c_long), intent(in) :: periods(n)  ! window duration in hours
        integer(c_long), intent(in) :: block_n  ! the number of the block currently on
        integer(c_long), intent(in) :: max_n    ! the maximum number of blocks in the data file
        integer(c_long), intent(in) :: block_samples  ! the number of data samples per block
        integer(c_long), intent(inout) :: starts(n, mxd)  ! the start index of windows
        integer(c_long), intent(inout) :: i_starts(n)  ! keeps track of where in starts we are
        integer(c_long), intent(inout) :: stops(n, mxd)  ! the stop index of windows
        integer(c_long), intent(inout) :: i_stops(n)  ! keeps track of where in stops we are
        ! local
        integer(c_long) :: i, idx_start, idx_stop
        real(c_double) :: base_sec, period_sec, dtmp, dtmp2, curr, block_dt
        logical :: in_win1, in_win2

        curr = dtime%hour * SEC_HOUR + dtime%min * SEC_MIN + dtime%sec + real(dtime%msec, c_double) / 1000.

        do i=1, n
            base_sec = bases(i) * SEC_HOUR
            dtmp = real(periods(i) + bases(i), c_double)
            period_sec = mod(dtmp, 24.0) * SEC_HOUR

            ! +1 to account for fortran indexing starting at 1
            idx_start = i_starts(i) + 1
            idx_stop = i_stops(i) + 1

            if (block_n == 0) then  ! if the first data block
                ! Recording start 10:45, windows 0-24, 11-11, 12-6, 8-12, 6-10, 12-11, 11-10

                ! 0-24  yes   10.75 in 0-24  : yes  10.75 in -24-0   : no   || yes
                ! 11-35 yes   10.75 in 11-35 : no   10.75 in -13-11  : yes  || yes
                ! 12-18 no    10.75 in 12-18 : no   10.75 in -12--6  : no   || no
                ! 8-12  yes   10.75 in 8-12  : yes  10.75 in -16--12 : no   || yes
                ! 6-10  no    10.75 in 6-10  : no   10.75 in -18--14 : no   || no
                ! 12-35 yes   10.75 in 12-35 : no   10.75 in -12-11  : yes  || yes
                ! 11-34 no    10.75 in 11-34 : no   10.75 in -13-10  : no   || no

                in_win1 = (curr > base_sec) .and. (curr < (dtmp * SEC_HOUR))
                in_win2 = (curr > (base_sec - DAY_SEC)) .and. (curr < ((dtmp * SEC_HOUR) - DAY_SEC))

                if (in_win1 .or. in_win2) then
                    starts(i, 1) = 0
                    i_starts(i) = 1  ! can't be anything higher than this
                end if
            end if
            
            ! account for differences in fortran/c indexing
            if ((block_n == max_n) .or. (block_n == (max_n - 1))) then
                in_win1 = (curr > base_sec) .and. (curr < (dtmp * SEC_HOUR))
                in_win2 = (curr > (base_sec - DAY_SEC)) .and. (curr < ((dtmp * SEC_HOUR) - DAY_SEC))

                if (in_win1 .or. in_win2) then
                    stops(i, idx_stop) = max_n * block_samples - 1
                    ! don't increment idx_stop here so that if it hits here 2x, dont get 2 entries
                end if
            end if

            ! check if the (base + period) is during this block
            dtmp = period_sec - curr
            dtmp2 = dtmp + DAY_SEC

            if (((dtmp >= 0) .and. (dtmp < block_t_delta)) .or. (dtmp2 < block_t_delta)) then
                stops(i, idx_stop) = block_samples * block_n + int(fs * min(abs(dtmp), abs(dtmp2)), c_long)
                i_stops(i) = i_stops(i) + 1
            end if

            ! check if base is during this block
            dtmp = base_sec - curr
            dtmp2 = dtmp + DAY_SEC

            if (((dtmp >= 0) .and. (dtmp < block_t_delta)) .or. (dtmp2 < block_t_delta)) then
                starts(i, idx_start) = block_samples * block_n + int(fs * min(abs(dtmp), abs(dtmp2)), c_long)
                i_starts(i) = i_starts(i) + 1
            end if
        end do
    end subroutine get_day_indexing

end module custom_time