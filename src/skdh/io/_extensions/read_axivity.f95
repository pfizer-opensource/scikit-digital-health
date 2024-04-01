! -*- f95 -*-

! Copyright (c) 2021. Pfizer Inc. All rights reserved.


module axivity
    use, intrinsic :: iso_c_binding
    implicit none

    type :: metadata
        character(2) :: header                      ! @1  [2] "MD" metadata block
        integer(c_int16_t) :: blockSize             ! @3  [2] Packet size (should be 508 - 512 less first 4 bytes)
        integer(c_int8_t) :: performClear           ! @5  [1] 
        integer(c_int16_t) :: deviceID              ! @6  [2] device identifier
        integer(c_int32_t) :: sessionID             ! @8  [4] session identifier
        integer(c_int16_t) :: upperDeviceID         ! @12 [2] upper word of device ID (treat 0xffff as 0x0000)
        integer(c_int32_t) :: loggingStartTime      ! @14 [4]
        integer(c_int32_t) :: loggingEndTime        ! @18 [4]
        integer(c_int32_t) :: loggingCapacity       ! @22 [4] deprecated, should be 0
        integer(c_int8_t)  :: reserved1             ! @26 [1]
        integer(c_int8_t)  :: flashLED              ! @27 [1]
        integer(c_int8_t)  :: reserved2(8)          ! @28 [8]
        integer(c_int8_t)  :: sensorConfig          ! @36 [1] Fixed rate sensor config. 0x00 or 0xff=>accel only, otherwise bottom nibble is gyro range, top nibble non-zero is magnetometer enabled
        integer(c_int8_t)  :: samplingRate          ! @37 [1] Sampling rate/accelerometer range/scale
    end type metadata

    type :: datapacket
        integer(c_int16_t) :: header                  ! @1  [2] ASCII "AX", little endian (0x5841)
        integer(c_int16_t) :: length            ! @3  [2] Packet length, 508 bytes, with header (4) = 512 bytes
        integer(c_int16_t) :: deviceID          ! @5  [2] Top bit set, 15-bit fraction of a second for the time stamp, the timestampOffset was already adjusted to minimize this assuming ideal sample rate; top bit clear: 15-bit device identifier, 0=unknown
        integer(c_int32_t) :: sessionID         ! @7  [4] Unique session identifier, 0 = unkonwn
        integer(c_int32_t) :: sequenceID        ! @11 [4] Sequence number, 0 indexed
        integer(c_int32_t) :: timestamp         ! @15 [4] last reported RTC value, 0 = unknown
        integer(c_int16_t) :: light             ! @19 [2] AAAGGGLLLLLLLLLL bottom 10 is light, top 3 are unpacked accel scale (1/2^(8+n) g), next 3 are gyro range (8000/(2^n) dps)
        integer(c_int16_t) :: temperature       ! @21 [2] Last recorded temperature value (bottom 10 bits)
        integer(c_int8_t) :: events             ! @23 [1] event flags since last packet
        integer(c_int8_t) :: battery            ! @24 [1] last recorded battery level in scaled/cropped raw units (double and add 512 for 10-bit ADC value)
        integer(c_int8_t) :: sampleRate         ! @25 [1] sample rate code, frequency
        integer(c_int8_t) :: numAxesBPS         ! @26 [1] 0x32 top nibble: number of axes, bottom: packing format - 2=3x 16 bit signed, 0=3x 10bit signed + 2 bit exponent
        integer(c_int16_t) :: timestampOffset   ! @27 [2] Relativ esample index from the start of the buffer where the whole second timestamp is valid
        integer(c_int16_t) :: sampleCount       ! @29 [2] Number of sensor samples (if this sector is full -- Axyz: 80 or 120 samples, Gxyz/Axyz: 40 samples)
    end type datapacket

    type, bind(c) :: FileInfo_t
        integer(c_long) :: deviceID
        integer(c_long) :: sessionID
        integer(c_int) :: nblocks
        integer(c_int8_t) :: axes
        integer(c_int16_t) :: count
        real(c_double) :: tLast
        integer(c_int) :: N
        real(c_double) :: frequency
        integer(c_long) :: n_bad_blocks
    end type FileInfo_t

    type, bind(C) :: time_t
        integer(c_long) :: hour
        integer(c_long) :: min
        integer(c_long) :: sec
        integer(c_long) :: msec  ! NOTE that this is a integer!
    end type time_t

    real(c_double), parameter :: SEC_HOUR = 3600.0
    real(c_double), parameter :: SEC_MIN = 60.0
    real(c_double), parameter :: DAY_SEC = 86400.0

    ! converted from hex representations
    integer(c_int16_t), parameter :: HEADER_UNDEFINED = -1
    integer(c_int16_t), parameter :: HEADER_METADATA = 17485
    integer(c_int16_t), parameter :: HEADER_USAGEBLOCK = 16981
    integer(c_int16_t), parameter :: HEADER_ACCEL = 22593
    integer(c_int16_t), parameter :: HEADER_GYRO = 22855
    integer(c_int16_t), parameter :: HEADER_SESSIONINFO = 18771

    ! matching enum from read_binary_imu.h
    integer(c_int), parameter :: AX_READ_E_NONE = 0
    integer(c_int), parameter :: AX_READ_E_BAD_HEADER = 1
    integer(c_int), parameter :: AX_READ_E_MISMATCH_N_AXES = 2
    integer(c_int), parameter :: AX_READ_E_INVALID_BLOCK_SAMPLES = 3
    integer(c_int), parameter :: AX_READ_E_BAD_AXES_PACKED = 4
    integer(c_int), parameter :: AX_READ_E_BAD_PACKING_CODE = 5
    integer(c_int), parameter :: AX_READ_E_BAD_CHECKSUM = 6
    integer(c_int), parameter :: AX_READ_E_BAD_LENGTH_ZERO_TIMESTAMPS = 7

contains

    ! =============================================================================================
    ! axivity_close : close an open axivity file. Needed to be able to close from C
    ! =============================================================================================
    subroutine axivity_close(info) bind(C, name="axivity_close")
        type(FileInfo_t), intent(in) :: info

        close(info%N)
    end subroutine

    ! =============================================================================================
    ! axivity_read_header : read the header (first 1024 bytes) plus a little bit for error checking
    ! =============================================================================================
    subroutine axivity_read_header(flen, file, finfo, ierr) bind(C, name="axivity_read_header")
        integer(c_long), intent(in) :: flen  ! length of the file name string
        character(kind=c_char), intent(in) :: file(flen)  ! file name
        type(FileInfo_t), intent(inout) :: finfo  ! file info storage structure
        integer(c_int), intent(inout) :: ierr  ! error tracking/returning
        ! local
        type(metadata) :: hdr
        integer(c_long) :: itmp, i
        character(960) :: annotation_block
        character(flen) :: file_
        integer(c_int8_t) :: numAxesBps
        ! for file size
        integer(4) :: fstat_vals(13), fstat_err

        ! needed for converting c strings
        do i=1_c_long, flen
            file_(i:i + 1) = file(i)
        end do

        ! initialize
        finfo%N = 17_c_int
        finfo%tLast = -1000._c_double
        finfo%n_bad_blocks = 0_c_long

        open(unit=finfo%N, file=file_, access="stream", action="read")

        ! get file information
        call fstat(finfo%N, fstat_vals, fstat_err)

        if (fstat_err == 0) then
            finfo%nblocks = int(fstat_vals(8), c_int) / 512_c_int
            ! if errors, taken care of by c program which will error with unset nblocks value
        end if

        ! read the whole header
        read(finfo%N, pos=1) hdr
        ! read the annotation block
        read(finfo%N, pos=65) annotation_block

        if (hdr%header /= "MD") then
            ierr = AX_READ_E_BAD_HEADER
            return
        end if

        ! create device ID with unsigned conversion if necessary
        if (hdr%deviceID < 0) then
            finfo%deviceID = int(hdr%deviceID, c_long) + 2_c_long**16_c_long
        else
            finfo%deviceID = int(hdr%deviceID, c_long)
        end if
        if (hdr%upperDeviceID /= -1) then
            if (hdr%upperDeviceID < 0) then
                itmp = int(hdr%upperDeviceID, c_long) + 2_c_long**16_c_long
            else
                itmp = int(hdr%upperDeviceID, c_long)
            end if
            finfo%deviceID = ior(finfo%deviceID, ishft(itmp, 16))
        end if

        ! number of axes
        if ((hdr%sensorConfig == 0) .or. (hdr%sensorConfig == -1)) then
            finfo%axes = 3_c_int8_t  ! accel only
        else
            if (iand(ishft(hdr%sensorConfig, -4), z"0f") /= 0) then
                finfo%axes = 9_c_int8_t  ! magnetometer enabled
            else
                finfo%axes = 6_c_int8_t
            end if
        end if

        ! read ahead into the data block to check number of axes and get samples per block
        read(finfo%N, pos=1025 + 25) numAxesBps
        read(finfo%N, pos=1025 + 28) finfo%count

        if (finfo%axes /= iand(ishft(numAxesBps, -4), z"0f")) then
            ierr = AX_READ_E_MISMATCH_N_AXES
            return
        end if
        if ((finfo%axes == 3) .and. ((finfo%count /= 80) .and. (finfo%count /= 120))) then
            ierr = AX_READ_E_INVALID_BLOCK_SAMPLES
            return
        else if ((finfo%axes == 6) .and. (finfo%count /= 40)) then
            ierr = AX_READ_E_INVALID_BLOCK_SAMPLES
            return
        else if ((finfo%axes == 9) .and. (finfo%count > 26)) then
            ierr = AX_READ_E_INVALID_BLOCK_SAMPLES
            return
        end if

        ! sampling frequency
        finfo%frequency = 3200. / shiftl(1, 15 - iand(hdr%samplingRate, z'0f'))
    end subroutine

    ! =============================================================================================
    ! axivity_read_block : read a single block (512 bytes) of data from an axivity file and 
    !   put the data into its respective storage arrays
    ! =============================================================================================
    subroutine axivity_read_block(info, pos, imudata, timestamps, temp, ierr) bind(C, name="axivity_read_block")
        type(FileInfo_t), intent(inout) :: info  ! file information storage structure
        ! position of the file to read from. should be a multiple of 512
        integer(c_int), intent(in) :: pos
        ! imu data array. shape(3/6/9, # samples). Order is [Gy]Ax[Mag]
        real(c_double), intent(out) :: imudata(info%axes, info%count * (info%nblocks - 2))
        ! timestamp data array
        real(c_double), intent(out) :: timestamps(info%count * (info%nblocks - 2))
        real(c_double), intent(out) :: temp(info%count * (info%nblocks - 2))  ! light data array
        integer(c_int), intent(out) :: ierr  ! error recording and returning to calling function
        ! local
        type(datapacket) :: pkt
        real(c_double) :: accelScale, gyroScale, magScale
        real(c_double) :: block_temp
        integer(c_short) :: wordsum
        integer(c_int32_t) :: i1, i2
        integer(c_int16_t) :: rawData(info%axes, info%count), k, checksum
        integer(c_int8_t) :: bps, expnt
        integer(c_int32_t), allocatable :: packedData(:)

        read(info%N, pos=pos) pkt
        if ((pkt%header /= HEADER_ACCEL) .or. (pkt%length /= 508_c_int16_t)) then
            ierr = AX_READ_E_NONE  ! no error just returning
            info%n_bad_blocks = info%n_bad_blocks + 1_c_long
            ! set the last time to 0 so that we dont use it to adjust timestamps for
            ! the next block
            info%tLast = -1.0
            return
        end if

        ! initialize for later
        wordsum = 0_c_short

        accelScale = 256._c_double  ! 1g = 256
        gyroScale = 2000._c_double  ! 32768 = 2000dps
        magScale = 16._c_double     ! 1uT = 16
        ! light is LS 10 bits, accel scale 3 msb, gyro scale next 3
        ! light(pkt%sequenceID + 1) = iand(pkt%light, z"3ff")
        accelScale = real(2**(8 + iand(ishft(pkt%light, -13), z"07")), c_double)
        gyroScale = real(8000 / (2**iand(ishft(pkt%light, -10), z"07")), c_double)

        gyroScale = 32768. / gyroScale

        ! get the temperature
        block_temp = iand(pkt%temperature, z"3ff")

        ! check bytes per sample
        if (iand(pkt%numAxesBPS, z"0f") == 0) then
            ! check # of axes
            if (info%axes /= 3) then
                ierr = AX_READ_E_BAD_AXES_PACKED
                return
            end if
            bps = 4_c_int8_t
        else if (iand(pkt%numAxesBPS, z"0f") == 2) then
            bps = info%axes * 2_c_int8_t
        else
            ierr = AX_READ_E_BAD_PACKING_CODE
            return
        end if

        ! extract the data values
        if (bps == 4) then
            if (info%count /= 120) then
                ierr = AX_READ_E_INVALID_BLOCK_SAMPLES
                return
            end if

            allocate(packedData(info%count))

            read(info%N, pos=pos+30) packedData
            ! read the checksum
            read(info%N, pos=pos + 510) checksum

            ! make sure the checksum is good
            call data_packet_sum_packed(pkt, packedData, checksum, wordsum)

            if (wordsum /= 0) then
                info%n_bad_blocks = info%n_bad_blocks + 1_c_long
                ierr = AX_READ_E_NONE  ! no error, just skip populating the block with data
                ! set the last time to 0 so that we dont use it to adjust timestamps for
                ! the next block
                info%tLast = -1.0
                return
            end if

            do k=1, info%count
                expnt = int(ishft(packedData(k), -30), c_int8_t)
                if (expnt < 0) expnt = expnt + 4_c_int8_t
                ! unpack data
                rawData(1, k) = int(iand(ishft(packedData(k), 6), z"ffc0"), c_int16_t)
                rawData(2, k) = int(iand(ishft(packedData(k), -4), z"ffc0"), c_int16_t)
                rawData(3, k) = int(iand(ishft(packedData(k), -14), z"ffc0"), c_int16_t)

                rawData(:, k) = shifta(rawData(:, k), 6 - expnt)
            end do

            deallocate(packedData)
        else  ! 16 bit signed values
            info%count = min(info%count, 80_c_int16_t)
            if (info%count < 0) then
                ierr = AX_READ_E_INVALID_BLOCK_SAMPLES
                return
            end if
            
            read(info%N, pos=pos+30) rawData
            read(info%N, pos=pos+510) checksum

            ! make sure block checksum is good
            call data_packet_sum_unpacked(pkt, rawData, checksum, wordsum)
            if (wordsum /= 0) then
                info%n_bad_blocks = info%n_bad_blocks + 1_c_long
                ierr = AX_READ_E_NONE  ! no error, just skip populating the block with data
                return
            end if
        end if

        ! i1 = (pkt%sequenceID - info%n_bad_blocks) * info%count + 1_c_int16_t
        ! above would result in data gaps that would result in bad timestamps
        ! going forward bad blocks will be left as all 0 values, and timestamps
        ! will be fixed later
        i1 = pkt%sequenceID * info%count + 1_c_int16_t
        i2 = i1 + info%count

        ! set the temperature for the block, and convert to deg C
        temp(i1:i2) = (block_temp - 171.0) / 3.142

        ! get the data into its final storage
        if (info%axes == 3) then
            imudata(:, i1:i2) = rawData / accelScale
        else if (info%axes == 6) then
            imudata(1:3, i1:i2) = rawData(1:3, :) / gyroScale
            imudata(4:6, i1:i2) = rawData(4:6, :) / accelScale
        else if (info%axes == 9) then
            imudata(1:3, i1:i2) = rawData(1:3, :) / gyroScale
            imudata(4:6, i1:i2) = rawData(4:6, :) / accelScale
            imudata(7:9, i1:i2) = rawData(7:9, :) / magScale
        end if

        ! convert and create the timestamps
        call get_time(info, pkt, timestamps(i1:i2))

        ierr = AX_READ_E_NONE
    end subroutine

    ! =============================================================================================
    ! get_time : creates the timestamps from the singular timestamp and offsets provided per data
    ! block. 
    ! =============================================================================================
    subroutine get_time(info, pkt, time)

        type(FileInfo_t), intent(inout) :: info  ! file info storage structure
        type(datapacket), intent(in) :: pkt  ! data block info storage structure
        ! part of time array corresponding to current block
        real(c_double), intent(out) :: time(info%count)
        ! local
        integer(c_long), parameter :: EPOCH = 2440588_c_long
        integer(c_long) :: i, days, year, month, day
        type(time_t) :: t
        real(c_double) :: freq, t0, t1, tDelta

        year  = iand(shifta(pkt%timestamp, 26), z'3f') + 2000_c_long  ! since 0 CE
        month = iand(shifta(pkt%timestamp, 22), z'0f')
        day   = iand(shifta(pkt%timestamp, 17), z'1f')
        t%hour  = iand(shifta(pkt%timestamp, 12), z'1f')
        t%min   = iand(shifta(pkt%timestamp, 6), z'3f')
        t%sec   = iand(pkt%timestamp, z'3f')
        t%msec  = 0_c_long

        ! compute days
        days = day - 32075_c_long + 1461_c_long * (year + 4800_c_long + (month - 14_c_long) / 12_c_long) / 4_c_long &
            + 367_c_long * (month - 2_c_long - (month - 14_c_long) / 12_c_long * 12_c_long) / 12_c_long &
            - 3_c_long * ((year + 4900_c_long + (month - 14_c_long) / 12_c_long) / 100_c_long) / 4_c_long
        
        ! remove days to 1970
        days = days - EPOCH
        ! convert to seconds
        t0 = days * DAY_SEC
        ! add hours, minutes, seconds
        t0 = t0 + (t%hour * SEC_HOUR) + (t%min * SEC_MIN) + real(t%sec, c_double)

        freq = 3200. / shiftl(1, 15 - iand(pkt%sampleRate, z'0f'))
        if (freq <= 0.) freq = 1.
        
        t0 = t0 - pkt%timestampOffset / freq
        t1 = t0 + pkt%sampleCount / freq
        ! for indexing. Can be negative, as it just gets added into the current timestamp
        t%msec = int(-pkt%timestampOffset / freq * 1000, c_long)

        if ((info%tLast > 0.) .and. ((t0 - info%tLast) < 1.)) then
            t%msec = t%msec - int((t0 - info%tLast) * 1000, c_long)  ! convert to microseconds
            t0 = info%tLast
        end if
        info%tLast = t1

        tDelta = (t1 - t0) / pkt%sampleCount
        time = (/ (t0 + i * tDelta, i=0, info%count - 1) /)
    end subroutine

    ! =============================================================================================
    ! adjust_timestamps : adjust timestamps for missing/bad data blocks
    ! =============================================================================================
    subroutine adjust_timestamps(info, timestamps, ierr) bind(C, name="adjust_timestamps")
        type(FileInfo_t), intent(inout) :: info  ! file information storage structure
        ! timestamp data array
        real(c_double), intent(inout) :: timestamps(info%count * (info%nblocks - 2))
        integer(c_int), intent(out) :: ierr  ! error recording and returning to calling function
        ! local
        ! for starts and lengths we can make some assumptions about that
        ! only full blocks should have 0 values, so we can have smaller arrays
        ! but add a little bit of buffer space
        integer(c_int) :: starts(info%nblocks + 10)
        integer(c_int) :: lengths(info%nblocks + 10)
        integer(c_int) :: n, i, j, i_start, curr_len
        real(c_double) :: t0, t1, ta, tb, delta_t

        n = info%count * (info%nblocks - 2)  ! for easier referencing

        curr_len = 0  ! initialize to avoid warnings

        i_start = 1  ! keep track of where we are
        if (timestamps(1) == 0._c_double) then
            starts(1) = 1
            i_start = 2
        end if
        do i=2, n
            if ((timestamps(i) == 0._c_double) .and. (timestamps(i - 1) /= 0._c_double)) then
                starts(i_start) = i
                i_start = i_start + 1
                curr_len = 1
            else if (timestamps(i) == 0._c_double) then
                curr_len = curr_len + 1
            else if ((timestamps(i) /= 0._c_double) .and. (timestamps(i - 1) == 0._c_double)) then
                lengths(i_start - 1) = curr_len
            end if
        end do

        ! iterate over starts and fill
        do i=1, i_start - 1
            if (mod(lengths(i), info%count) /= 0) then
                ierr = AX_READ_E_BAD_LENGTH_ZERO_TIMESTAMPS
                return
            end if
            ! start time
            ta = timestamps(starts(i) - 1) + 1. / info%frequency  ! timestamp if this is NOT the first block
            tb = timestamps(starts(i) + lengths(i)) - lengths(i) / info%frequency
            t0 = merge(ta, tb, starts(i) > 1_c_int16_t)

            ! end time
            ta = timestamps(starts(i) + lengths(i))  ! - 1. / info%frequency
            tb = timestamps(starts(i)) + lengths(i) / info%frequency
            t1 = merge(ta, tb, starts(i) + lengths(i) <= n)

            delta_t = (t1 - t0) / lengths(i)

            timestamps(starts(i)) = t0
            do j=1, lengths(i) - 1
                timestamps(starts(i) + j) = t0 + j * delta_t
            end do
        end do

    end subroutine

    ! =============================================================================================
    ! data_packet_sum_unpacked : computes the total checksum calculation for a data block if the
    !   data is in the unpacked format (ie 2bytes per sample)
    ! =============================================================================================
    subroutine data_packet_sum_unpacked(pkt, raw_data, checksum, val)
        type(datapacket), intent(inout) :: pkt  ! data block info storage structure
        integer(c_int16_t), intent(in) :: raw_data(:, :)  ! read in raw data 
        integer(c_int16_t), intent(in) :: checksum  ! checksum value from the block
        integer(c_short), intent(out) :: val  ! checksum storage
        ! local
        integer :: i, j

        val = 0

        call plus16(pkt%header, val)
        call plus16(pkt%length, val)
        call plus16(pkt%deviceID, val)
        call plus32(pkt%sessionID, val)
        call plus32(pkt%sequenceId, val)
        call plus32(pkt%timestamp, val)
        call plus16(pkt%light, val)
        call plus16(pkt%temperature, val)
        call plus8(pkt%events, pkt%battery, val)
        call plus8(pkt%sampleRate, pkt%numAxesBPS, val)
        call plus16(pkt%timestampOffset, val)
        call plus16(pkt%sampleCount, val)

        do j=1, size(raw_data, 2)
            do i=1, size(raw_data, 1)
                call plus16(raw_data(i, j), val)
            end do
        end do

        val = val + checksum

        val = not(val) + 1_c_short
    end subroutine

    ! =============================================================================================
    ! data_packet_sum_unpacked : computes the total checksum calculation for a data block if the
    !   data is in the packed format (accel data only, 4 bytes per 3 samples)
    ! =============================================================================================
    subroutine data_packet_sum_packed(pkt, raw_packed, checksum, val)
        type(datapacket), intent(inout) :: pkt  ! data block info storage structure
        integer(c_int32_t), intent(in) :: raw_packed(120)  ! raw packed data samples
        integer(c_int16_t), intent(in) :: checksum  ! checksum value from the block
        integer(c_short), intent(out) :: val  ! checksum storage
        ! local
        integer :: i

        val = 0

        call plus16(pkt%header, val)
        call plus16(pkt%length, val)
        call plus16(pkt%deviceID, val)
        call plus32(pkt%sessionID, val)
        call plus32(pkt%sequenceId, val)
        call plus32(pkt%timestamp, val)
        call plus16(pkt%light, val)
        call plus16(pkt%temperature, val)
        call plus8(pkt%events, pkt%battery, val)
        call plus8(pkt%sampleRate, pkt%numAxesBPS, val)
        call plus16(pkt%timestampOffset, val)
        call plus16(pkt%sampleCount, val)

        do i=1, 120  ! packed data will always be 120
            call plus32(raw_packed(i), val)
        end do

        val = val + checksum

        val = not(val) + 1_c_short  ! bitwise not and add 1

    end subroutine

    ! =============================================================================================
    ! plus16 : subroutine for adding a 2byte word to the checksum value
    ! =============================================================================================
    subroutine plus16(x, val)
        integer(c_int16_t), intent(in) :: x
        integer(c_short), intent(inout) :: val

        ! don't need to account for unsigned, because it is taken care of by the truncation/swapping
        ! around the max value for signed.
        ! ie - bits are same even if representation isn't
        val = val + int(x, c_short)
    end subroutine

    ! =============================================================================================
    ! plus8 : subroutine for adding 2 1-byte values to the checksum value
    ! =============================================================================================
    subroutine plus8(x, y, val)
        integer(c_int8_t), intent(in) :: x, y
        integer(c_short), intent(inout) :: val

        val = val + int(x, c_short)
        val = val + ishft(int(y, c_short), 8)
    end subroutine

    ! =============================================================================================
    ! plus32 : subroutine for adding 1 4-byte value to the checksum value
    ! =============================================================================================
    subroutine plus32(x, val)
        integer(c_int32_t), intent(in) :: x
        integer(c_short), intent(inout) :: val

        val = val + int(iand(x, z"ffff"), c_short)
        val = val + int(iand(ishft(x, -16), z"ffff"), c_short)

    end subroutine

end module axivity
