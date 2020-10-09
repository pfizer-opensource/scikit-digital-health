module axvty
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
    character(2) :: header                  ! @1  [2] ASCII "AX", little endian (0x5841)
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
    ! integer(c_int8_t) :: data(480)          ! @31 [480] raw sample data
    ! integer(c_int16_t) :: checksum          ! @511 [2] checksum of packet
  end type datapacket

  type, bind(c) :: FileInfo_t
    integer(c_long) :: base
    integer(c_long) :: period
    integer(c_long) :: deviceID
    integer(c_long) :: sessionID
    integer(c_int) :: nblocks
    integer(c_int8_t) :: axes
    integer(c_int16_t) :: count
    real(c_double) :: tLast
    integer(c_int) :: N
    real(c_double) :: frequency
  end type FileInfo_t

  public :: fread_cwa
  private :: read_header, read_block, get_time

contains

  subroutine fread_cwa(flen, file, finfo, imudata, timestamps, indx, light) bind(C, name="fread_cwa")
    integer(c_long), intent(in) :: flen
    character(kind=c_char), intent(in) :: file(flen)
    type(FileInfo_t), intent(inout) :: finfo
    ! integer(c_int), intent(in) :: nblocks
    ! integer(c_int8_t), intent(in) :: numAxes
    ! integer(c_int16_t), intent(in) :: sampleCount
    real(c_double), intent(out) :: imudata(finfo%axes, finfo%count * (finfo%nblocks-2))
    real(c_double), intent(out) :: timestamps(finfo%count * (finfo%nblocks-2))
    integer(c_long), intent(out) :: indx(finfo%nblocks-2), light(finfo%nblocks-2)
    ! local
    integer(c_long) :: i
    integer(c_int) :: j, position
    character(flen) :: file_

    do i=1_c_long, flen
      file_(i:i+1_c_long) = file(i)
    end do

    ! initialize
    finfo%N = 17_c_int
    finfo%tLast = -1000._c_double

    indx = -2_c_long * finfo%nblocks * finfo%count
    ! end of intialize

    open(unit=finfo%N, file=file_, access="stream", action="read")

    ! read the header
    call read_header(finfo)

    do j=3_c_int, finfo%nblocks
      position = (j-1_c_int) * 512_c_int + 1_c_int
      call read_block(finfo, position, imudata, timestamps, indx, light)
    end do

    close(finfo%N)

  end subroutine

  subroutine read_header(info)
    type(FileInfo_t), intent(inout) :: info
    ! local
    type(metadata) :: hdr
    integer(c_long) :: itmp
    character(960) :: annotationBlock

    read(info%N, pos=1) hdr
    read(info%N, pos=65) annotationBlock

    if (hdr%header /= "MD") return

    ! create device ID with unsigned conversion if necessary
    if (hdr%deviceID < 0_c_int16_t) then
      info%deviceID = int(hdr%deviceID, c_long) + 2_c_long**16_c_long
    else
      info%deviceID = int(hdr%deviceID, c_long)
    end if
    if (hdr%upperDeviceID /= -1_c_int16_t) then
      if (hdr%upperDeviceID < 0_c_int16_t) then
        itmp = int(hdr%upperDeviceID, c_long) + 2_c_long**16_c_long
      else
        itmp = int(hdr%upperDeviceID, c_long)
      end if
      info%deviceID = ior(info%deviceID, ishft(itmp, 16))
    end if

    ! session ID
    if (hdr%sessionID < 0_c_int32_t) then
      info%sessionID = int(hdr%sessionID, c_long) + 2_c_long**32_c_long
    else
      info%sessionID = int(hdr%sessionID, c_long)
    end if

    ! sampling frequency
    info%frequency = 3200._c_double / shiftl(1, 15_c_int8_t - iand(hdr%samplingRate, z'0f'))
  end subroutine

  subroutine read_block(info, pos, imudata, timestamps, indx, light)
    type(FileInfo_t), intent(inout) :: info
    integer(c_int), intent(in) :: pos
    real(c_double), intent(out) :: imudata(info%axes, info%count * (info%nblocks-2))
    real(c_double), intent(out) :: timestamps(info%count * (info%nblocks-2))
    integer(c_long), intent(out) :: indx(info%nblocks-2), light(info%nblocks-2)
    ! local
    type(datapacket) :: pkt
    real(c_double) :: accelScale, gyroScale, magScale
    integer(c_int32_t) :: i1, i2
    integer(c_int16_t) :: words(256), rawData(info%axes, info%count), k
    integer(c_int8_t) :: bps, expnt
    integer(c_int32_t), allocatable :: packedData(:)

    read(info%N, pos=pos) pkt
    if ((pkt%header /= "AX") .OR. (pkt%length /= 508_c_int16_t)) return

    ! read the whole block to check 16bit wordwise sum
    read(info%N, pos=pos) words
    if (sum(words) /= 0_c_int16_t) then
      print *, '[ERROR: checksum not equal to 0]'
      return
    end if

    accelScale = 256._c_double  ! 1g = 256
    gyroScale = 2000._c_double  ! 32768 = 2000dps
    magScale = 16._c_double     ! 1uT = 16
    ! light is LS 10 bits, accel scale 3MSB, gyro scale next 3
    light(pkt%sequenceID+1) = iand(pkt%light, z'3ff')
    accelScale = real(2**(8_c_int16_t + iand(ishft(pkt%light, -13), z'07')), c_double)
    gyroScale = real(8000_c_int16_t / (2**iand(ishft(pkt%light, -10), z'07')), c_double)

    gyroScale = 32768._c_double / gyroScale

    ! check bytes per sample
    if (iand(pkt%numAxesBPS, z'0f') == 0_c_int8_t) then
      ! check # of axes
      if (info%axes /= 3) return
      bps = 4_c_int8_t
    else if (iand(pkt%numAxesBPS, z'0f') == 2_c_int8_t) then
      bps = info%axes * 2_c_int8_t
    else
      return
    end if

    ! extract the data values
    if (bps == 4) then
      if (info%count /= 120_c_int16_t) return

      allocate(packedData(info%count))

      ! packedData = transfer(pkt%data, packedData)
      read(info%N, pos=pos+30) packedData

      do k=1, info%count
        expnt = int(ishft(packedData(k), -30), c_int8_t)
        if (expnt < 0) expnt = expnt + 4_c_int8_t
        ! unpack data
        rawData(1, k) = int(iand(ishft(packedData(k), 6), z'ffc0'), c_int16_t)
        rawData(2, k) = int(iand(ishft(packedData(k), -4), z'ffc0'), c_int16_t)
        rawData(3, k) = int(iand(ishft(packedData(k), -14), z'ffc0'), c_int16_t)

        rawData(:, k) = shifta(rawData(:, k), 6 - expnt)
      end do
      
      deallocate(packedData)
    else  ! 16 bit signed values
      if (info%count > 80_c_int16_t) then
        info%count = 80_c_int16_t
      else if (info%count < 0_c_int16_t) then
        return
      end if

      ! rawData = reshape(transfer(pkt%data, rawData), shape(rawData))
      read(info%N, pos=pos+30) rawData
    end if

    i1 = pkt%sequenceID * info%count + 1_c_int16_t
    i2 = i1 + info%count

    ! get the data into its final storage
    if (info%axes == 3_c_int8_t) then
      imudata(:, i1:i2) = rawData / accelScale
    else if (info%axes == 6_c_int8_t) then
      imudata(1:3, i1:i2) = rawData(1:3, :) / gyroScale
      imudata(4:6, i1:i2) = rawData(4:6, :) / accelScale
    else if (info%axes == 9_c_int8_t) then
      imudata(1:3, i1:i2) = rawData(1:3, :) / gyroScale
      imudata(4:6, i1:i2) = rawData(4:6, :) / accelScale
      imudata(7:9, i1:i2) = rawData(7:9, :) / magScale
    end if

    ! convert and create the timestamps
    call get_time(info, pkt, timestamps(i1:i2), indx)
  end subroutine


  subroutine get_time(info, pkt, time, indx)
    type(FileInfo_t), intent(inout) :: info
    type(datapacket), intent(in) :: pkt
    real(c_double), intent(out) :: time(:)
    integer(c_long), intent(out) :: indx(:)
    ! local for time conversion
    integer(c_long), parameter :: EPOCH = 2440588_c_long
    real(c_double), parameter :: SECPERMIN = 60._c_double
    real(c_double), parameter :: SECPERHOUR = 3600._c_double
    real(c_double), parameter :: SECPERDAY = 86400._c_double
    integer(c_long) :: days, year, month, day, hour, mnt, sec
    ! local
    real(c_double) :: t0, t1, freq, tDelta
    real(c_double) :: secBase, secPeriod, th0, th1, tdiff
    integer(c_int16_t) :: i

    secBase = info%base * SECPERHOUR
    secPeriod = mod(info%base + info%period, 24_c_long) * SECPERHOUR

    year  = iand(shifta(pkt%timestamp, 26), z'3f') + 2000_c_long  ! since 0 CE
    month = iand(shifta(pkt%timestamp, 22), z'0f')
    day   = iand(shifta(pkt%timestamp, 17), z'1f')
    hour  = iand(shifta(pkt%timestamp, 12), z'1f')
    mnt   = iand(shifta(pkt%timestamp, 6), z'3f')
    sec   = iand(pkt%timestamp, z'3f')

    ! compute days
    days = day - 32075_c_long + 1461_c_long * (year + 4800_c_long + (month - 14_c_long) / 12_c_long) / 4_c_long &
        + 367_c_long * (month - 2_c_long - (month - 14_c_long) / 12_c_long * 12_c_long) / 12_c_long &
        - 3_c_long * ((year + 4900_c_long + (month - 14_c_long) / 12_c_long) / 100_c_long) / 4_c_long
    
    ! remove days to 1970
    days = days - EPOCH
    ! conver to seconds
    t0 = days * SECPERDAY
    ! add hours, minutes, and seconds
    th0 = (hour * SECPERHOUR) + (mnt * SECPERMIN) + real(sec, c_double)
    t0 = t0 + th0

    freq = 3200._c_double / shiftl(1, 15_c_int8_t - iand(pkt%sampleRate, z'0f'))
    if (freq <= 0._c_double) freq = 1._c_double

    t0 = t0 - pkt%timestampOffset / freq
    t1 = t0 + pkt%sampleCount / freq
    ! times for only the part of the day (hour fractional)
    th0 = th0 - pkt%timestampOffset / freq
    th1 = th0 + pkt%sampleCount / freq

    tdiff = 0._c_double
    if ((info%tLast > 0._c_double) .AND. ((t0 - info%tLast) < 1._c_double)) then
      th0 = th0 - (t0 - info%tLast)
      t0 = info%tLast
    end if
    info%tLast = t1
    
    tDelta = (t1 - t0) / pkt%sampleCount
    time = (/ (t0 + i * tDelta, i=0, info%count - 1_c_int16_t) /)

    ! check windowing
    if ((secPeriod >= th0) .AND. (secPeriod < th1)) then
      indx(pkt%sequenceID+1) = -(int(floor((secPeriod - th0) / tDelta, c_long)) + (pkt%sequenceID * info%count))
    end if
    if ((secBase >= th0) .AND. (secBase < th1)) then
      indx(pkt%sequenceID+1) = int(floor((secBase - th0) / tDelta, c_long)) + (pkt%sequenceID * info%count)
    end if
  end subroutine

end module