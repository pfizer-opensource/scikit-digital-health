#include "read_binary_imu.h"

/*
---------------------
CONSTANTS
---------------------
*/
// gt3x log record types
enum LogRecordType {
    RECORDTYPE_ACTIVITY         = 0x00, // One second of raw activity samples packed into 12-bit values in YXZ order.
    RECORDTYPE_BATTERY          = 0x02, // Battery voltage in millivolts as a little-endian unsigned short (2 bytes).
    RECORDTYPE_EVENT            = 0x03, // Logging records used for internal debugging.
    RECORDTYPE_VHEART_RATE_BPM  = 0x04, // Heart rate average beats per minute (BPM) as one byte unsigned integer.
    RECORDTYPE_LUX              = 0x05, // Lux value as a little-endian unsigned short (2 bytes).
    RECORDTYPE_METADATA         = 0x06, // Arbitrary metadata content. The first record in every log is contains subject data in JSON format.
    RECORDTYPE_TAG              = 0x07, // 13 Byte Serial, 1 Byte Tx Power, 1 Byte (signed) RSSI
    RECORDTYPE_EPOCH            = 0x09, // EPOCH	60-second epoch data
    RECORDTYPE_HEART_RATE_ANT   = 0x0B, // Heart Rate RR information from ANT+ sensor.
    RECORDTYPE_EPOCH2           = 0x0C, // 60-second epoch data
    RECORDTYPE_CAPSENSE         = 0x0D, // Capacitive sense data
    RECORDTYPE_HEART_RATE_BLE   = 0x0E, // Bluetooth heart rate information (BPM and RR). This is a Bluetooth standard format.
    RECORDTYPE_EPOCH3           = 0x0F, // 60-second epoch data
    RECORDTYPE_EPOCH4           = 0x10, // 60-second epoch data
    RECORDTYPE_PARAMETERS       = 0x15, // Records various configuration parameters and device attributes on initialization.
    RECORDTYPE_SENSOR_SCHEMA    = 0x18, // This record allows dynamic definition of a SENSOR_DATA record format.
    RECORDTYPE_SENSOR_DATA      = 0x19, // This record stores sensor data according to a SENSOR_SCHEMA definition.
    RECORDTYPE_ACTIVITY2        = 0x1A  // One second of raw activity samples as little-endian signed-shorts in XYZ order.
};

/* Value of .NET ticks at EPOCH (1970/01/01 00:00:00) */
const long long NET_TICKS_1970 = 621355968000000000LL;
/* Record Separator */
#define RECORD_SEP 30

/* Needed to decode float parameter values */
// const double PARAM_FLOAT_MINIMUM = 0.00000011920928955078125;  /* 2^-23 */
#define PARAM_FLOAT_MAXIMUM 8388608.0                             /* 2^23  */
#define PARAM_ENCODED_MINIMUM 0x00800000
#define PARAM_ENCODED_MAXIMUM 0x007FFFFF
#define PARAM_SIGNIFICAND_MASK 0x00FFFFFFu
// const int PARAM_EXPONENT_MINIMUM = -128;
// const int PARAM_EXPONENT_MAXIMUM = 127;
#define PARAM_EXPONENT_MASK 0xFF000000u
#define PARAM_EXPONENT_OFFSET 24
/*
---------------------
END CONSTANTS
---------------------
*/

/* parse a version number from text */
void parse_version(char *vers, AG_Version_t *vers_info)
{
    char *end;

    vers_info->major = strtol(vers, &end, 10);
    vers = end + 1;  /* increment to after period */
    vers_info->minor = strtol(vers, &end, 10);
    vers = end + 1;
    vers_info->build = strtol(vers, &end, 10);
}

/* parse net ticks into a unix timestamp */
double parse_NET_ticks(char *tick_str)
{
    long long ticks = strtoll(tick_str, NULL, 10);
    if (ticks == 0LL) return 0.0;
    /* remove ticks from 1970, and divide to get seconds */
    return (double)(ticks - NET_TICKS_1970) / 1.0e7;
}

/* get the number of samples based on start and end times */
void get_n_samples(AG_Info_t *info, AG_SensorInfo_t *sensor)
{
    /* end time is last sample time if not 0, or stop time if not 0, or download time */
    double end = 0.0;
    if (sensor->last_sample_time > 0.0)
        end = sensor->last_sample_time;
    else
    {
        if (sensor->stop_time > 0)
            end = sensor->stop_time;
        else
            end = sensor->download_time;
    }

    info->samples = (int)lround(end - sensor->start_time) * sensor->sample_rate;
    info->n_days = (int)ceil((end - sensor->start_time) / (60 * 60 * 24));  /* round up # of days */

    if (info->samples <= 0)
    {
        fprintf(stderr, "Invalid # of samples estimated, using maximum samples (100 days)\n");
        info->samples = 100 * 24 * 60 * 60 * sensor->sample_rate;
        info->n_days = 100;  /* set to 100 days */
    }
}

