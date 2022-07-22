// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#include "read_binary_imu.h"
/* put this here so if we are not compiling ActiGraph we dont need zip.h */
#include <zip.h>

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

/* check if the file is using the old version */
int is_old_version(AG_SensorInfo_t *sensor)
{
    int serial = (strncmp(sensor->serial, "MRA", 3) == 0) || (strncmp(sensor->serial, "NEO", 3) == 0);
    int version = 0;
    if (sensor->firmware.major < 2)
    {
        version = 1;
    }
    else if (sensor->firmware.major >= 3)
    {
        version = 0;
    }
    else  /* major = 2 */
    {
        if (sensor->firmware.minor < 5)
        {
            version = 1;
        }
        else if (sensor->firmware.minor >= 6)
        {
            version = 0;
        }
        else  /* minor = 5 */
        {
            version = sensor->firmware.build == 0;
        }
    }
    return serial && version;
}

/*
Parse the info file in a GT3X archive

Parameters
----------
archive : zip_t
    Open GT3X archive file
pinfo : ParseInfo_t
    Struct for storing information relevant during the parsing of the file
sensor : GT3XInfo_t
    Struct for storing the information from the info file

Modifies
--------
sensor
info.samples
    Number of samples contained in the GT3X archive.
*/
int parse_info(zip_t *archive, AG_Info_t *info, AG_SensorInfo_t *sensor)
{
    /* get the file stats first */
    zip_stat_t stats;
    int err = 0;
    unsigned long bytes;
    err = zip_stat(archive, "info.txt", ZIP_FL_ENC_GUESS, &stats);
    /* make sure the size is valid */
    if (((stats.valid & ZIP_STAT_SIZE) != 0) && (err == 0))
    {
        bytes = stats.size;
    }
    else
    {
        return AG_READ_E_INFO_OPEN;
    }
    /* open the info text file from inside the zip archive */
    zip_file_t *info_file = zip_fopen(archive, "info.txt", ZIP_FL_ENC_GUESS);
    if (info_file == NULL) {
        return AG_READ_E_INFO_OPEN;
    }

    /* allocate buffer based on file size */
    char *buffer = malloc(bytes);
    if (!buffer)
    {
        return AG_READ_E_MALLOC;
    }

    int sep = 0;  /* line colon index */
    char *line = NULL;

    AG_DBGPRINT("Parsing info file ...");
    zip_fread(info_file, buffer, bytes);

    /* get the line token splitting on new lines */
    line = strtok(buffer, "\n");
    while (line)
    {
        sep = strcspn(line, ":") + 2;  // ": " 2 extra characters
        if (strstr(line, "Serial Number") != NULL)
        {
            memcpy(sensor->serial, &line[sep], 13);
            AG_DBGPRINT1("Serial number: %s", sensor->serial)
        }
        else if (strstr(line, "Firmware") != NULL)
        {
            parse_version(&line[sep], &(sensor->firmware));
            if (info->debug)
                fprintf(stdout, "Firmware: %d.%d.%d\n", sensor->firmware.major, sensor->firmware.minor, sensor->firmware.build);

        }
        else if (strstr(line, "Sample Rate") != NULL)
        {
            sensor->sample_rate = strtol(&line[sep], NULL, 10);
            AG_DBGPRINT1("Sample rate: %i", sensor->sample_rate)

        }
        else if (strstr(line, "Start Date") != NULL)
        {
            sensor->start_time = parse_NET_ticks(&line[sep]);
            AG_DBGPRINT1("Start date: %f", sensor->start_time)

        }
        else if (strstr(line, "Stop Date") != NULL)
        {
            sensor->stop_time = parse_NET_ticks(&line[sep]);
            AG_DBGPRINT1("Stop date: %f", sensor->stop_time)

        }
        else if (strstr(line, "Last Sample Time") != NULL)
        {
            sensor->last_sample_time = parse_NET_ticks(&line[sep]);
            AG_DBGPRINT1("Last sample time: %f", sensor->last_sample_time)

        }
        else if (strstr(line, "Download Date") != NULL)
        {
            sensor->download_time = parse_NET_ticks(&line[sep]);
            AG_DBGPRINT1("Download date: %f", sensor->download_time)

        }
        else if (strstr(line, "Acceleration Scale") != NULL)
        {
            sensor->accel_scale = strtod(&line[sep], NULL);
            AG_DBGPRINT1("Acceleration scale: %f\n", sensor->accel_scale)
        }

        line = strtok(NULL, "\n");
    }

    zip_fclose(info_file);
    free(buffer);

    AG_DBGPRINT("Computing # of samples ...")
    /* check if the file is the old version as this determines sample calculation */
    info->is_old_version = is_old_version(sensor);

    get_n_samples(info, sensor);
    /* if it is the old version, check file size to determine # of samples */
    if (info->is_old_version)
    {
        err = zip_stat(archive, "activity.bin", ZIP_FL_ENC_GUESS, &stats);
        if (((stats.valid & ZIP_STAT_SIZE) != 0) && (err = 0))
        {
            /* bytes -> bits, 36 bits per 3axis sample */
            int est_n_samples = (int)stats.size * 8 / 36;
            if (est_n_samples > info->samples)
                info->samples = (int)est_n_samples;
        }
        else
        {
            return AG_READ_E_INFO_STAT;
        }
    }
    AG_DBGPRINT1("Number of samles: %i", info->samples);

    return AG_READ_E_NONE;
}


/*
=========================
Activity parser helpers
=========================
*/
int bytes2samplesize(const unsigned char type, const unsigned short bytes)
{
    if (type == RECORDTYPE_ACTIVITY)
        return (bytes * 2) / 9;
    else if (type == RECORDTYPE_ACTIVITY2)
        return (bytes / 2) / 3;
    else
        return 0;
}

double decode_float_param(const uint32_t value)
{
    double significand, exponent;
    int32_t i32;

    /* handle parameters that are too big */
    if (value == PARAM_ENCODED_MAXIMUM)
        return DBL_MAX;
    else if (value == PARAM_ENCODED_MINIMUM)
        return -DBL_MAX;

    /* extract exponent */
    i32 = (int32_t)((value & PARAM_EXPONENT_MASK) >> PARAM_EXPONENT_OFFSET);
    if ((i32 & 0x80) != 0)
        i32 = (int32_t)((uint32_t)i32 | 0xFFFFFF00u);
    
    exponent = (double)i32;

    /* extract significand */
    i32 = (int32_t)(value & PARAM_SIGNIFICAND_MASK);
    if ((i32 & PARAM_ENCODED_MINIMUM) != 0)
        i32 = (int32_t)((uint32_t)i32 | 0xFF000000u);
    
    significand = (double)i32 / PARAM_FLOAT_MAXIMUM;

    /* calculate floating point value */
    return significand * pow(2.0, exponent);
}

void parse_parameters(zip_file_t *file, AG_SensorInfo_t *sensor, const unsigned short bytes, unsigned int *start_time)
{
    int n_params = bytes / 8;
    unsigned short address = 0;
    unsigned short key = 0;
    unsigned int value = 0;

    for (int i = 0; i < n_params; ++i)
    {
        zip_fread(file, &address, 2);
        zip_fread(file, &key, 2);
        zip_fread(file, &value, 4);

        if (address == 0)
        {
            /* accel scale is key 55 */
            if (key == 55)
            {
                if (value != 0)
                {
                    sensor->accel_scale = decode_float_param(value);
                }
            }
        }
        if (address == 1)
        {
            /* start time is key 12 */
            if (key == 12)
            {
                *start_time = value;  /* convert int UNIX time to double */
            }
        }
    }
}