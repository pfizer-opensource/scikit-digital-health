// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#include "gt3x.h"

// ---------
// Constants
// ---------

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

// Value of .NET ticks at EPOCH (1970/01/01 00:00:00)
const long long NET_TICKS_1970 = 621355968000000000LL;
// record separator
#define RECORD_SEPARATOR 30

// needed to decode float parameter values
// const double PARAM_FLOAT_MINIMUM = 0.00000011920928955078125;  /* 2^-23 */
#define PARAM_FLOAT_MAXIMUM 8388608.0                             /* 2^23  */
#define PARAM_ENCODED_MINIMUM 0x00800000
#define PARAM_ENCODED_MAXIMUM 0x007FFFFF
#define PARAM_SIGNIFICAND_MASK 0x00FFFFFFu
// const int PARAM_EXPONENT_MINIMUM = -128;
// const int PARAM_EXPONENT_MAXIMUM = 127;
#define PARAM_EXPONENT_MASK 0xFF000000u
#define PARAM_EXPONENT_OFFSET 24

// -------------
// END Constants
// -------------

/* parse a version number from text */
void parse_version(char *vers, Version_t *vers_info){
    char *end;

    vers_info->major = strtol(vers, &end, 10);
    vers = end + 1;  // increment to after the period
    vers_info->minor = strtol(vers, &end, 10);
    vers = end + 1;
    vers_info->build = strtol(vers, &end, 10);
}

/* parse net ticks into UNIX time (seconds since 1970) */
double parse_NET_ticks(char *tick_str){
    long long ticks = strtoll(tick_str, NULL, 10);
    if (ticks == 0LL) return 0.0;
    // remove ticks to 1970, and divide to get seconds
    return (double)(ticks - NET_TICKS_1970) / 1.0e7;
}

/* get the number of samples based on start and end times */
void get_n_samples(ParseInfo_t *pinfo, GT3XInfo_t *iinfo){
    // end time is last sample time if not 0, or stop time if not 0, or download time
    double end  = iinfo->last_sample_time > 0.0 ? iinfo->last_sample_time : (iinfo->stop_time > 0 ? iinfo->stop_time : iinfo->download_time);
    pinfo->samples = (int)lround((end - iinfo->start_time)) * iinfo->sample_rate;
    pinfo->n_days = (int)ceil((end - iinfo->start_time) / (60 * 60 * 24));  // round up # of days

    if (pinfo->samples <= 0){
        fprintf(stderr, "Invalid # of samples estimated, using maximum sampels (100 days)\n");
        pinfo->samples = 100 * 24 * 60 * 60 * iinfo->sample_rate;
        pinfo->n_days = 100;  // set to 100 days
    }
}

/* check if the file is using the old version */
bool is_old_version(GT3XInfo_t *iinfo){
    bool serial = (strncmp(iinfo->serial, "MRA", 3) == 0) || (strncmp(iinfo->serial, "NEO", 3) == 0);
    bool version = false;
    if (iinfo->firmware.major < 2){
        version = true;
    } else if (iinfo->firmware.major >= 3){
        version = false;
    } else {  // major is 2
        if (iinfo->firmware.minor < 5){
            version = true;
        } else if (iinfo->firmware.minor >= 6){
            version = false;
        } else { // minor is 5
            version = iinfo->firmware.build == 0 ? true : false;
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
iinfo : GT3XInfo_t
    Struct for storing the information from the info file

Modifies
--------
iinfo
pinfo.samples
    Number of samples contained in the GT3X archive.
*/
int parse_info(zip_t *archive, ParseInfo_t *pinfo, GT3XInfo_t *iinfo, int *ierr){
    // get the file stats first
    zip_stat_t stats;
    int err = 0;
    unsigned long bytes = 0;
    err = zip_stat(archive, "info.txt", ZIP_FL_ENC_GUESS, &stats);
    // make sure the size is valid
    if (((stats.valid & ZIP_STAT_SIZE) != 0) && (err == 0)){
        bytes = stats.size;
    } else {
        *ierr = E_INFO_STAT;
        return 0;
    }
    // open the info text file from inside the zip archive
    zip_file_t *info_file = zip_fopen(archive, "info.txt", ZIP_FL_ENC_GUESS);
    if (info_file == NULL) {
        *ierr = E_INFO_OPEN;
        return 0;
    }

    // allocate buffer based on file size
    char *buffer = malloc(bytes);
    if (!buffer){
        *ierr = E_MALLOC;
        return 0;
    }

    int sep = 0;  // line colon index
    char *line = NULL;

    DBGPRINT("Parsing info file ...")
    zip_fread(info_file, buffer, bytes);

    // get the line token splitting on new lines
    line = strtok(buffer, "\n");
    while (line){
        sep = strcspn(line, ":") + 2;  // ": " 2 extra characters
        if (strstr(line, "Serial Number") != NULL){
            memcpy(iinfo->serial, &line[sep], 13);
            DBGPRINT1("Serial number: %s", iinfo->serial)

        } else if (strstr(line, "Firmware") != NULL){
            parse_version(&line[sep], &(iinfo->firmware));
            if (pinfo->debug) fprintf(stdout, "Firmware: %d.%d.%d\n", iinfo->firmware.major, iinfo->firmware.minor, iinfo->firmware.build);

        } else if (strstr(line, "Sample Rate") != NULL){
            iinfo->sample_rate = strtol(&line[sep], NULL, 10);
            DBGPRINT1("Sample rate: %i", iinfo->sample_rate)

        } else if (strstr(line, "Start Date") != NULL){
            iinfo->start_time = parse_NET_ticks(&line[sep]);
            DBGPRINT1("Start date: %f", iinfo->start_time)

        } else if (strstr(line, "Stop Date") != NULL){
            iinfo->stop_time = parse_NET_ticks(&line[sep]);
            DBGPRINT1("Stop date: %f", iinfo->stop_time)

        } else if (strstr(line, "Last Sample Time") != NULL){
            iinfo->last_sample_time = parse_NET_ticks(&line[sep]);
            DBGPRINT1("Last sample time: %f", iinfo->last_sample_time)

        } else if (strstr(line, "Download Date") != NULL){
            iinfo->download_time = parse_NET_ticks(&line[sep]);
            DBGPRINT1("Download date: %f", iinfo->download_time)

        } else if (strstr(line, "Acceleration Scale") != NULL){
            iinfo->accel_scale = strtod(&line[sep], NULL);
            DBGPRINT1("Acceleration scale: %f\n", iinfo->accel_scale)
        }

        line = strtok(NULL, "\n");
    }

    zip_fclose(info_file);
    free(buffer);

    DBGPRINT("Computing # of samples ...")
    // check if the file is the old version as this determines sample calculation
    pinfo->is_old_version = is_old_version(iinfo);

    get_n_samples(pinfo, iinfo);
    // if it is the old version, check file size to determine # of samples
    if (pinfo->is_old_version){
        err = zip_stat(archive, "activity.bin", ZIP_FL_ENC_GUESS, &stats);
        if (((stats.valid & ZIP_STAT_SIZE) != 0) && (err == 0)){
            // bytes->bits, 36 bits per 3axis sample
            int est_n_samples = (int)stats.size * 8 / 36;
            if (est_n_samples > pinfo->samples) pinfo->samples = (int)est_n_samples;
        } else {
            *ierr = E_INFO_STAT;
            return 0;
        }
    }
    DBGPRINT1("Number of samples: %i", pinfo->samples);

    return 1;
}

// -----------------------
// Activity parser helpers
// -----------------------
int bytes2samplesize(const unsigned char type, const unsigned short bytes){
    if (type == RECORDTYPE_ACTIVITY) return (bytes * 2) / 9;
    else if (type == RECORDTYPE_ACTIVITY2) return (bytes / 2) / 3;
    else return 0;
}

double decode_float_param(const uint32_t value){
    double significand, exponent;
    int32_t i32;

    // handle parameters that are too big
    if (value == PARAM_ENCODED_MAXIMUM){
        return DBL_MAX;
    } else if (value == PARAM_ENCODED_MINIMUM){
        return -DBL_MAX;
    }

    // extract exponent
    i32 = (int32_t)((value & PARAM_EXPONENT_MASK) >> PARAM_EXPONENT_OFFSET);
    if ((i32 & 0x80) != 0)
        i32 = (int32_t)((uint32_t)i32 | 0xFFFFFF00u);
    
    exponent = (double)i32;

    // extract significand
    i32 = (int32_t)(value & PARAM_SIGNIFICAND_MASK);
    if ((i32 & PARAM_ENCODED_MINIMUM) != 0)
        i32 = (int32_t)((uint32_t)i32 | 0xFF000000u);
    significand = (double)i32 / PARAM_FLOAT_MAXIMUM;

    // calculate floating point value
    return significand * pow(2.0, exponent);
}

void parse_parameters(zip_file_t *file, GT3XInfo_t *iinfo, const unsigned short bytes, unsigned int *start_time){
    int n_params = bytes / 8;
    unsigned short address = 0, key = 0;
    unsigned int value = 0;

    for (int i = 0; i < n_params; ++i){
        zip_fread(file, &address, 2);
        zip_fread(file, &key, 2);
        zip_fread(file, &value, 4);

        if (address == 0){
            // accel scale is key 55
            if (key == 55){
                if (value != 0){
                    iinfo->accel_scale = decode_float_param(value);
                }
            }
        }
        if (address == 1){
            // start time is key 12
            if (key == 12){
                *start_time = value;  // convert int UNIX time to double
            }
        }
    }
}

void check_window_start_stop(int index[], ParseInfo_t *pinfo, const int sample_size, const int sample_rate, const unsigned int expected_payload_start){
    int curr_i = pinfo->current_sample, local_sample_size = sample_size;

    struct tm ttm;
    time_t t = (long)expected_payload_start;
    ttm = * gmtime(&t);
    int sec_start = ttm.tm_hour * 3600 + ttm.tm_min * 60 + ttm.tm_sec;

    while (local_sample_size > 0){
        if ((sec_start <= pinfo->period) && (pinfo->period < (sec_start + sample_size / sample_rate))){
            index[pinfo->ndi] = -((int)(floor((double)(pinfo->period - sec_start) * (double)sample_rate)) + curr_i);
            if (pinfo->period != pinfo->base) pinfo->ndi += 1;
        }
        if ((sec_start <= pinfo->base) && (pinfo->base < (sec_start + sample_size / sample_rate))){
            index[pinfo->ndi] = (int)(floor((double)(pinfo->base - sec_start) * (double)sample_rate)) + curr_i;
            pinfo->ndi += 1;
        }
        // increment to next day
        curr_i += 24 * 3600 * sample_rate;
        local_sample_size -= 24 * 3600 * sample_rate;
    }
}

void impute(double accel[], double time[], int index[], ParseInfo_t *pinfo, const int sample_size, const int sample_rate, unsigned int expected_payload_start, bool use_zeros){
    int curr_i = pinfo->current_sample;

    check_window_start_stop(index, pinfo, sample_size, sample_rate, expected_payload_start);

    for (int i = 0; i < sample_size; ++i){
        if (use_zeros){
            accel[(curr_i + i) * 3 + 0] = 0.0;  // get previous x acceleration
            accel[(curr_i + i) * 3 + 1] = 0.0;
            accel[(curr_i + i) * 3 + 2] = 0.0;
        } else {
            accel[(curr_i + i) * 3 + 0] = accel[(curr_i + i) * 3 - 3];  // get previous x acceleration
            accel[(curr_i + i) * 3 + 1] = accel[(curr_i + i) * 3 - 2];
            accel[(curr_i + i) * 3 + 2] = accel[(curr_i + i) * 3 - 1];
        }
        time[curr_i + i] = (double)expected_payload_start + (double)i / (double)sample_rate;
    }
}

// ---------------------------
// END Activity parser helpers
// ---------------------------

// ----------------------------
// Activity & Activity2 Parsers
// ----------------------------
void parse_activity_record(zip_file_t *file, double accel[], double time[], int index[], ParseInfo_t *pinfo, GT3XInfo_t *iinfo, const int sample_size, const int payload_start){
    bool odd = false;  // keep track of where the nibble goes
    int current = 0;
    const double digit_mult = pow(10, 3);  // for rounding to 3 decimal places

    check_window_start_stop(index, pinfo, sample_size, iinfo->sample_rate, payload_start);

    for (int i = 0; i < sample_size; ++i){
        for (int j = 0; j < 3; ++j){
            uint16_t shifter;
            // samples are packed in 12 bits, in YXZ order
            if (!odd){
                // can likely be simplified into 1 read
                zip_fread(file, &current, 1);
                shifter = (uint16_t)((current & 0xFF) << 4);
                zip_fread(file, &current, 1);
                shifter |= (uint16_t)((current & 0xF0) >> 4);
            } else {
                shifter = (uint16_t)((current & 0x0F) << 8);
                zip_fread(file, &current, 1);
                shifter |= (uint16_t)(current & 0xFF);
            }

            // sign extension
            if ((shifter & 0x0800) != 0){
                shifter |= 0xF000;
            }

            // convert to signed int (then double), scale, and deal with xyz order
            if (j == 0)
                accel[(pinfo->current_sample + i) * 3 + 1] = round((double)((int16_t)shifter) / iinfo->accel_scale * digit_mult) / digit_mult;
            else if (j == 1)
                accel[(pinfo->current_sample + i) * 3 + 0] = round((double)((int16_t)shifter) / iinfo->accel_scale * digit_mult) / digit_mult;
            else if (j == 2)
                accel[(pinfo->current_sample + i) * 3 + 2] = round((double)((int16_t)shifter) / iinfo->accel_scale * digit_mult) / digit_mult;
        }
        time[pinfo->current_sample + i] = (double)payload_start + (double)i / (double)(iinfo->sample_rate);
    }
}

void parse_activity2_record(zip_file_t *file, double accel[], double time[], int index[], ParseInfo_t *pinfo, GT3XInfo_t *iinfo, const int sample_size, const int payload_start){
    int16_t item;
    const double digit_mult = pow(10, 3);  // for rounding to 3 decimal places

    check_window_start_stop(index, pinfo, sample_size, iinfo->sample_rate, payload_start);

    for (int i = 0; i < sample_size; ++i){
        for (int j = 0; j < 3; ++j){
            zip_fread(file, &item, 2);
            // fprintf(stdout, "\n%f\n", round((double)item / iinfo->accel_scale * digit_mult) / digit_mult);
            accel[(pinfo->current_sample + i) * 3 + j] = round((double)item / iinfo->accel_scale * digit_mult) / digit_mult;
        }
        time[pinfo->current_sample + i] = (double)payload_start + (double)i / (double)(iinfo->sample_rate);
    }
}

/*
Parse the activity (acceleration data) out of a gt3x archive file, in the old format

Parameters
----------
archive : zip_t
    Open GT3X archive file
pinfo : ParseInfo_t
    Struct for storing information relevant during the parsing of the file
iinfo : GT3XInfo_t
    Struct for storing the information from the info file
accel : double
    Pointer to the acceleration array
time : double
    Pointer to the timestamps array
ierr : int
    Indicator for the error

Modifies
--------
accel
time
*/
int parse_activity_old(zip_t *archive, ParseInfo_t *pinfo, GT3XInfo_t *iinfo, double accel[], double time[], double lux[], int index[], int *ierr){
    // open and parse the activity/acceleration data
    zip_file_t *file = zip_fopen(archive, "activity.bin", ZIP_FL_ENC_GUESS);
    if (file == NULL) {
        *ierr = E_OLD_ACTIVITY_OPEN;
        return 0;
    }

    pinfo->current_sample = 0;  // make sure it is set to 0

    parse_activity_record(file, accel, time, index, pinfo, iinfo, pinfo->samples, iinfo->start_time);
    zip_fclose(file);

    // open and parse the lux data
    double lux_scale = 0.0, lux_max = 0.0;
    // get the lux scale factor and max based on serial number
    if (strncmp(iinfo->serial, "MRA", 3) == 0){
        lux_scale = 3.25;
        lux_max = 6000.0;
    } else if (strncmp(iinfo->serial, "NEO", 3) == 0){
        lux_scale = 1.25;
        lux_max = 2500.0;
    }

    file = zip_fopen(archive, "lux.bin", ZIP_FL_ENC_GUESS);
    if (file == NULL) {
        *ierr = E_OLD_LUX_OPEN;
        return 0;
    }

    uint16_t item;
    for (int i = 0; i < pinfo->samples; ++i){
        zip_fread(file, &item, 2);

        if (item < 20) lux[i] = 0.0;
        else if (item >= 65535) lux[i] = 0.0;
        else lux[i] = round(fmin((double)item * lux_scale, lux_max));  // round to nearest int
    }
    zip_fclose(file);
    return 1;
}

/*
Parse the activity (acceleration data) out of a gt3x archive file

Parameters
----------
archive : zip_t
    Open GT3X archive file
pinfo : ParseInfo_t
    Struct for storing information relevant during the parsing of the file
iinfo : GT3XInfo_t
    Struct for storing the information from the info file
accel : double
    Pointer to the acceleration array
time : double
    Pointer to the timestamps array
lux : double
    Pointer to the lux array
ierr : int
    Indicator for the error

Modifies
--------
accel
time
lux
*/
int parse_activity(zip_t *archive, ParseInfo_t *pinfo, GT3XInfo_t *iinfo, double **accel, double **time, double **lux, int **index, int *ierr){
    // if the old version, call the old version parser
    if (pinfo->is_old_version){
        return parse_activity_old(archive, pinfo, iinfo, *accel, *time, *lux, *index, ierr);
    }

    DBGPRINT("\nParsing acitivity file [new version] ...")

    zip_file_t *log_file = zip_fopen(archive, "log.bin", ZIP_FL_ENC_GUESS);
    if (log_file == NULL) {
        *ierr = E_LOG_OPEN;
        return 0;
    }

    int eof = 0;
    unsigned char item = 0, type = 0;
    unsigned int payload_start = 0.0, expected_payload_start = 0.0, payload_timediff = 0.0;
    unsigned short size = 0;
    int sample_size = 0;

    int chksum = 0;
    bool have_activity = false, have_activity2 = false;

    // make sure current location is correctly set to 0
    pinfo->current_sample = 0;
    // make sure current n_days sample is tracked
    pinfo->ndi = 0;

    while (!eof){
        // zip_fread returns # of bytes read, so eof is false by subtracting expected # of bytes
        eof = zip_fread(log_file, &item, 1) - 1;
        if (eof) break;

        if (item == RECORD_SEPARATOR){
            // read the header
            zip_fread(log_file, &type, 1);
            zip_fread(log_file, &payload_start, 4);
            zip_fread(log_file, &size, 2);

            // convert payload size to sample size
            sample_size = bytes2samplesize(type, size);

            if (sample_size > iinfo->sample_rate){
                sample_size = iinfo->sample_rate;
            }
            if ((sample_size + pinfo->current_sample) > pinfo->samples) break;
        } else {
            continue; // if not a record separator, continue until we get one
        }

        // if the payload is the Parameters payload, we need a few items from here (expected start time and accel scale)
        if (type == RECORDTYPE_PARAMETERS){
            parse_parameters(log_file, iinfo, size, &expected_payload_start);
        } else if ((type == RECORDTYPE_ACTIVITY) || (type == RECORDTYPE_ACTIVITY2)){
            // check if the expected payload start matches what the payload actually says
            payload_timediff = payload_start - expected_payload_start;

            // if the start times dont match
            if (payload_timediff > 0.0){
                int n_missing = (int)payload_timediff * iinfo->sample_rate;
                if (n_missing >= 0){
                    // fill with the latest actual value
                    impute(*accel, *time, *index, pinfo, n_missing, iinfo->sample_rate, expected_payload_start, false);
                    pinfo->current_sample += n_missing;  // account for filling
                }
            }

            // if over the allocated size of the arrays
            if ((sample_size + pinfo->current_sample) > pinfo->samples) break;
            // update for next payload
            expected_payload_start = payload_start + 1.0;

            // if no samples in this block, add a second of samples
            if ((sample_size == 0) && (pinfo->current_sample < pinfo->samples)){
                impute(*accel, *time, *index, pinfo, iinfo->sample_rate, iinfo->sample_rate, payload_start, true);
                pinfo->current_sample += iinfo->sample_rate;
            }

            // parse 2 different types of recordings
            if ((type == RECORDTYPE_ACTIVITY) && (sample_size > 0)){
                have_activity = true;
                parse_activity_record(log_file, *accel, *time, *index, pinfo, iinfo, sample_size, payload_start);
                pinfo->current_sample += sample_size;
            } else if ((type == RECORDTYPE_ACTIVITY2) && (sample_size > 0)){
                have_activity2 = true;
                parse_activity2_record(log_file, *accel, *time, *index, pinfo, iinfo, sample_size, payload_start);
                pinfo->current_sample += sample_size;
            }
        } else {
            // skip the current payload if it is a type we aren't concerned with
            zip_fseek(log_file, size, SEEK_CUR);
        }

        // read the checksum, not doing anything with this for now
        eof = zip_fread(log_file, &chksum, 1) - 1;
        // make sure we don't have multiple types of activity blocks
        if (have_activity && have_activity2){
            *ierr = E_LOG_MULTIPLE_ACTIVITY_TYPES;
            return 0;
        }
    }
    DBGPRINT1("# of samples read: %i", pinfo->current_sample);
    return 1;
}