#include "bin_convert.h"

void parseline(FILE *fp, char *buff, int buff_len, char **key, char **value){
    fgets(buff, buff_len, fp);
    *key = strtok(buff, ":");
    *value = strtok(NULL, ":");
}

void read_header(FILE *fp, Info_t *info, int debug){
    char buff[255];
    char *k = NULL, *v = NULL;

    // read the first 47 lines
    for (int i = 1; i < 48; ++i){
        READLINE;
    }

    // get the gain and offset values
    for (int i = 48, j = 0; i < 54; i += 2, ++j){
        parseline(fp, buff, 255, &k, &v);
        info->gain[j] = (double)strtol(v, NULL, 10);
        parseline(fp, buff, 255, &k, &v);
        info->offset[j] = (double)strtol(v, NULL, 10);
    }

    // get the volts and lux values
    READLINE;
    info->volts = (double)strtol(&buff[6], NULL, 10);
    READLINE;
    info->lux = (double)strtol(&buff[4], NULL, 10);

    // read a few more lines
    for (int i = 56; i < 58; ++i){
        READLINE;
    }
    READLINE;  // line 58
    info->npages = strtol(&buff[16], NULL, 10);

    READLINE; // line 59, last line of header

    if (debug){
        fprintf(stdout, "Gain: %f  %f  %f\n", info->gain[0], info->gain[1], info->gain[2]);
        fprintf(stdout, "Offset: %f  %f  %f\n", info->offset[0], info->offset[1], info->offset[2]);
    }
}


int read_block(FILE *fp, long *base, long *period, Info_t *info, Data_t *data){
    char buff[255], data_str[3610], p[4], time[40];
    long N, Nps, t_;
    double fs, base_sec, period_sec, td_;

    // skip first 2 lines
    READLINE; READLINE;
    READLINE;  // 3rd line is the sequence number
    N = strtol(&buff[16], NULL, 10);
    Nps = N * PAGE_SAMPLES;
    info->max_n = N > info->max_n ? N : info->max_n;  // maximum sequence number encountered so far

    // read the line containing the timestamp
    if (!fgets(time, 40, fp)) return TIMESTAMP_ERROR;

    // skip a line then read the line with the temperature
    READLINE; READLINE;
    td_ = strtod(&buff[12], NULL);
    for (int i = Nps; i < (Nps + PAGE_SAMPLES); ++i){
        data->temp[i] = td_;
    }
    // skip 2 more lines then read the sampling rate
    READLINE; READLINE; READLINE;
    fs = strtod(&buff[22], NULL);

    // read the data
    if (!fgets(data_str, 3610, fp)) return DATA_ERROR;
    // check length
    if (strlen(data_str) < 3601) return DATA_LEN_ERROR;

    // put the data block into the appropriate data stores
    int k = 0, j = 0, jj = 0;
    for (int i = 0; i < 3600; i += 3){
        memcpy(p, &data_str[i], 3);
        t_ = strtol(p, NULL, 16);
        if (k < 3){
            t_ = t_ > 2047 ? -4096 + t_ : t_;  // sign conversion
            data->acc[Nps * 3 + j] = ((double)t_ * 100.0f - info->offset[k]) / info->gain[k];
            ++j; ++k;
        } else {
            data->light[Nps + jj] = (double)(t_ >> 2) * (info->lux / info->volts);
            ++jj;
            k = 0;  // reset the cycle
        }
    }

    struct tm tm0;
    double t0;
    long hour, min, sec, msec;  // stuff needed later

    // time
    hour = DATETIME_HOUR(time);
    min = DATETIME_MIN(time);
    sec = DATETIME_SEC(time);
    msec = DATETIME_MSEC(time) * 1000;

    memset(&tm0, 0, sizeof(tm0));
    tm0.tm_year = DATETIME_YEAR(time) - 1900;  // need years since 1990
    tm0.tm_mon  = DATETIME_MONTH(time) - 1;  // 0 indexed
    tm0.tm_mday = DATETIME_DAY(time);
    tm0.tm_hour = hour;
    tm0.tm_min  = min;
    tm0.tm_sec  = sec;

    // convert to seconds since epoch
    t0 = (double)timegm(&tm0);
    t0 += (double)msec / 1000000.0f;  // add microseconds

    // create the full timestamp array for the block
    for (int j = 0; j < PAGE_SAMPLES; ++j){
        data->ts[Nps + j] = t0 + (double)j / fs;
    }

    // INDEXING stuff
    // convert base and period to seconds since 00:00:00
    base_sec = (double)(*base * SECHOUR);
    period_sec = (double)(((*period + *base) % 24) * SECHOUR);
    // get the time in seconds since day start
    double hours_sec = (double)(
        (hour * SECHOUR)
        + (min * SECMIN)
        + sec
    ) + ((double)msec) / 1000000.0f;

    double tmp = period_sec - hours_sec;
    double tmp2 = tmp + 86400.0;  // deal with time being a cycle
    double _dt = FPAGE_SAMPLES / fs;

    if (((tmp >= 0) && (tmp < _dt)) || (tmp2 < _dt)){
        data->idx[N] = -(Nps + (long)(fs * fmin(fabs(tmp), fabs(tmp2))));
    }

    tmp = base_sec - hours_sec;
    tmp2 = tmp + 86400.0;
    if (((tmp >= 0) && (tmp < _dt)) || (tmp2 < _dt)){
        data->idx[N] = Nps + (long)(fs * fmin(fabs(tmp), fabs(tmp2)));
    }
    return -1;
}