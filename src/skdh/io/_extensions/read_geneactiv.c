// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#include "read_binary_imu.h"

void parseline(FILE *fp, char *buff, int buff_len, char **key, char **val)
{
    fgets(buff, buff_len, fp);
    *key = strtok(buff, ":");
    *val = strtok(NULL, ":");
}

int geneactiv_read_header(FILE *fp, GN_Info_t *info)
{
    char buff[255];
    char *k = NULL, *v = NULL;

    /* read the first 19 lines */
    DEBUG_PRINTF("reading first 19 lines\n");
    for (int i = 1; i < 20; ++i)
        GN_READLINE;
    
    /* sampling frequency */
    DEBUG_PRINTF("getting sampling frequency\n");
    parseline(fp, buff, 255, &k, &v);
    info->fs = (double)strtol(v, NULL, 10);

    /* read another group of lines */
    DEBUG_PRINTF("reading next batch of lines\n");
    /* this should read from line 21 to 48, but will also handle
       cases where a subject comment spans multiple lines
     */
    while (strncmp(buff, "Calibration Data", 16) != 0)
    {
        GN_READLINE;
    }
    
    /* get the gain and offset values */
    DEBUG_PRINTF("getting gain and offset\n");
    for (int i = 48, j = 0; i < 54; i += 2, ++j)
    {
        parseline(fp, buff, 255, &k, &v);
        info->gain[j] = (double)strtol(v, NULL, 10);
        parseline(fp, buff, 255, &k, &v);
        info->offset[j] = (double)strtol(v, NULL, 10);
    }
    
    /* get the volts and lux values */
    DEBUG_PRINTF("getting volts and lux values\n");
    GN_READLINE;
    info->volts = (double)strtol(&buff[6], NULL, 10);
    GN_READLINE;
    info->lux = (double)strtol(&buff[4], NULL, 10);

    /* read and skip a few more lines. Last line read is 58 */
    for (int i = 56; i < 59; ++i)
        GN_READLINE;

    DEBUG_PRINTF("getting number of pages\n");
    info->npages = strtol(&buff[16], NULL, 10);  /* line 58 */

    GN_READLINE;  /* line 59, last line of the header */

    DEBUG_PRINTF("finished with reading the header\n");
    return GN_READ_E_NONE;
}


int get_timestamps(long *Nps, char time[40], GN_Info_t *info, GN_Data_t *data)
{
    struct tm tm0;
    double t0;
    Time_t t;

    /* time */
    t.hour = GN_DATE_HOUR(time);
    t.min = GN_DATE_MIN(time);
    t.sec = GN_DATE_SEC(time);
    t.msec = GN_DATE_MSEC(time);

    memset(&tm0, 0, sizeof(tm0));
    tm0.tm_year = GN_DATE_YEAR(time) - 1900;  /* need years since 1900 */
    tm0.tm_mon  = GN_DATE_MONTH(time) - 1;  /* 0 indexed */
    tm0.tm_mday = GN_DATE_DAY(time);
    tm0.tm_hour = t.hour;
    tm0.tm_min  = t.min;
    tm0.tm_sec  = t.sec;

    /* convert to seconds since epoch */
    t0 = (double)timegm(&tm0);
    t0 += (double)t.msec / 1000.0f;  /* add microseconds */

    /* create the full timestamp array for the block */
    for (int j = 0; j < GN_SAMPLES; ++j)
        data->ts[*Nps + j] = t0 + (double)j / info->fs;
    
    /* INDEXING */
    long mdays = MAX_DAYS;
    long gns = GN_SAMPLES;
    double block_t_delta = GN_SAMPLESf / info->fs;

    return GN_READ_E_NONE;
}


int geneactiv_read_block(FILE *fp, GN_Info_t *info, GN_Data_t *data)
{
    char buff[255], data_str[3610], p[4], time[40];
    long N = 0, Nps = 0, t_ = 0;
    double fs, temp;
    int ier = GN_READ_E_NONE;

    /* read/skip first 2 lines */
    if (GN_READLINE == NULL)  /* make sure that the first line is actually a "Recorded Data" block */
        return GN_READ_E_BLOCK_MISSING_BLOCK_WARN;
    GN_READLINE;
    GN_READLINE;  /* 3d line is sequence number */
    N = strtol(&buff[16], NULL, 10);
    Nps = N * GN_SAMPLES;
    info->max_n = (N > info->max_n) ? N : info->max_n;  /* max N found so far */

    /* read the line containing the timestamp */
    if (fgets(time, 40, fp) == NULL)
        return GN_READ_E_BLOCK_TIMESTAMP;
    
    /* skip a line then read the line with the temperature */
    GN_READLINE; GN_READLINE;
    temp = strtod(&buff[12], NULL);
    for (int i = Nps; i < (Nps + GN_SAMPLES); ++i)
        data->temp[i] = temp;
    
    /* skip 2 more lines then read the sampling rate */
    GN_READLINE; GN_READLINE; GN_READLINE;
    fs = strtod(&buff[22], NULL);
    if ((fs != info->fs) && (info->fs_err < 1)){
        info->fs_err ++;  /* increment the error counter, this error should only happen once */
        /* set the sampling frequency to that of the block */
        info->fs = fs;

        ier = GN_READ_E_BLOCK_FS_WARN;  /* set so that the warning message can be printed after function */
    } else if ((fs != info->fs) && (info->fs_err >= 1))
        return GN_READ_E_BLOCK_FS;

    /* read the 3600 character data string */
    if (fgets(data_str, 3610, fp) == NULL)
        return GN_READ_E_BLOCK_DATA;
    /* check the length */
    if (strlen(data_str) < 3601)
        return GN_READ_E_BLOCK_DATA_3600;

    /* put the block data into the appropiate location */
    int j = 0, jj = 0;
    for (int i = 0; i < 3600; i += 12)
    {
        for (int k = 0; k < 3; ++k)  /* first 3 values are accel x, y, z */
        {
            memcpy(p, &data_str[i + k * 3], 3);
            t_ = strtol(p, NULL, 16);
            t_ = (t_ > 2047) ? -4096 + t_ : t_;
            data->acc[Nps * 3 + j] = ((double)t_ * 100.0f - info->offset[k]) / info->gain[k];
            ++j;
        }
        memcpy(p, &data_str[i + 9], 3);  /* last value is light */
        t_ = strtol(p, NULL, 16);
        data->light[Nps + jj] = floor((double)(t_ >> 2) * (info->lux / info->volts));
        ++jj;
    }

    get_timestamps(&Nps, time, info, data);

    return ier;
}
