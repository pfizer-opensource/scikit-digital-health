#include "read_bin.h"

void parseline(FILE *fp, char *buff, int buff_len, char **key, char **val)
{
    fgets(buff, buff_len, fp);
    *key = strtok(buff, ":");
    *val = strtok(NULL, ":");
}

int read_header(FILE *fp, Info_t *info)
{
    char buff[255];
    char *k = NULL, *v = NULL;

    /* read the first 19 lines */
    for (int i = 1; i < 20; ++i)
        READLINE;
    
    /* sampling frequency */
    parseline(fp, buff, 255, &k, &v);
    info->fs = (double)strtol(v, NULL, 10);

    /* read another group of lines */
    for (int i = 21; i < 48; ++i)
        READLINE;
    
    /* get the gain and offset values */
    for (int i = 48, j = 0; i < 54; i += 2, ++j)
    {
        parseline(fp, buff, 255, &k, &v);
        info->gain[j] = (double)strtol(v, NULL, 10);
        parseline(fp, buff, 255, &k, &v);
        info->offset[j] = (double)strtol(v, NULL, 10);
    }
    
    /* get the volts and lux values */
    READLINE;
    info->volts = (double)strtol(&buff[6], NULL, 10);
    READLINE;
    info->lux = (double)strtol(&buff[4], NULL, 10);

    /* read and skip a few more lines. Last line read is 58 */
    for (int i = 56; i < 59; ++i)
        READLINE;
    
    info->npages = strtol(&buff[16], NULL, 10);  /* line 58 */

    READLINE;  /* line 59, last line of the header */

    return READ_E_NONE;
}


int get_day_indexing(long *Nps, long *hour, long *min, long *sec, long *msec, Window_t *winfo, Info_t *info, Data_t *data)
{   
    double base_sec=-100.f;  /* seconds since 00:00:00.000 for the base time */
    double period_sec=-100.f;  /* seconds since 00:00:00.000 for the base + period time */
    double dtmp = -100.f, dtmp2 = -100.0f;  /* temp values for storing time difference */
    double _dt = FPAGE_SAMPLES / info->fs;  /* number of seconds in each block */
    double curr = (double)(*hour * SECHOUR + *min * SECMIN + *sec) + (double)*msec / 1000000.0f;
    for (int i = 0; i < winfo->n; ++i)
    {
        /* convert the window into seconds since day start */
        base_sec = (double)(winfo->bases[i] * SECHOUR);
        dtmp = (double)(winfo->periods[i] + winfo->bases[i]);
        period_sec = fmod(dtmp, 24.f) * SECHOUR;
        /* index if needex */
        size_t idx_start = winfo->i_start[i] * winfo->n + i;
        size_t idx_stop = winfo->i_stop[i] * winfo->n + i;

        if (info->max_n == 0)  /* if the recording starts in the middle of the window */
        {
            /*
            Recording start 10:45, windows 0-24, 11-11, 12-6, 8-12, 6-10, 12-11, 11-10

            0-24  yes   10.75 in 0-24  : yes  10.75 in -24-0   : no   || yes
            11-35 yes   10.75 in 11-35 : no   10.75 in -13-11  : yes  || yes
            12-18 no    10.75 in 12-18 : no   10.75 in -12--6  : no   || no
            8-12  yes   10.75 in 8-12  : yes  10.75 in -16--12 : no   || yes
            6-10  no    10.75 in 6-10  : no   10.75 in -18--14 : no   || no
            12-35 yes   10.75 in 12-35 : no   10.75 in -12-11  : yes  || yes
            11-34 no    10.75 in 11-34 : no   10.75 in -13-10  : no   || no
            */
            char in_win1 = (curr > base_sec) && (curr < (dtmp * SECHOUR));
            char in_win2 = (curr > (base_sec - DAYSEC)) && (curr < ((dtmp * SECHOUR) - DAYSEC));
            if (in_win1 || in_win2)
            {
                data->day_starts[idx_start] = 0;
                ++winfo->i_start[i];  /* increment index counter */
            }
        }
        if (*Nps == (info->npages - 1) * PAGE_SAMPLES)  /* if recording ends during a window */
        {
            char in_win1 = (curr > base_sec) && (curr < (dtmp * SECHOUR));
            char in_win2 = (curr > (base_sec - DAYSEC)) && (curr < ((dtmp * SECHOUR) - DAYSEC));
            
            if (in_win1 || in_win2)
            {
                data->day_stops[idx_stop] = info->npages * PAGE_SAMPLES - 1;
                ++winfo->i_stop[i];  /* unnecessary, but w/e ... */
            }
        }

        /* check if the (base + period) is during this block */
        dtmp = period_sec - curr; 
        dtmp2 = dtmp + DAYSEC;  /* deal with time being a cycle */

        if (((dtmp >= 0) && (dtmp < _dt)) || (dtmp2 < _dt))
        {
            data->day_stops[idx_stop] = *Nps + (long)(info->fs * fmin(fabs(dtmp), fabs(dtmp2)));
            ++winfo->i_stop[i];  /* increment index counter */
        }
        /* check if base is during this block */
        dtmp = base_sec - curr;
        dtmp2 = dtmp + DAYSEC;

        if (((dtmp >= 0) && (dtmp < _dt)) || (dtmp2 < _dt))
        {
            data->day_starts[idx_start] = *Nps + (long)(info->fs * fmin(fabs(dtmp), fabs(dtmp2)));
            ++winfo->i_start[i];  /* increment index counter */
        }
    }
    return READ_E_NONE;
}


int get_timestamps(long *Nps, char time[40], Info_t *info, Data_t *data, Window_t *winfo)
{
    struct tm tm0;
    double t0;
    double base_sec;  /* base time converted to seconds */
    double period_sec;  /* period time + base time converted to seconds */
    double curr_sec;  /* block time since start of day */
    double tmp = -1.f, tmp2 = -1.f, _dt = -1.f;  /* for indexing */
    long hour, min, sec, msec;  /* time values needed multiple times */
    long ltmp;

    /* time */
    hour = DATETIME_HOUR(time);
    min = DATETIME_MIN(time);
    sec = DATETIME_SEC(time);
    msec = DATETIME_MSEC(time) * 1000;

    memset(&tm0, 0, sizeof(tm0));
    tm0.tm_year = DATETIME_YEAR(time) - 1900;  /* need years since 1900 */
    tm0.tm_mon  = DATETIME_MONTH(time) - 1;  /* 0 indexed */
    tm0.tm_mday = DATETIME_DAY(time);
    tm0.tm_hour = hour;
    tm0.tm_min  = min;
    tm0.tm_sec  = sec;

    /* convert to seconds since epoch */
    t0 = (double)timegm(&tm0);
    t0 += (double)msec / 1000000.0f;  /* add microseconds */

    /* create the full timestamp array for the block */
    for (int j = 0; j < PAGE_SAMPLES; ++j)
        data->ts[*Nps + j] = t0 + (double)j / info->fs;
    
    /* INDEXING */
    int idx_err = get_day_indexing(Nps, &hour, &min, &sec, &msec, winfo, info, data);

    return READ_E_NONE;
}


int read_block(FILE *fp, Window_t *w_info, Info_t *info, Data_t *data)
{
    char buff[255], data_str[3610], p[4], time[40];
    long N = 0, Nps = 0, t_ = 0;
    double fs, temp;

    /* skip first 2 lines */
    READLINE; READLINE;
    READLINE;  /* 3d line is sequence number */
    N = strtol(&buff[16], NULL, 10);
    Nps = N * PAGE_SAMPLES;
    info->max_n = (N > info->max_n) ? N : info->max_n;  /* max N found so far */

    /* read the line containing the timestamp */
    if (fgets(time, 40, fp) == NULL)
        return READ_E_BLOCK_TIMESTAMP;
    
    /* skip a line then read the line with the temperature */
    READLINE; READLINE;
    temp = strtod(&buff[12], NULL);
    for (int i = Nps; i < (Nps + PAGE_SAMPLES); ++i)
        data->temp[i] = temp;
    
    /* skip 2 more lines then read the sampling rate */
    READLINE; READLINE; READLINE;
    fs = strtod(&buff[22], NULL);
    if ((fs != info->fs) && (info->fs_err < 1)){
        /* set a warning */
        char warn_str[120];
        sprintf(warn_str, "Block (%li) fs [%.2f] is not the same as header fs [%.2f]. Setting fs to block fs.", N, fs, info->fs);
        int err_ret = PyErr_WarnEx(PyExc_RuntimeWarning, warn_str, 1);

        info->fs_err ++;  /* increment the error counter, this error should only happen once */
        /* set the sampling frequency to that of the block */
        info->fs = fs;

        /* if warnings are being caught as exceptions */
        if (err_ret == -1)
            return READ_E_BLOCK_FS;
    } else if ((fs != info->fs) && (info->fs_err >= 1))
        return READ_E_BLOCK_FS;

    /* read the 3600 character data string */
    if (fgets(data_str, 3610, fp) == NULL)
        return READ_E_BLOCK_DATA;
    /* check the length */
    if (strlen(data_str) < 3601)
        return READ_E_BLOCK_DATA_LEN;

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
        data->light[Nps + jj] = (double)(t_ >> 2) * (info->lux / info->volts);
        ++jj;
    }

    get_timestamps(&Nps, time, info, data, w_info);

    return READ_E_NONE;
}
