#include "window_days.h"


long finalize_guess(size_t n, double *timestamps, long *guess, double target)
{
    // create safe indexes for the sides of the guess
    long i1 = (long)fmax((double)*guess - 1.0, 0.0);
    long i2 = *guess;
    long i3 = (long)fmin((double)*guess + 1.0, (double)n - 1.0);

    // check for values out of bounds
    if (i2 <= 0) return 0;
    if (i2 >= (n - 1)) return (long)n - 1;

    bool check1 = fabs(timestamps[i2] - target) <= fabs(timestamps[i1] - target);
    bool check3 = fabs(timestamps[i2] - target) <= fabs(timestamps[i3] - target);

    // path 1: guess is smallest value
    if (check1 && check3)
    {
        return i2;
    }
    // path 2: smaller value to the left side
    else if (!check1)
    {
        *guess -= 1;
    }
    // path 3: smaller value to the right side
    else if (!check3)
    {
        *guess += 1;
    }

    // only got here for path 2 & 3 - continue with updated guess value
    return finalize_guess(n, timestamps, guess, target);
}


void window(size_t n, double *timestamps, double fs, long *base, long *period, long *windows)
{
    // convert the first timestamp
    time_t epoch = (time_t)timestamps[0];
    struct tm ts = *gmtime(&epoch);

    // one day timedelta
    time_t day_delta_s = 86400;

    // number of samples per day
    size_t n_perday = (size_t)round((double)day_delta_s * fs);

    // find the end time for the windows
    long period2 = (*base + *period) % 24;

    // create the first day start, and back off by a day to make sure that we get the
    // actual start
    struct tm tm_base = ts;
    struct tm tm_period = ts;

    tm_base.tm_hour = *base;
    tm_base.tm_min = 0;
    tm_base.tm_sec = 0;

    tm_period.tm_hour = period2;
    tm_period.tm_min = 0;
    tm_period.tm_sec = 0;

    time_t ts_base = timegm(&tm_base);
    time_t ts_period = timegm(&tm_period);

    // subtract a day to make sure we start at the beginning of the timestamp array
    ts_base -= day_delta_s;
    ts_period -= day_delta_s;

    // check that the period occurs after the base
    if (ts_period <= ts_base) ts_period += day_delta_s;

    // make sure that at least one of the timestamps is during the recording
    while (ts_period < timestamps[0])
    {
        ts_base += day_delta_s;
        ts_period += day_delta_s;
    }

    // create a first guess for the indices
    long guess_base = (long)(((double)ts_base - timestamps[0]) * fs);
    long guess_period = (long)(((double)ts_period - timestamps[0]) * fs);

    size_t i = 0;
    while (ts_base < timestamps[n - 1])
    {
        windows[i] = finalize_guess(n, timestamps, &guess_base, ts_base);
        windows[i + 1] = finalize_guess(n, timestamps, &guess_period, ts_period);

        // increment by a day
        guess_base += n_perday;
        guess_period += n_perday;

        ts_base += day_delta_s;
        ts_period += day_delta_s;

        // increment counter - array is 2d;
        i += 2;
    }
}