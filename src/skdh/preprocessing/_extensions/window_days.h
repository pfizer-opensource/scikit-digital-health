#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifdef _WIN32
#define timegm _mkgmtime
#endif

long finalize_guess(size_t n, double *timestamps, long *guess, double target);
void window(size_t n, double *timestamps, double fs, long *base, long *period, long *windows);