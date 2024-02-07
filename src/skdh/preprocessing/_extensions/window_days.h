#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

long finalize_guess(size_t n, double *timestamps, long *guess, double target);
void window(size_t n, double *timestamps, double fs, long *base, long *period, long *windows);