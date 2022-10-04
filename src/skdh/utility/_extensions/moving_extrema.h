#ifndef MOVING_EXTREMA_H_  // guard
#define MOVING_EXTREMA_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// moving extrema functions for 1d arrays
void moving_max_c(long *n, double x[], long *wlen, long *skip, double res[]);
void moving_min_c(long *n, double x[], long *wlen, long *skip, double res[]);

#endif  // MOVING_EXTREMA_H_
