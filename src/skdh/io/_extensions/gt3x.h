// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <zip.h>
#include <time.h>

// -----------------
// ERROR DEFINES
// -----------------
enum Errors_t{
    // info file errors
    E_INFO_STAT  = 1,
    E_INFO_OPEN  = 2,
    // log file errors
    E_LOG_OPEN = 3,
    E_LOG_MULTIPLE_ACTIVITY_TYPES = 4,
    // old file errors
    E_OLD_ACTIVITY_OPEN = 5,
    E_OLD_LUX_OPEN = 6,
    // general errors
    E_MALLOC = 9
};

// -----------------
// END ERROR DEFINES
// -----------------

#define DBGPRINT(a) if (pinfo->debug) fprintf(stdout, a "\n");
#define DBGPRINT1(a, b) if (pinfo->debug) fprintf(stdout, a "\n", b);

typedef struct {
    bool debug;
    bool is_old_version;  // if the file is using the old verison 
    int samples;  // number of samples in the file
    int n_days;  // keep track of number of days
    int ndi;  // n_days index tracker
    int current_sample;  // track the current sample in the arrays
    int open_err;  // error saving for the zip archive
    long base;  // base hour for windowing
    long period;  // # of hours per window
} ParseInfo_t;

typedef struct {
    int major;
    int minor;
    int build;
} Version_t;

typedef struct {
    char serial[14];
    int sample_rate;
    double start_time;
    double stop_time;
    double last_sample_time;
    double download_time;
    double accel_scale;
    Version_t firmware;
} GT3XInfo_t;

// --------------------
// Function definitions
// --------------------
int parse_info(zip_t *archive, ParseInfo_t *pinfo, GT3XInfo_t *iinfo, int *ierr);

int parse_activity(zip_t *archive, ParseInfo_t *pinfo, GT3XInfo_t *iinfo, double **accel, double **time, double **lux, int **index, int *ierr);
