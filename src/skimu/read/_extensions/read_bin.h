#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define PAGE_SAMPLES 300
#define FPAGE_SAMPLES 300.0f
#define SECMIN 60
#define SECHOUR 3600
#define DAYSEC 86400.f

#define READLINE fgets(buff, 255, fp)

#define DATETIME_YEAR(_v)  strtol(&_v[10], NULL, 10)
#define DATETIME_MONTH(_v) strtol(&_v[15], NULL, 10)
#define DATETIME_DAY(_v)   strtol(&_v[18], NULL, 10)
#define DATETIME_HOUR(_v)  strtol(&_v[21], NULL, 10)
#define DATETIME_MIN(_v)   strtol(&_v[24], NULL, 10)
#define DATETIME_SEC(_v)   strtol(&_v[27], NULL, 10)
#define DATETIME_MSEC(_v)  strtol(&_v[30], NULL, 10)

/* READ ERRORS */
typedef enum {
    READ_E_NONE,  /* no error return value */
    READ_E_BLOCK_TIMESTAMP,  /* issue reading timestamp from block */
    READ_E_BLOCK_FS,  /* block FS does not match header fs */
    READ_E_BLOCK_DATA,  /* error reading block data */
    READ_E_BLOCK_DATA_LEN  /* data is less than 3600 characters */
} Read_Bin_Error_t;

/* Information structures */
typedef struct {
    double fs;  /* sampling frequency */
    double gain[3];  /* raw accel value gain for converting to g */
    double offset[3];  /* raw accel value offset for converting to g */
    double volts;
    double lux;
    long npages;
    long max_n;
} Info_t;

typedef struct {
    double *acc;
    double *light;
    double *temp;
    double *ts;
    long *day_starts;
    long *day_stops;
} Data_t;

typedef struct {
    long n;  /* number of windows */
    long *bases;  /* base hours for windowing */
    long *periods;  /* lengths of windows */
    long *i_start;  /* index for start array */
    long *i_stop;  /* index for end array */
} Window_t;


int read_header(FILE *fp, Info_t *info);
int read_block(FILE *fp, Window_t *w_info, Info_t *info, Data_t *data);
int read_block2(FILE *fp, Window_t *winfo, Info_t *info, Data_t *data);
