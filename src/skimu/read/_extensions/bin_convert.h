#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define PAGE_SAMPLES 300
#define FPAGE_SAMPLES 300.0f
#define SECMIN 60
#define SECHOUR 3600

#define READLINE fgets(buff, 255, fp)

#define DATETIME_YEAR(_v)  strtol(&_v[10], NULL, 10)
#define DATETIME_MONTH(_v) strtol(&_v[15], NULL, 10)
#define DATETIME_DAY(_v)   strtol(&_v[18], NULL, 10)
#define DATETIME_HOUR(_v)  strtol(&_v[21], NULL, 10)
#define DATETIME_MIN(_v)   strtol(&_v[24], NULL, 10)
#define DATETIME_SEC(_v)   strtol(&_v[27], NULL, 10)
#define DATETIME_MSEC(_v)  strtol(&_v[30], NULL, 10)

/* BLOCK READING ERRORS */
typedef enum {
    TIMESTAMP_ERROR,
    DATA_ERROR,
    DATA_LEN_ERROR
} Block_Read_Error_t;

// information structs
typedef struct {
  double gain[3];
  double offset[3];
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
  long *idx;
} Data_t;

void read_header(FILE *fp, Info_t *info, int debug);

int read_block(FILE *fp, long *base, long *period, Info_t *info, Data_t *data);
