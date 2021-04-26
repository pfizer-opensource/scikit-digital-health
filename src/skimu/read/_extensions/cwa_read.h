typedef struct {
    long base;
    long period;
    long deviceID;
    long sessionID;
    int nblocks;
    int8_t axes;
    int16_t count;
    double tLast;
    int N;
    double frequency;
} FileInfo_t;

typedef struct {
    long n;  /* number of windows */
    long *bases;  /* base hours for windowing */
    long *periods;  /* lengths of windows */
    long *i_start;  /* index for start array */
    long *i_stop;  /* index for stop array */
} Window_t;

#define MAX_DAYS 25 /* upper limit on number of days device can record */

extern void fread_cwa(long *, char[], FileInfo_t *, double *, double *, long *, long *);