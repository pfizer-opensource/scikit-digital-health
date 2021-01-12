#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PAGE_SAMPLES 300
#define FPAGE_SAMPLES 300.0f
#define SECMIN 60
#define SECHOUR 3600

#define READLINE fgets(buff, 255, fp)

#define STR2PY PyUnicode_FromString


#define DATETIME_YEAR(_v)  strtol(&_v[10], NULL, 10)
#define DATETIME_MONTH(_v) strtol(&_v[15], NULL, 10)
#define DATETIME_DAY(_v)   strtol(&_v[18], NULL, 10)
#define DATETIME_HOUR(_v)  strtol(&_v[21], NULL, 10)
#define DATETIME_MIN(_v)   strtol(&_v[24], NULL, 10)
#define DATETIME_SEC(_v)   strtol(&_v[27], NULL, 10)
#define DATETIME_MSEC(_v)  strtol(&_v[30], NULL, 10)


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
  long *dt;
} Data_t;

void parseline(FILE *fp, char *buff, int buff_len, char **key, char **value){
  fgets(buff, buff_len, fp);
  *key = strtok(buff, ":"); *value = strtok(NULL, ":");
}

void read_header(FILE *fp, Info_t *info, PyObject *hdr_dict){
  char buff[255];
  char *k = NULL, *v = NULL;

  // read the first 47 lines
  for (int i=0; i < 47; i++){
    parseline(fp, buff, 255, &k, &v);
    // if the line was split into 2 pieces
    if (v){
      v[strcspn(v, "\r\n")] = 0;
      PyDict_SetItem(hdr_dict, STR2PY(k), STR2PY(v));
    }
  }

  // get the gain and offset values, but don't really need to store these values for python
  for (int i=47, j=0; i < 53; i+=2, j++){
    parseline(fp, buff, 255, &k, &v);
    info->gain[j] = (double)strtol(v, NULL, 10);
    parseline(fp, buff, 255, &k, &v);
    info->offset[j] = (double)strtol(v, NULL, 10);
  }

  // get the volts and lux values
  READLINE;  // line 53
  info->volts = (double)strtol(&buff[6], NULL, 10);
  READLINE;  // line 54
  info->lux = (double)strtol(&buff[4], NULL, 10);

  // read a few more lines
  for (int i=55; i < 57; i++){
    parseline(fp, buff, 255, &k, &v);
    if (v){
      v[strcspn(v, "\r\n")] = 0;
      PyDict_SetItem(hdr_dict, STR2PY(k), STR2PY(v));
    }
  }
  READLINE;  // line 57
  info->npages = strtol(&buff[16], NULL, 10);

  READLINE;  // line 58, last line of header
}


int read_block(FILE *fp, double *sec_base, double *sec_period, Info_t *info, Data_t *data){
  char buff[255], data_str[3610], p[4], time[40];
  long N, Nps, t_;
  double fs;

  // skip first 2 lines
  READLINE; READLINE;
  READLINE;
  N = strtol(&buff[16], NULL, 10);  // sequence number
  Nps = N * PAGE_SAMPLES;
  info->max_n = N > info->max_n ? N : info->max_n;

  // read the line containing the timestamp
  if (!fgets(time, 40, fp)) return 0;
  // skip a line then read the line with the temperature
  READLINE; READLINE;
  data->temp[N] = strtod(&buff[12], NULL);
  // skip 2 more lines then read the sampling rate
  READLINE; READLINE; READLINE;
  fs = strtod(&buff[22], NULL);

  // read the data
  if (!fgets(data_str, 3610, fp)) return 0;
  // check length
  if (strlen(data_str) < (size_t)3601) return 0;

  int k=0, j=0, jj = 0;
  for (int i = 0; i < 3600; i += 3){
    memcpy(p, &data_str[i], 3);
    t_ = strtol(p, NULL, 16);
    if (k < 3){
      t_ = t_ > 2047 ? -4096 + t_ : t_;
      data->acc[Nps * 3 + j] = ((double)t_ * 100.0f - info->offset[k]) / info->gain[k];
      j++; k++;
    } else {
      data->light[Nps + jj] = (double)(t_ >> 2) * (info->lux / info->volts);
      jj++;
      k = 0;
    }
  }

  struct tm tm0;
  double t0;

  data->dt[N * 7 + 0] = DATETIME_YEAR(time);
  data->dt[N * 7 + 1] = DATETIME_MONTH(time);
  data->dt[N * 7 + 2] = DATETIME_DAY(time);
  data->dt[N * 7 + 3] = DATETIME_HOUR(time);
  data->dt[N * 7 + 4] = DATETIME_MIN(time);
  data->dt[N * 7 + 5] = DATETIME_SEC(time);
  data->dt[N * 7 + 6] = DATETIME_MSEC(time) * 1000;

  // time
  memset(&tm0, 0, sizeof(tm0));
  tm0.tm_year = data->dt[N * 7 + 0] - 1900;  // need years since 1990
  tm0.tm_mon  = data->dt[N * 7 + 1] - 1;  // 0 indexed
  tm0.tm_mday = data->dt[N * 7 + 2];
  tm0.tm_hour = data->dt[N * 7 + 3];
  tm0.tm_min  = data->dt[N * 7 + 4];
  tm0.tm_sec  = data->dt[N * 7 + 5];
  
  // convert to seconds since epoch
  t0 = (double)timegm(&tm0);
  t0 += ((double)data->dt[N * 7 + 6]) / 1000000.0f;  // add microseconds

  // create the timestamps
  for (int j = 0; j < PAGE_SAMPLES; j++){
    data->ts[Nps + j] = t0 + ((double)j) / fs;
  }

  // get the time in seconds since day start
  double sec_hours = (double)((data->dt[N * 7 + 3] * SECHOUR)
                     + (data->dt[N * 7 + 4] * SECMIN)
                     + data->dt[N * 7 + 5])
                     + ((double)data->dt[N * 7 + 6]) / 1000000.0f;
  
  double tmp = *sec_period - sec_hours;
  double tmp2 = tmp + 86400.0;
  double _dt = FPAGE_SAMPLES / fs;

  if (((tmp >= 0) && (tmp < _dt)) || (tmp2 < _dt)){
    data->idx[N] = -(Nps + (long)(fs * fmin(fabs(tmp), fabs(tmp2))));
  }
  tmp = *sec_base - sec_hours;
  tmp2 = tmp + 86400.0;
  if (((tmp >= 0) && (tmp < _dt)) || (tmp2 < _dt)){
    data->idx[N] = Nps + (long)(fs * fmin(fabs(tmp), fabs(tmp2)));
  }
  return 1;
}




static PyObject * bin_convert(PyObject *NPY_UNUSED(self), PyObject *args){
  char *file;
  int ierr, fail;
  long base, period;
  double sec_base, sec_period;
  FILE *fp;
  Info_t info;
  Data_t data;

  info.max_n = 0;  // initialize

  if (!PyArg_ParseTuple(args, "sll:bin_convert", &file, &base, &period)){
    return NULL;
  }

  // compute the base and period in seconds since day start
  sec_base = (double)(base * SECHOUR);
  sec_period = (double)(((period + base) % 24) * SECHOUR);

  fp = fopen(file, "r");
  if (!fp){
    PyErr_SetString(PyExc_IOError, "Error openining file");
    return NULL;
  }

  PyObject *hdict = PyDict_New();

  read_header(fp, &info, hdict);

  // dimension stuff for return values
  npy_intp dim3[2] = {info.npages * PAGE_SAMPLES, 3},
           dim[1]  = {info.npages * PAGE_SAMPLES},
           dim_np[1] = {info.npages},
           dim_dt[2] = {info.npages, 7};
  // create the data arrays
  PyArrayObject *accel = (PyArrayObject *)PyArray_Empty(2, dim3, PyArray_DescrFromType(NPY_DOUBLE), 0),
                *time  = (PyArrayObject *)PyArray_Empty(1, dim, PyArray_DescrFromType(NPY_DOUBLE), 0),
                *light = (PyArrayObject *)PyArray_Empty(1, dim, PyArray_DescrFromType(NPY_DOUBLE), 0),
                *temperature = (PyArrayObject *)PyArray_Empty(1, dim_np, PyArray_DescrFromType(NPY_DOUBLE), 0),
                *index = (PyArrayObject *)PyArray_Empty(1, dim_np, PyArray_DescrFromType(NPY_LONG), 0),
                *datetime = (PyArrayObject *)PyArray_Empty(2, dim_dt, PyArray_DescrFromType(NPY_LONG), 0);

  data.acc   = (double *)PyArray_DATA(accel);
  data.ts    = (double *)PyArray_DATA(time);
  data.light = (double *)PyArray_DATA(light);
  data.temp  = (double *)PyArray_DATA(temperature);
  data.idx   = (long *)PyArray_DATA(index);
  data.dt    = (long *)PyArray_DATA(datetime);

  // initialize index values
  for (int i = 0; i < info.npages; i++){
    data.idx[i] = -2 * PAGE_SAMPLES * info.npages;
  }

  fail = 0;
  // read the file
  for (int i = 0; i < info.npages; i++){
    ierr = read_block(fp, &sec_base, &sec_period, &info, &data);

    if (!ierr){
      fail = 1;
      break;
    }
  }

  // close the file
  fclose(fp);

  if (fail){
    Py_XDECREF(hdict);
    Py_XDECREF(accel);
    Py_XDECREF(time);
    Py_XDECREF(temperature);
    Py_XDECREF(index);
    
    PyErr_SetString(PyExc_RuntimeError, "Error reading GeneActiv .bin file");
    return NULL;
  }

  return Py_BuildValue(
    "lOOOOOOO",
    info.max_n,
    hdict, 
    (PyObject *)accel, 
    (PyObject *)time, 
    (PyObject *)light, 
    (PyObject *)temperature, 
    (PyObject *)index,
    (PyObject *)datetime
  );
  
}


static const char bin_convert__doc__[] = "bin_convert(file, base, period)\n"
"Read a Geneactiv File\n\n"
"Parameters\n"
"----------\n"
"file : str\n"
"  File name to read from\n"
"base : int\n"
"  Base time for providing windowing. Must be in [0, 23]\n"
"period : int\n"
"  number of hours in each window. Must be in [1, 24]\n";


static struct PyMethodDef methods[] = {
  {"bin_convert", bin_convert, 1, bin_convert__doc__},
  {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "bin_convert",
  NULL,
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};


/* Initialization function for the module */
PyMODINIT_FUNC PyInit_bin_convert(void){
  PyObject *m;
  m = PyModule_Create(&moduledef);
  if (m == NULL){
    return NULL;
  }

  /* import the array object */
  import_array();

  /* add constants here */

  return m;
}