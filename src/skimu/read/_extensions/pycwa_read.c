#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

extern void fread_cwa(long *, char[], FileInfo_t *, double *, double *, long *, long *);


PyObject * read_cwa(PyObject *NPY_UNUSED(self), PyObject *args){

  PyObject* bytes; 
  char* filename;
  char annotation_block[960];
  Py_ssize_t flen;
  FileInfo_t info;

  if (!PyArg_ParseTuple(args, "O&ll:read_cwa", PyUnicode_FSConverter, &bytes, &(info.base), &(info.period))) return NULL;
  PyBytes_AsStringAndSize(bytes, &filename, &flen);

  FILE *fp = fopen(filename, "rb");
  if (!fp){
    PyErr_SetString(PyExc_IOError, "Error openining file");
    Py_XDECREF(bytes);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  unsigned long length_bytes = ftell(fp);
  info.nblocks = length_bytes / 512;

  fseek(fp, 64, SEEK_SET);  // read the annotation block
  fread(annotation_block, 1, 960, fp);  // current position = 1024
  fseek(fp, 25, SEEK_CUR);  // current position = 1024 + 25
  fread(&(info.axes), 1, 1, fp);  // current position = 1024+25+1
  fseek(fp, 2, SEEK_CUR);  // current position = 1024 + 28 = 1024+25+1+2
  fread(&(info.count), 2, 1, fp);

  info.axes = (info.axes >> 4) & 0x0f;

  fclose(fp);

  PyObject *metadata = PyDict_New();
  if (!metadata) return NULL;
  
  int ierr = PyDict_SetItem(
    metadata, 
    PyUnicode_FromStringAndSize("Annotations", 11), 
    PyBytes_FromStringAndSize(annotation_block, 960)
  );
  if (ierr == -1){
    Py_XDECREF(metadata);
    return NULL;
  }

  npy_intp imu_dims[2]   = {info.count * (info.nblocks-2), info.axes},
           ts_dims[1]    = {info.count * (info.nblocks-2)},
           block_dims[1] = {info.nblocks-2};
  
  PyArrayObject *imudata    = (PyArrayObject *)PyArray_Empty(2, imu_dims, PyArray_DescrFromType(NPY_DOUBLE), 0),
                *timestamps = (PyArrayObject *)PyArray_Empty(1, ts_dims, PyArray_DescrFromType(NPY_DOUBLE), 0),
                *light = (PyArrayObject *)PyArray_Empty(1, block_dims, PyArray_DescrFromType(NPY_LONG), 0),
                *index = (PyArrayObject *)PyArray_Empty(1, block_dims, PyArray_DescrFromType(NPY_LONG), 0);

  double *data_ptr = (double *)PyArray_DATA(imudata);
  double *ts_ptr = (double *)PyArray_DATA(timestamps);
  long *lgt_ptr = (long *)PyArray_DATA(light);
  long *idx_ptr = (long *)PyArray_DATA(index);

  fread_cwa(&flen, filename, &info, data_ptr, ts_ptr, idx_ptr, lgt_ptr);

  ierr = PyDict_SetItem(
    metadata, 
    PyUnicode_FromStringAndSize("Frequency", 9), 
    PyFloat_FromDouble(info.frequency)
  );
  ierr += PyDict_SetItem(
    metadata, 
    PyUnicode_FromStringAndSize("Device ID", 9), 
    PyLong_FromLong(info.deviceID)
  );
  ierr += PyDict_SetItem(
    metadata, 
    PyUnicode_FromStringAndSize("Session ID", 10), 
    PyLong_FromLong(info.sessionID)
  );
  if (ierr != 0){
    Py_XDECREF(metadata);
    Py_XDECREF(imudata);
    Py_XDECREF(timestamps);
    Py_XDECREF(light);
    Py_XDECREF(index);
    return NULL;
  }

  return Py_BuildValue(
    "NNNNN",  /* N doesnt increase reference counter */
    metadata,
    (PyObject *)imudata,
    (PyObject *)timestamps,
    (PyObject *)index,
    (PyObject *)light
  );
}

static struct PyMethodDef methods[] = {
    {"read_cwa",   read_cwa,   1, NULL},  // last is test__doc__
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cwa_read",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_cwa_read(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    /* Import the array object */
    import_array();

    /* XXXX Add constants here */

    return m;
}