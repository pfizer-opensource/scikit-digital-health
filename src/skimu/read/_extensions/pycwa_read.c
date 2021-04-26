#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "cwa_read.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


PyObject * read_cwa(PyObject *NPY_UNUSED(self), PyObject *args){

    PyObject* bytes;
    char* filename;
    char annotation_block[960];
    Py_ssize_t flen;
    FileInfo_t info;
    Window_t winfo;
    PyObject *bases_, *periods_;

    if (!PyArg_ParseTuple(args, "O&ll:read_cwa", PyUnicode_FSConverter, &bytes, &bases, &periods_)) return NULL;
    PyBytes_AsStringAndSize(bytes, &filename, &flen);

    /* GET NUMPY ARRAYS */
    PyArrayObject *bases = (PyArrayObject *)PyArray_FromAny(
        bases_,
        PyArray_DescrFromType(NPY_LONG),
        1,
        0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO,  /* ensure its an array and c-contiguous */
        NULL
    );
    PyArrayObject *periods = (PyArrayObject *)PyArray_FromAny(
        periods_,
        PyArray_DescrFromType(NPY_LONG),
        1,
        0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO,  /* ensure its an array and c-contiguous */
        NULL
    );

    if (!bases || !periods){
        Py_XDECREF(bases);
        Py_XDECREF(periods);
        return NULL;
    }

    /* WINDOWING INFO INIT */
    winfo.n = PyArray_Size(bases);
    if (winfo.n != PyArray_Size(periods))
    {
        Py_XDECREF(bases);
        Py_XDECREF(periods);
        PyErr_SetString(PyExc_ValueError, "Size mismatch between bases and periods");
        return NULL;
    }
    winfo.i_start = (long *)malloc(winfo.n * sizeof(winfo.i_start));
    winfo.i_stop = (long *)malloc(winfo.n * sizeof(winfo.i_stop));
    winfo.bases = (long *)PyArray_DATA(bases);
    winfo.periods = (long *)PyArray_DATA(periods);
    /* set the indices to 0 initially */
    memset(winfo.i_start, 0, winfo.n * sizeof(winfo.i_start))
    memset(winfo.i_stop, 0, winfo.n * sizeof(winfo.i_stop))

    /* OPEN THE FILE */
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
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
    if (!metadata)
        Py_XDECREF(bases);
        Py_XDECREF(periods);
        return NULL;
  
    int ierr = PyDict_SetItem(
        metadata,
        PyUnicode_FromStringAndSize("Annotations", 11),
        PyBytes_FromStringAndSize(annotation_block, 960)
    );
    if (ierr == -1)
    {
        Py_XDECREF(bases);
        Py_XDECREF(periods);
        Py_XDECREF(metadata);
        return NULL;
    }

    npy_intp imu_dims[2]   = {info.count * (info.nblocks-2), info.axes};
    npy_intp ts_dims[1]    = {info.count * (info.nblocks-2)},
    npy_intp block_dims[1] = {info.nblocks-2};
    npy_intp idx_dims[2] = {MAX_DAYS, winfo.n};
  
    PyArrayObject *imudata    = (PyArrayObject *)PyArray_Empty(2, imu_dims, PyArray_DescrFromType(NPY_DOUBLE), 0)
    PyArrayObject *timestamps = (PyArrayObject *)PyArray_Empty(1, ts_dims, PyArray_DescrFromType(NPY_DOUBLE), 0)
    PyArrayObject *light = (PyArrayObject *)PyArray_Empty(1, block_dims, PyArray_DescrFromType(NPY_LONG), 0)

    PyArrayObject *starts = (PyArrayObject *)PyArray_ZEROS(2, idx_dims, NPY_LONG, 0);
    PyArrayObject *stops = (PyArrayObject *)PyArray_ZEROS(2, idx_dims, NPY_LONG, 0);

    if (!imudata || !timestamps || !light || !starts || !stops)
    {
        Py_XDECREF(bases);
        Py_XDECREF(periods);

        Py_XDECREF(imudata);
        Py_XDECREF(timestamps);
        Py_XDECREF(light);
        Py_XDECREF(starts);
        Py_XDECREF(stops);

        free(winfo.i_start);
        free(winfo.i_stop);

        return NULL;
    }

    /* SET POINTERS */
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
    if (ierr != 0)
    {
        Py_XDECREF(bases);
        Py_XDECREF(periods);
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
        "read_cwa",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_cwa_convert(void)
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
