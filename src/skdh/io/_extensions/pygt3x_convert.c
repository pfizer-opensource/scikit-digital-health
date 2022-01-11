// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include "gt3x.h"

// stdio, string + others are included in Python.h

PyObject * read_gt3x(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject* bytes; 
    char* filename;
    Py_ssize_t flen;

    ParseInfo_t info;  // for storing info used during the file parsing
    GT3XInfo_t file_info;  // for storing information from the "info.txt" file

    // explicitly set debug
    info.debug = 0;

    if (!PyArg_ParseTuple(args, "O&ll:read_cwa", PyUnicode_FSConverter, &bytes, &(info.base), &(info.period))) return NULL;
    PyBytes_AsStringAndSize(bytes, &filename, &flen);

    // convert base and period to seconds
    info.period = ((info.base + info.period) % 24) * 3600;
    info.base *= 3600;

    int archive_err = 0;
    zip_t *gt3x = zip_open(filename, ZIP_RDONLY, &archive_err);
    if (!gt3x){
        PyErr_SetString(PyExc_IOError, "Error opening file");
        return (PyObject *)NULL;
    }

    int read_err = 0;
    if (!parse_info(gt3x, &info, &file_info, &read_err)){
        char buffer[42];
        snprintf(buffer, sizeof buffer, "[%i] Error parsing info.txt file", read_err);
        PyErr_SetString(PyExc_RuntimeError, buffer);
        return (PyObject *)NULL;
    }

    // create dimensions
    npy_intp acc_dims[2] = {info.samples, 3},
             dims[1] = {info.samples},
             idx_dims[1] = {info.n_days * 2 + 4};  // start and stop index per day, plus safety margin

    // allocate numpy arrays and pointers to the data
    PyArrayObject *accel = (PyArrayObject *)PyArray_EMPTY(2, acc_dims, NPY_DOUBLE, 0),
                  *time  = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_DOUBLE, 0),
                  *lux   = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0),
                  *index = (PyArrayObject *)PyArray_ZEROS(1, idx_dims, NPY_INT, 0);

    double *accel_ptr = (double *)PyArray_DATA(accel),
           *time_ptr  = (double *)PyArray_DATA(time),
           *lux_ptr   = (double *)PyArray_DATA(lux);
    int *index_ptr = (int *)PyArray_DATA(index);

    // set the index values to outside possible range
    for (int i = 0; i < info.n_days * 2 + 4; ++i){
        index_ptr[i] = - 2 * info.samples;
    }

    // read the data into the arrays
    if (!parse_activity(gt3x, &info, &file_info, &accel_ptr, &time_ptr, &lux_ptr, &index_ptr, &read_err)){
        char buffer[55];
        snprintf(buffer, sizeof buffer, "[%i] Error parsing time-series data", read_err);
        PyErr_SetString(PyExc_RuntimeError, buffer);
        return (PyObject *)NULL;
    }

    zip_close(gt3x);

    return Py_BuildValue(
        "NNNNN",  /* N doesnt increase ref counter */
        (PyObject *)time,
        (PyObject *)accel,
        (PyObject *)lux,
        (PyObject *)index,
        PyLong_FromLong(info.current_sample)
    );
}

const static char docstring[] = "parse_gt3x(filename)\n\nParse an Actigraph GT3X file.\n\nParameters\n----------\nfilename : str\n\tPath to the GT3X file.\n\nReturns\n-------\ntime : numpy.ndarray\naccel : numpy.ndarray\nlight : numpy.ndarry\nindex : numpy.ndarray\nn_elem : int\n\tNumber of valid (read) elements in the time, accel, and light arrays.";

static struct PyMethodDef methods[] = {
    {"read_gt3x", read_gt3x, 1, docstring},
    {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "read_gt3x",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_gt3x_convert(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL){
        return NULL;
    }

    /* import the array object */
    import_array();

    /* XXXX add constants here */
    
    return m;
}
