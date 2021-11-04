// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>


extern void autocorr_1d(long *, double *, long *, int *, double *);
extern void linear_regression_1d(long *, double *, double *, double *);


PyObject * autocorrelation(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long lag;
    int norm;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Oli:autocorrelation", &x_, &lag, &norm)) return NULL;

    if (norm !=0 && norm != 1){
        PyErr_SetString(PyExc_ValueError, "norm argument must be 0/1");
        return NULL;
    }
    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

    // catch size 0 inputs
    if (PyArray_SIZE(data) == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Input data size must be larger than 0.");
        return NULL;
    }

    int ndim = PyArray_NDIM(data);

    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp *rdims = (npy_intp *)malloc((ndim - 1) * sizeof(npy_intp));
    if (!rdims){
        Py_XDECREF(data); return NULL;
    }
    for (int i = 0; i < (ndim - 1); ++i){
        rdims[i] = ddims[i];
    }

    PyArrayObject *res = (PyArrayObject *)PyArray_Empty(ndim-1, rdims, PyArray_DescrFromType(NPY_DOUBLE), 0);
    free(rdims);

    if (!res) fail = 1;
    if (!fail){
        double *dptr = (double *)PyArray_DATA(data);
        double *rptr = (double *)PyArray_DATA(res);

        long stride = ddims[ndim-1];
        int nrepeats = PyArray_SIZE(data) / stride;

        for (int i = 0; i < nrepeats; ++i){
            autocorr_1d(&stride, dptr, &lag, &norm, rptr);
            dptr += stride;
            rptr ++;
        }
    }
    if (fail){
        Py_XDECREF(data);
        Py_XDECREF(res);
        return NULL;
    }
    Py_XDECREF(data);

    return (PyObject *)res;
}


PyObject * linear_regression(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    double fs;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Od:linear_regression", &x_, &fs)) return NULL;

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;
    // catch size 0 inputs
    if (PyArray_SIZE(data) == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Input data size must be larger than 0.");
        return NULL;
    }

    int ndim = PyArray_NDIM(data);

    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp *rdims = (npy_intp *)malloc((ndim - 1) * sizeof(npy_intp));
    if (!rdims){
        Py_XDECREF(data); return NULL;
    }
    for (int i = 0; i < (ndim - 1); ++i){
        rdims[i] = ddims[i];
    }

    PyArrayObject *res = (PyArrayObject *)PyArray_Empty(ndim-1, rdims, PyArray_DescrFromType(NPY_DOUBLE), 0);
    free(rdims);

    if (!res) fail = 1;
    if (!fail){
        double *dptr = (double *)PyArray_DATA(data);
        double *rptr = (double *)PyArray_DATA(res);

        long stride = ddims[ndim-1];
        int nrepeats = PyArray_SIZE(data) / stride;

        for (int i = 0; i < nrepeats; ++i){
            linear_regression_1d(&stride, dptr, &fs, rptr);
            dptr += stride;
            rptr ++;
        }
    }
    if (fail){
        Py_XDECREF(data);
        Py_XDECREF(res);
        return NULL;
    }
    Py_XDECREF(data);

    return (PyObject *)res;
}



static struct PyMethodDef methods[] = {
    {"autocorrelation",   autocorrelation,   1, NULL},  // last is test__doc__
    {"linear_regression",   linear_regression,   1, NULL},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "statistics",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_statistics(void)
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
