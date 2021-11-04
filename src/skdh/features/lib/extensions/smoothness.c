// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>


extern void jerk_1d(long *, double *, double *, double *);
extern void dimensionless_jerk_1d(long *, double *, long *, double *);
extern void sparc_1d(long *, double *, double *, long *, double *, double *, double *);
extern void destroy_plan(void);

PyObject * jerk_metric(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    double fs;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Od:jerk_metric", &x_, &fs)) return NULL;

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
            jerk_1d(&stride, dptr, &fs, rptr);
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


PyObject * dimensionless_jerk_metric(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long stype;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Ol:dimensionless_jerk_metric", &x_, &stype)) return NULL;

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
            dimensionless_jerk_1d(&stride, dptr, &stype, rptr);
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


PyObject * SPARC(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    double fs, fc, amp_thresh;
    long padlevel;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Odldd:SPARC", &x_, &fs, &padlevel, &fc, &amp_thresh)) return NULL;

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
            sparc_1d(&stride, dptr, &fs, &padlevel, &fc, &amp_thresh, rptr);
            dptr += stride;
            rptr ++;
        }
    }
    if (fail){
        Py_XDECREF(data);
        Py_XDECREF(res);
        destroy_plan();  // destroy the FFT plan
        return NULL;
    }
    Py_XDECREF(data);

    destroy_plan();  // destroy the FFT plan
    return (PyObject *)res;
}


static struct PyMethodDef methods[] = {
    {"jerk_metric",   jerk_metric,   1, NULL},  // last is test__doc__
    {"dimensionless_jerk_metric", dimensionless_jerk_metric, 1, NULL},
    {"SPARC", SPARC, 1, NULL},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "smoothness",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_smoothness(void)
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
