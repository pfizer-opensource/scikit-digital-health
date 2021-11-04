// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>

extern void signal_entropy_1d(long *, double *, double *);
extern void sample_entropy_1d(long *, double *, long *, double *, double *);
extern void permutation_entropy_1d(long *, double *, long *, long *, int *, double *);


PyObject * signal_entropy(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "O:signal_entropy", &x_)) return NULL;

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
            signal_entropy_1d(&stride, dptr, rptr);
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


PyObject * sample_entropy(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long L;
    double r;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Old:sample_entropy", &x_, &L, &r)) return NULL;

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
            sample_entropy_1d(&stride, dptr, &L, &r, rptr);
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


PyObject * permutation_entropy(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long order, delay;
    int normalize;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Olli:permutation_entropy", &x_, &order, &delay, &normalize)) return NULL;

    if (normalize != 0) normalize = 1;  // make sure set to 1 if not 0

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
            permutation_entropy_1d(&stride, dptr, &order, &delay, &normalize, rptr);
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
    {"signal_entropy",   signal_entropy,   1, NULL},
    {"sample_entropy",   sample_entropy,   1, NULL},
    {"permutation_entropy",   permutation_entropy,   1, NULL},  // last is test__doc__
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "entropy",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_entropy(void)
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
