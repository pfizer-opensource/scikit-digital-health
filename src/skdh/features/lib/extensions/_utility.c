// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void mean_sd_1d(long *, double *, double *, double *);
extern void unique(long *, double *, double *, long *, long *);
extern void gmean(long *, double *, double *);
extern void embed_sort(long *, long *, double *, long *, long *, long *);
extern void hist(long *, double *, long *, double *, double *, long *);
extern void histogram(long *, long *, double *, double *, long *);

extern void insertion_sort_2d(long *, long *, double *, long *);
extern void quick_argsort_(long *, double *, long *);
extern void quick_sort_(long *, double *);

extern void f_rfft(long *, double *, long *, double *);
extern void destroy_plan(void);


PyObject * cf_mean_sd_1d(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;

    if (!PyArg_ParseTuple(args, "O:cf_mean_sd_1d", &x_)) return NULL;

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

    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions cannot be other than 1.");
        Py_XDECREF(data);
        return NULL;
    }

    double mean = 0., stdev = 0.;
    double *dptr = (double *)PyArray_DATA(data);
    npy_intp n_elem = PyArray_SIZE(data);

    mean_sd_1d(&n_elem, dptr, &mean, &stdev);

    Py_XDECREF(data);

    return Py_BuildValue(
        "dd",
        PyFloat_FromDouble(mean),
        PyFloat_FromDouble(stdev)
    );
}

PyObject * cf_unique(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;

    if (!PyArg_ParseTuple(args, "O:cf_unique", &x_)) return NULL;

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

    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions cannot be other than 1.");
        Py_XDECREF(data);
        return NULL;
    }
    npy_intp n_elem = PyArray_SIZE(data);

    PyArrayObject *unq = (PyArrayObject *)PyArray_ZEROS(1, &n_elem, NPY_DOUBLE, 0);
    PyArrayObject *cnt = (PyArrayObject *)PyArray_ZEROS(1, &n_elem, NPY_LONG, 0);

    double *dptr = (double *)PyArray_DATA(data);
    double *uptr = (double *)PyArray_DATA(unq);
    long *cptr = (long *)PyArray_DATA(cnt);
    long n_unique = 0;

    unique(&n_elem, dptr, uptr, cptr, &n_unique);

    Py_XDECREF(data);

    return Py_BuildValue(
        "OOl",
        (PyObject *)unq,
        (PyObject *)cnt,
        PyLong_FromLong(n_unique)
    );
}


PyObject * cf_gmean(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;

    if (!PyArg_ParseTuple(args, "O:cf_gmean", &x_)) return NULL;

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

    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions cannot be other than 1.");
        Py_XDECREF(data);
        return NULL;
    }

    npy_intp n_elem = PyArray_SIZE(data);

    double geo_mean = 0.;
    double *dptr = (double *)PyArray_DATA(data);

    gmean(&n_elem, dptr, &geo_mean);

    Py_XDECREF(data);

    return PyFloat_FromDouble(geo_mean);
}


PyObject * cf_embed_sort(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long order = 3, delay = 1;

    if (!PyArg_ParseTuple(args, "Oll:cf_embed_sort", &x_, &order, &delay)) return NULL;

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

    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions cannot be other than 1.");
        Py_XDECREF(data);
        return NULL;
    }

    npy_intp n_elem = PyArray_SIZE(data);

    long nsi = n_elem - (order - 1) * delay;

    npy_intp rdims[2] = {nsi, order};

    PyArrayObject *res = (PyArrayObject *)PyArray_EMPTY(2, rdims, NPY_LONG, 0);

    double *dptr = (double *)PyArray_DATA(data);
    long *rptr = (long *)PyArray_DATA(res);

    embed_sort(&n_elem, &nsi, dptr, &order, &delay, rptr);

    Py_XDECREF(data);

    return (PyObject *)res;
}


PyObject * cf_hist(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long ncells = 1;
    double min_val = 0., max_val = 100.;

    if (!PyArg_ParseTuple(args, "Oldd:cf_hist", &x_, &ncells, &min_val, &max_val)) return NULL;

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

    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions cannot be other than 1.");
        Py_XDECREF(data);
        return NULL;
    }

    npy_intp n_elem = PyArray_SIZE(data);

    PyArrayObject *res = (PyArrayObject *)PyArray_ZEROS(1, &ncells, NPY_LONG, 0);

    double *dptr = (double *)PyArray_DATA(data);
    long *rptr = (long *)PyArray_DATA(res);

    hist(&n_elem, dptr, &ncells, &min_val, &max_val, rptr);

    Py_XDECREF(data);

    return (PyObject *)res;
}


PyObject * cf_histogram(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;

    if (!PyArg_ParseTuple(args, "O:cf_histogram", &x_)) return NULL;

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

    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions cannot be other than 1.");
        Py_XDECREF(data);
        return NULL;
    }

    npy_intp n_elem = PyArray_SIZE(data);

    long n3 = 3;
    long ncells = (long)ceil(sqrt(n_elem));

    PyArrayObject *d = (PyArrayObject *)PyArray_EMPTY(1, &n3, NPY_DOUBLE, 0);
    PyArrayObject *counts = (PyArrayObject *)PyArray_ZEROS(1, &ncells, NPY_LONG, 0);

    double *data_ptr = (double *)PyArray_DATA(data);
    double *d_ptr = (double *)PyArray_DATA(d);
    long *counts_ptr = (long *)PyArray_DATA(counts);

    histogram(&n_elem, &ncells, data_ptr, d_ptr, counts_ptr);

    Py_XDECREF(data);

    return Py_BuildValue(
        "OO",
        (PyObject *)d,
        (PyObject *)counts
    );
}


PyObject * cf_rfft(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long nfft;

    if (!PyArg_ParseTuple(args, "Ol:cf_rfft", &x_, &nfft)) return NULL;

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
    if (ndim != 1){
        PyErr_SetString(PyExc_ValueError, "Dimensions over 1 not supported");
        return NULL;
    }
    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp rdims[1] = {2 * nfft + 2};

    PyArrayObject *res = (PyArrayObject *)PyArray_EMPTY(1, rdims, NPY_DOUBLE, 0);
    if (!res){
        PyErr_SetString(PyExc_ValueError, "Failed to create results array.");
        Py_XDECREF(data);
        return NULL;
    }
    double *dptr = (double *)PyArray_DATA(data);
    double *rptr = (double *)PyArray_DATA(res);

    f_rfft(&ddims[0], dptr, &nfft, rptr);

    Py_XDECREF(data);
    destroy_plan();

    return (PyObject *)res;
}


static struct PyMethodDef methods[] = {
    {"cf_mean_sd_1d",   cf_mean_sd_1d,   1, NULL},  // last is test__doc__
    {"cf_unique",   cf_unique,   1, NULL},
    {"cf_gmean",   cf_gmean,   1, NULL},
    {"cf_embed_sort",   cf_embed_sort,   1, NULL},
    {"cf_hist",   cf_hist,   1, NULL},
    {"cf_histogram",   cf_histogram,   1, NULL},
    {"cf_rfft", cf_rfft, 1, NULL},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_utility",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__utility(void)
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
