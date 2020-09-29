#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


extern void fautocorr(long *, long *, long *, double *, long *, int *, double *);


PyObject * autocorr(PyObject *NPY_UNUSED(self), PyObject *args){
    
    PyObject *x_;
    long lag;
    int norm;
    
    if (!PyArg_ParseTuple(args, "Oli:autocorr", &x_, &lag, &norm)) return NULL;
    
    if (norm != 0 && norm != 1){
        PyErr_SetString(PyExc_ValueError, "norm argument must be True/False");
        return NULL;
    }

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0, 
        NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST, NULL
    );
    if (!data) return NULL;
        
    int ndim = PyArray_NDIM(data);
    if (ndim != 3){
        PyErr_SetString(PyExc_ValueError, "Number of dimensions not 3.");
        return NULL;
    }
    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp rdims[2] = {ddims[0], ddims[1]};
    
    PyArrayObject *res = (PyArrayObject *)PyArray_Empty(
        2, rdims, PyArray_DescrFromType(NPY_DOUBLE), 0
    );
    
    double *dptr = (double *)PyArray_DATA(data);
    double *rptr = (double *)PyArray_DATA(res);
    
    fautocorr(&ddims[0], &ddims[1], &ddims[2], dptr, &lag, &norm, rptr);
    
    return (PyObject *)res;
    
}

static struct PyMethodDef methods[] = {
    {"autocorr",   autocorr,   1, NULL},  // last is test__doc__
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "autocorr",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_acorr(void)
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
