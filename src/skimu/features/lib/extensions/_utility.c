#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>

extern void mean_sd_1d(long *, double *, double *, double *);
extern void unique(long *, real *, real *, long *, long *);
extern void gmean(long *, double *, double *);

extern void insertion_sort_2d(long *, long *, double *, long *);
extern void quick_argsort_(long *, double *, long *);
extern void quick_sort_(long *, double *);


PyObject * c_mean_sd_1d(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    int fail;

    if (!PyArg_ParseTuple(args, "O:c_mean_sd_1d", &x_)) return NULL;

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

    int ndim = PyArray_NDIM(data);

    if (ndim != 1){
        PyExc_ValueError("Number of dimensions cannot be other than 1.")
        Py_XDECREF(data);
        return NULL;
    }

    double mean = 0., stdev = 0.;
    double *dptr = (double *)PyArray_DATA(data);
    npy_intp *ddims = PyArray_DIMS(data);

    mean_sd_1d(&ddims[0], dptr, &mean, &stdev);

    Py_XDECREF(data);

    return Py_BuildValue(
        "OO",
        PyFloat_FromDouble(mean),
        PyFloat_FromDouble(stdev)
    )
}


static struct PyMethodDef methods[] = {
    {"c_mean_sd_1d",   permutation_entropy,   1, NULL},  // last is test__doc__
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_utility_extensions",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__utility_extensions(void)
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
