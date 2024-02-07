#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdlib.h>
#include "window_days.h"


PyObject * cwindow_days(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *a, *bases_, *periods_;
    double fs;

    if (!PyArg_ParseTuple(args, "OdOO:window_c", &a, &fs, &bases_, &periods_)) return NULL;

    PyArrayObject *time = (PyArrayObject *)PyArray_FromAny(
        a, PyArray_DescrFromType(NPY_DOUBLE), 1, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    PyArrayObject *bases = (PyArrayObject *)PyArray_FromAny(
        bases_, PyArray_DescrFromType(NPY_LONG), 1, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    PyArrayObject *periods = (PyArrayObject *)PyArray_FromAny(
        periods_, PyArray_DescrFromType(NPY_LONG), 1, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );

    if (!time || !bases || !periods)
    {
        Py_XDECREF(time);
        Py_XDECREF(bases);
        Py_XDECREF(periods);
        return NULL;
    }


    // catch size 0 inputs
    if (PyArray_SIZE(time) == 0)
    {
        Py_XDECREF(time); Py_XDECREF(bases); Py_XDECREF(periods);
        PyErr_SetString(PyExc_ValueError, "Input time array size must be larger than 0.");
        return NULL;
    }
    if (PyArray_SIZE(bases) == 0)
    {
        Py_XDECREF(time); Py_XDECREF(bases); Py_XDECREF(periods);
        PyErr_SetString(PyExc_ValueError, "Input bases array size must be larger than 0.");
        return NULL;
    }
    if (PyArray_SIZE(periods) == 0)
    {
        Py_XDECREF(time); Py_XDECREF(bases); Py_XDECREF(periods);
        PyErr_SetString(PyExc_ValueError, "Input periods array size must be larger than 0.");
        return NULL;
    }
    // ensure bases and periods are the same size
    size_t w_N = (size_t)PyArray_SIZE(bases);
    if (w_N != (size_t)PyArray_SIZE(periods))
    {
        Py_XDECREF(time); Py_XDECREF(bases); Py_XDECREF(periods);
        PyErr_SetString(PyExc_ValueError, "Bases and periods must have the same number of elements.");
        return NULL;
    }

    // make sure time array is 1d
    if (PyArray_NDIM(time) != 1)
    {
        Py_XDECREF(time); Py_XDECREF(bases); Py_XDECREF(periods);
        PyErr_SetString(PyExc_ValueError, "Time array must be 1D.");
        return NULL;
    }

    // pointer to the actual time data, bases, and periods
    double *dptr = (double *)PyArray_DATA(time);
    long *bptr = (long *)PyArray_DATA(bases);
    long *pptr = (long *)PyArray_DATA(periods);
    // length of data
    size_t N = (size_t)PyArray_SIZE(time);

    // figure out return value size
    size_t ndays = (size_t)ceil((dptr[N - 1] - dptr[0]) / 86400.0) + 2;
    npy_intp dims[] = {w_N, ndays, 2};

    // allocate starts and stops arrays
    PyArrayObject *windows = (PyArrayObject *)PyArray_ZEROS(3, dims, NPY_LONG, 0);
    if (!windows)
    {
        Py_XDECREF(time); Py_XDECREF(bases); Py_XDECREF(periods);
        return NULL;
    }

    // pointer for the windows data
    long *wptr = (long *)PyArray_DATA(windows);
    // strides for each pair of bases and periods in the output array
    long stride = ndays * 2;

    for (size_t i = 0; i < w_N; ++i)
    {
        window(N, dptr, fs, bptr, pptr, wptr);
        // stride through the window array
        wptr += stride;
        ++bptr;
        ++pptr;
    }

    Py_XDECREF(time);
    Py_XDECREF(bases);
    Py_XDECREF(periods);

    return Py_BuildValue("N", (PyObject *)windows);
}


static const char cwindow_days__doc__[] = "window_c(timestamps, fs, bases, periods)\n"
"Compute the windows for a timestamp array.\n\n"
"Parameters\n"
"----------\n"
"timestamps : numpy.ndarray\n"
"    1D array of timestamp values, in seconds since 1970-01-01. Assumed to be in local time.\n"
"fs : float\n"
"    Sampling frequency in Hz for `timestamps`.\n"
"bases : array-like\n"
"    The start hours of windows, in 24hr format.\n"
"periods : array-like\n"
"    The length, in hours, of the windows.\n\n"
"Returns\n"
"-------\n"
"window_indices : numpy.ndarray\n"
"    `(bases.size, n_days+2, 2)` array of indices. Last dimension is `[i_start, i_stop]`.";

static struct PyMethodDef methods[] = {
    {"cwindow_days", cwindow_days, 1, cwindow_days__doc__},
    {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pywindow",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};


/* initialization function for the module */
PyMODINIT_FUNC PyInit_pywindow(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    import_array();

    return m;
}