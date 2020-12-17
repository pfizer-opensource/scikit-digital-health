#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>

extern void dominant_freq_1d(long *, double *, long *, double *, double *, double *, double *);
extern void dominant_freq_value_1d(long *, double *, long *, double *, double *, double *, double *);
extern void power_spectral_sum_1d(long *, double *, long *, double *, double *, double *, double *);
extern void spectral_entropy_1d(long *, double *, long *, double *, double *, double *, double *);
extern void destroy_plan();


PyObject * dominant_frequency(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long nfft;
    double fs = 0., low_cut=0., hi_cut=12.;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Olddd:dominant_frequency", &x_, &nfft, &fs, &low_cut, &hi_cut)) return NULL;

    if (fs <= 0.){
        PyErr_SetString(PyExc_ValueError, "Sampling frequency cannot be negative");
        return NULL;
    }
    if (hi_cut < low_cut){
        PyErr_SetString(PyExc_ValueError, "High frequency cutoff cannot be lower than low cutoff");
        return NULL;
    }

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

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
            dominant_freq_1d(&stride, dptr, &nfft, &fs, &low_cut, &hi_cut, rptr);
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
    // destroy the FFT plan created in the fortran module
    destroy_plan();

    return (PyObject *)res;
}


PyObject * dominant_frequency_value(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long nfft;
    double fs = 0., low_cut=0., hi_cut=12.;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Olddd:dominant_frequency_value", &x_, &nfft, &fs, &low_cut, &hi_cut)) return NULL;

    if (fs <= 0.){
        PyErr_SetString(PyExc_ValueError, "Sampling frequency cannot be negative");
        return NULL;
    }
    if (hi_cut < low_cut){
        PyErr_SetString(PyExc_ValueError, "High frequency cutoff cannot be lower than low cutoff");
        return NULL;
    }

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

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
            dominant_freq_value_1d(&stride, dptr, &nfft, &fs, &low_cut, &hi_cut, rptr);
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

    // destroy the FFT plan created in the fortran module
    destroy_plan();

    return (PyObject *)res;
}


PyObject * power_spectral_sum(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long nfft;
    double fs = 0., low_cut=0., hi_cut=12.;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Olddd:power_spectral_sum", &x_, &nfft, &fs, &low_cut, &hi_cut)) return NULL;

    if (fs <= 0.){
        PyErr_SetString(PyExc_ValueError, "Sampling frequency cannot be negative");
        return NULL;
    }
    if (hi_cut < low_cut){
        PyErr_SetString(PyExc_ValueError, "High frequency cutoff cannot be lower than low cutoff");
        return NULL;
    }

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

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
            power_spectral_sum_1d(&stride, dptr, &nfft, &fs, &low_cut, &hi_cut, rptr);
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

    // destroy the FFT plan created in the fortran module
    destroy_plan();

    return (PyObject *)res;
}


PyObject * spectral_entropy(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long nfft;
    double fs = 0., low_cut=0., hi_cut=12.;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Olddd:spectral_entropy", &x_, &nfft, &fs, &low_cut, &hi_cut)) return NULL;

    if (fs <= 0.){
        PyErr_SetString(PyExc_ValueError, "Sampling frequency cannot be negative");
        return NULL;
    }
    if (hi_cut < low_cut){
        PyErr_SetString(PyExc_ValueError, "High frequency cutoff cannot be lower than low cutoff");
        return NULL;
    }

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

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
            spectral_entropy_1d(&stride, dptr, &nfft, &fs, &low_cut, &hi_cut, rptr);
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

    // destroy the FFT plan created in the fortran module
    destroy_plan();

    return (PyObject *)res;
}


PyObject * spectral_flatness(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long nfft;
    double fs = 0., low_cut=0., hi_cut=12.;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "Olddd:spectral_flatness", &x_, &nfft, &fs, &low_cut, &hi_cut)) return NULL;

    if (fs <= 0.){
        PyErr_SetString(PyExc_ValueError, "Sampling frequency cannot be negative");
        return NULL;
    }
    if (hi_cut < low_cut){
        PyErr_SetString(PyExc_ValueError, "High frequency cutoff cannot be lower than low cutoff");
        return NULL;
    }

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

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
            spectral_flatness_1d(&stride, dptr, &nfft, &fs, &low_cut, &hi_cut, rptr);
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

    // destroy the FFT plan created in the fortran module
    destroy_plan();

    return (PyObject *)res;
}


static struct PyMethodDef methods[] = {
    {"dominant_frequency",   dominant_frequency,   1, NULL},  // last is test__doc__
    {"dominant_frequency_value",   dominant_frequency_value,   1, NULL},
    {"power_spectral_sum",   power_spectral_sum,   1, NULL},
    {"spectral_entropy",   spectral_entropy,   1, NULL},
    {"spectral_flatness",   spectral_flatness,   1, NULL},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "frequency",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_frequency(void)
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