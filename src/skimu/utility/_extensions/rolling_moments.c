#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern void rolling_moments_1(long *, double *, long *, long *, double *);
extern void rolling_moments_2(long *, double *, long *, long *, double *, double *);
extern void rolling_moments_3(long *, double *, long *, long *, double *, double *, double *);
extern void rolling_moments_4(long *, double *, long *, long *, double *, double *, double *, double *);


PyObject * rolling_mean(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long wlen, skip;

    if (!PyArg_ParseTuple(args, "Oll:rolling_mean", &x_, &wlen, &skip)) return NULL;

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

    // get the number of dimensions, and the shape
    int ndim = PyArray_NDIM(data);
    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp *rdims = (npy_intp *)malloc(ndim * sizeof(npy_intp));
    if (!rdims){
        Py_XDECREF(data);
        return NULL;
    }
    // create return shape
    for (int i = 0; i < (ndim - 1); ++i){
        rdims[i] = ddims[i];
    }
    rdims[ndim-1] = (ddims[ndim-1] - wlen) / skip + 1;  // dimension of the roll

    PyArrayObject *rmean = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0);

    if (!rmean){
        Py_XDECREF(data);
        Py_XDECREF(rmean);
        return NULL;
    }

    // data pointers
    double *dptr = (double *)PyArray_DATA(data),
           *rmean_ptr = (double *)PyArray_DATA(rmean);
    // for iterating over the data
    long stride = ddims[ndim-1];  // stride to get to the next computation "column"
    long res_stride = rdims[ndim-1];  // stride to get to the next results "column"
    int nrepeats = PyArray_SIZE(data) / stride;  // number of repetitions to cover all the data

    // has to be freed down here since its used by res_stride
    free(rdims);

    for (int i = 0; i < nrepeats; ++i){
        rolling_moments_1(&stride, dptr, &wlen, &skip, rmean_ptr);
        dptr += stride;
        rmean_ptr += res_stride;
    }

    Py_XDECREF(data);

    return (PyObject *)rmean;

}


PyObject * rolling_sd(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long wlen, skip;
    int return_others;

    if (!PyArg_ParseTuple(args, "Ollp:rolling_sd", &x_, &wlen, &skip, &return_others)) return NULL;

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

    // get the number of dimensions, and the shape
    int ndim = PyArray_NDIM(data);
    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp *rdims = (npy_intp *)malloc(ndim * sizeof(npy_intp));
    if (!rdims){
        Py_XDECREF(data);
        return NULL;
    }
    // create return shape
    for (int i = 0; i < (ndim - 1); ++i){
        rdims[i] = ddims[i];
    }
    rdims[ndim-1] = (ddims[ndim-1] - wlen) / skip + 1;  // dimension of the roll

    PyArrayObject *rsd   = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0),
                  *rmean = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0);

    if ((!rmean) || (!rsd)){
        Py_XDECREF(data);
        Py_XDECREF(rmean);
        Py_XDECREF(rsd);
        return NULL;
    }

    // data pointers
    double *dptr      = (double *)PyArray_DATA(data),
           *rmean_ptr = (double *)PyArray_DATA(rmean),
           *rsd_ptr   = (double *)PyArray_DATA(rsd);
    // for iterating over the data
    long stride = ddims[ndim-1];  // stride to get to the next computation "column"
    long res_stride = rdims[ndim-1];  // stride to get to the next results "column"
    int nrepeats = PyArray_SIZE(data) / stride;  // number of repetitions to cover all the data
    // has to be freed down here since its used by res_stride
    free(rdims);

    for (int i = 0; i < nrepeats; ++i){
        rolling_moments_2(&stride, dptr, &wlen, &skip, rmean_ptr, rsd_ptr);
        dptr += stride;
        rmean_ptr += res_stride;
        rsd_ptr += res_stride;
    }
    
    Py_XDECREF(data);

    if (return_others){
        return Py_BuildValue(
            "OO",
            (PyObject *)rsd,
            (PyObject *)rmean
        );
    } else {
        Py_XDECREF(rmean);
        return (PyObject *)rsd;
    }
}


PyObject * rolling_skewness(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long wlen, skip;
    int return_others;

    if (!PyArg_ParseTuple(args, "Ollp:rolling_skewness", &x_, &wlen, &skip, &return_others)) return NULL;

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

    // get the number of dimensions, and the shape
    int ndim = PyArray_NDIM(data);
    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp *rdims = (npy_intp *)malloc(ndim * sizeof(npy_intp));
    if (!rdims){
        Py_XDECREF(data);
        return NULL;
    }
    // create return shape
    for (int i = 0; i < (ndim - 1); ++i){
        rdims[i] = ddims[i];
    }
    rdims[ndim-1] = (ddims[ndim-1] - wlen) / skip + 1;  // dimension of the roll

    PyArrayObject *rsd   = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0),
                  *rmean = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0),
                  *rskew = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0);

    if ((!rmean) || (!rsd) || (!rskew)){
        Py_XDECREF(data);
        Py_XDECREF(rskew);
        Py_XDECREF(rsd);
        Py_XDECREF(rmean);
        return NULL;
    }

    // data pointers
    double *dptr      = (double *)PyArray_DATA(data),
           *rmean_ptr = (double *)PyArray_DATA(rmean),
           *rsd_ptr   = (double *)PyArray_DATA(rsd),
           *rskew_ptr = (double *)PyArray_DATA(rskew);
    // for iterating over the data
    long stride = ddims[ndim-1];  // stride to get to the next computation "column"
    long res_stride = rdims[ndim-1];  // stride to get to the next results "column"
    int nrepeats = PyArray_SIZE(data) / stride;  // number of repetitions to cover all the data
    // has to be freed down here since its used by res_stride
    free(rdims);

    for (int i = 0; i < nrepeats; ++i){
        rolling_moments_3(&stride, dptr, &wlen, &skip, rmean_ptr, rsd_ptr, rskew_ptr);
        dptr += stride;
        rmean_ptr += res_stride;
        rsd_ptr += res_stride;
        rskew_ptr += res_stride;
    }
    
    Py_XDECREF(data);

    if (return_others){
        return Py_BuildValue(
            "OOO",
            (PyObject *)rskew,
            (PyObject *)rsd,
            (PyObject *)rmean
        );
    } else {
        Py_XDECREF(rmean);
        Py_XDECREF(rsd);
        return (PyObject *)rskew;
    }
}


PyObject * rolling_kurtosis(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *x_;
    long wlen, skip;
    int return_others;

    if (!PyArg_ParseTuple(args, "Ollp:rolling_kurtosis", &x_, &wlen, &skip, &return_others)) return NULL;

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        x_, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL
    );
    if (!data) return NULL;

    // get the number of dimensions, and the shape
    int ndim = PyArray_NDIM(data);
    npy_intp *ddims = PyArray_DIMS(data);
    npy_intp *rdims = (npy_intp *)malloc(ndim * sizeof(npy_intp));
    if (!rdims){
        Py_XDECREF(data);
        return NULL;
    }
    // create return shape
    for (int i = 0; i < (ndim - 1); ++i){
        rdims[i] = ddims[i];
    }
    rdims[ndim-1] = (ddims[ndim-1] - wlen) / skip + 1;  // dimension of the roll

    PyArrayObject *rsd   = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0),
                  *rmean = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0),
                  *rskew = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0),
                  *rkurt = (PyArrayObject *)PyArray_EMPTY(ndim, rdims, NPY_DOUBLE, 0);

    if (!rmean || !rsd || !rskew || !rkurt){
        Py_XDECREF(data);
        Py_XDECREF(rkurt);
        Py_XDECREF(rskew);
        Py_XDECREF(rsd);
        Py_XDECREF(rmean);
        return NULL;
    }

    // data pointers
    double *dptr      = (double *)PyArray_DATA(data),
           *rmean_ptr = (double *)PyArray_DATA(rmean),
           *rsd_ptr   = (double *)PyArray_DATA(rsd),
           *rskew_ptr = (double *)PyArray_DATA(rskew),
           *rkurt_ptr = (double *)PyArray_DATA(rkurt);
    // for iterating over the data
    long stride = ddims[ndim-1];  // stride to get to the next computation "column"
    long res_stride = rdims[ndim-1];  // stride to get to the next results "column"
    int nrepeats = PyArray_SIZE(data) / stride;  // number of repetitions to cover all the data
    // has to be freed down here since its used by res_stride
    free(rdims);

    for (int i = 0; i < nrepeats; ++i){
        rolling_moments_4(&stride, dptr, &wlen, &skip, rmean_ptr, rsd_ptr, rskew_ptr, rkurt_ptr);
        dptr += stride;
        rmean_ptr += res_stride;
        rsd_ptr += res_stride;
        rskew_ptr += res_stride;
        rkurt_ptr += res_stride;
    }
    
    Py_XDECREF(data);

    if (return_others){
        return Py_BuildValue(
            "OOOO",
            (PyObject *)rkurt,
            (PyObject *)rskew,
            (PyObject *)rsd,
            (PyObject *)rmean
        );
    } else {
        Py_XDECREF(rmean);
        Py_XDECREF(rsd);
        Py_XDECREF(rskew);
        return (PyObject *)rkurt;
    }
}


static const char rmean_doc[] = "rolling_mean(a, wlen, skip)\n\n"
"Compute the rolling mean over windows of length `wlen` with `skip` samples between window starts.\n\n"
"Paramters\n"
"---------\n"
"a : array-like\n"
"    Array of data to compute the rolling mean for. Computation axis is the last axis.\n"
"wlen : int\n"
"    Window size in samples.\n"
"skip : int\n"
"    Samples between window starts. `skip=wlen` would result in non-overlapping sequential windows.\n\n"
"Returns\n"
"-------\n"
"rmean : numpy.ndarray\n"
"    Rolling mean.";

static const char rsd_doc[] = "rolling_sd(a, wlen, skip, return_previous)\n\n"
"Compute the rolling standard deviation over windows of length `wlen` with `skip` samples "
"between window starts.  Because previous rolling moments have to be computed as part of "
"the process, they are availble to return as well.\n\n"
"Paramters\n"
"---------\n"
"a : array-like\n"
"    Array of data to compute the rolling standar deviation for. Computation axis is the last axis.\n"
"wlen : int\n"
"    Window size in samples.\n"
"skip : int\n"
"    Samples between window starts. `skip=wlen` would result in non-overlapping sequential windows.\n"
"return_previous : bool\n"
"    Return the previous rolling moments."
"Returns\n"
"-------\n"
"rsd : numpy.ndarray\n"
"    Rolling sample standard deviation.\n"
"rmean : numpy.ndarray, optional\n"
"    Rolling mean. Only returned if `return_previous` is `True`.";

static const char rskew_doc[] = "rolling_skewness(a, wlen, skip, return_previous)\n\n"
"Compute the rolling skewness over windows of length `wlen` with `skip` samples "
"between window starts.  Because previous rolling moments have to be computed as part of "
"the process, they are availble to return as well.\n\n"
"Paramters\n"
"---------\n"
"a : array-like\n"
"    Array of data to compute the rolling skewness for. Computation axis is the last axis.\n"
"wlen : int\n"
"    Window size in samples.\n"
"skip : int\n"
"    Samples between window starts. `skip=wlen` would result in non-overlapping sequential windows.\n"
"return_previous : bool\n"
"    Return the previous rolling moments."
"Returns\n"
"-------\n"
"rskew : numpy.ndarray\n"
"    Rolling skewness.\n"
"rsd : numpy.ndarray, optional\n"
"    Rolling sample standard deviation. Only returned if `return_previous` is `True`.\n"
"rmean : numpy.ndarray, optional\n"
"    Rolling mean. Only returned if `return_previous` is `True`.";

static const char rkurt_doc[] = "rolling_kurtosis(a, wlen, skip, return_previous)\n\n"
"Compute the rolling kurtosis over windows of length `wlen` with `skip` samples "
"between window starts.  Because previous rolling moments have to be computed as part of "
"the process, they are availble to return as well.\n\n"
"Paramters\n"
"---------\n"
"a : array-like\n"
"    Array of data to compute the rolling kurtosis for. Computation axis is the last axis.\n"
"wlen : int\n"
"    Window size in samples.\n"
"skip : int\n"
"    Samples between window starts. `skip=wlen` would result in non-overlapping sequential windows.\n"
"return_previous : bool\n"
"    Return the previous rolling moments."
"Returns\n"
"-------\n"
"rkurt : numpy.ndarray\n"
"    Rolling kurtosis.\n"
"rskew : numpy.ndarray, optional\n"
"    Rolling skewness. Only returned if `return_previous` is `True`.\n"
"rsd : numpy.ndarray, optional\n"
"    Rolling sample standard deviation. Only returned if `return_previous` is `True`.\n"
"rmean : numpy.ndarray, optional\n"
"    Rolling mean. Only returned if `return_previous` is `True`.";

static struct PyMethodDef methods[] = {
    {"rolling_mean",   rolling_mean,   1, rmean_doc},  // last is the docstring
    {"rolling_sd",   rolling_sd,   1, rsd_doc},  // last is the docstring
    {"rolling_skewness",   rolling_skewness,   1, rskew_doc},  // last is the docstring
    {"rolling_kurtosis",   rolling_kurtosis,   1, rkurt_doc},  // last is the docstring
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "rolling_moments",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_rolling_moments(void)
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
