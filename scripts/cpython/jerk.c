#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


static int jerk(int npts, double x[], double fs, double *ret){
    double jsum = 0.;
    double xold = x[0];
    double amp = fabs(xold);
    
    for (size_t i=1; i<npts; i++){
        jsum += pow(x[i] - xold, 2.);
        if (fabs(x[i]) > amp) amp = fabs(x[i]);
        xold = x[i];
    }
    ret[0] = jsum / (720. * pow(amp, 2.)) * fs;
    return 0;
}

static PyObject * jerk_metric(PyObject *NPY_UNUSED(self), PyObject *args){
    PyObject *a1;
    double fs;

    if(!PyArg_ParseTuple(args, "Od:jerk_metric", &a1, &fs)){
        return NULL;
    }
    int fail = 0;
    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(
        a1, PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
        NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
        NULL
    );
    if (!data) return NULL;
    
    int ndim = PyArray_NDIM(data);
    if (ndim != 3){
        Py_XDECREF(data);
        return NULL;
    }
    const npy_intp *odim = PyArray_DIMS(data);
    int npts = odim[ndim - 1];
    
    npy_intp *rdim = (npy_intp *)malloc((ndim-1)*sizeof(npy_intp));
    if (!rdim) { Py_XDECREF(data); return NULL; }
    for (int d=0; d<ndim-1; ++d) rdim[d] = odim[d];
    PyArrayObject *ret = (PyArrayObject *)PyArray_Empty(ndim-1,
            rdim, PyArray_DescrFromType(NPY_DOUBLE), 0);
    free(rdim);
    
    if (!ret) fail=1;
    if (!fail){
        double *dptr = (double *)PyArray_DATA(data);
        double *rptr = (double *)PyArray_DATA(ret);
        
        int nrepeats = PyArray_SIZE(data) / npts;
        
        for (int i=0; i<nrepeats; i++){
            if (jerk(npts, dptr, fs, rptr) != 0){
                fail = 1;
                break;
            }
            rptr += 1;  // increment 1 place for every window
            dptr += npts;  // increment 1 window every loop
        }
    }
    if (fail){
        Py_XDECREF(data);
        Py_XDECREF(ret);
        return PyErr_NoMemory();
    }
    Py_DECREF(data);
    return (PyObject *)ret;
}


static const char jerk_metric__doc__[] = "Jerk Metric Documentation";

static struct PyMethodDef methods[] = {
    {"jerk_metric",   jerk_metric,   1, jerk_metric__doc__},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_jerkmetric",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__jerkmetric(void)
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