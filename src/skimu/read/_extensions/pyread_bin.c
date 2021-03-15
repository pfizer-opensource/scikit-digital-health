#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "read_bin.h"

#define STR2PY PyUnicode_FromString

void set_error_message(int ierr)
{
    switch (ierr)
    {
        case READ_E_BLOCK_TIMESTAMP :
            PyErr_SetString(PyExc_RuntimeError, "Error reading tiemstamp from data block");
            break;
        case READ_E_BLOCK_FS :
            PyErr_SetString(PyExc_RuntimeError, "Block sampling frequency does not match header");
            break;
        case READ_E_BLOCK_DATA :
            PyErr_SetString(PyExc_RuntimeError, "Error reading data from data block");
            break;
        case READ_E_BLOCK_DATA_LEN :
            PyErr_SetString(PyExc_RuntimeError, "Data length is shorter than 3600");
            break;
        default :
            PyErr_SetString(PyExc_RuntimeError, "Unkown error reading GeneActiv file");
    }
}

static PyObject *read_bin(PyObject *NPY_UNUSED(self), PyObject *args)
{
    char *file;
    int ierr = READ_E_NONE, fail = 0;
    PyObject *bases_, *periods_;

    FILE *fp;
    Info_t info;
    Data_t data;
    Window_t winfo;

    /* INITIALIZATION */
    info.max_n = 0;
    info.npages = -1;

    /* PYTHON ARGUMENTS */
    if (!PyArg_ParseTuple(args, "sOO:read_bin", &file, &bases_, &periods_))
        return NULL;  /* error is set for us */
    
    /* GET NUMPY ARRAYS */
    PyArrayObject *bases = (PyArrayObject *)PyArray_FromAny(
        bases_,
        PyArray_DescrFromType(NPY_LONG),
        1,
        0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO,  /* ensure its an array and c-contiguous */
        NULL
    );
    PyArrayObject *periods = (PyArrayObject *)PyArray_FromAny(
        periods_,
        PyArray_DescrFromType(NPY_LONG),
        1,
        0,
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO,  /* ensure its an array and c-contiguous */
        NULL
    );

    if (!bases || !periods)
        return NULL;
    
    /* WINDOING INFO INIT */
    winfo.n = PyArray_Size(bases);
    if (winfo.n != PyArray_Size(periods))
    {
        PyErr_SetString(PyExc_ValueError, "Size mismatch between bases and periods");
        return NULL;
    }
    winfo.i_start = (long *)malloc(winfo.n * sizeof(winfo.i_start));
    winfo.i_stop = (long *)malloc(winfo.n * sizeof(winfo.i_stop));
    winfo.bases = (long *)PyArray_DATA(bases);
    winfo.periods = (long *)PyArray_DATA(periods);
    /* set the indices to 0 initially */
    memset(winfo.i_start, 0, winfo.n * sizeof(winfo.i_stop));
    memset(winfo.i_stop, 0, winfo.n * sizeof(winfo.i_stop));
    
    /* OPEN THE FILE */
    fp = fopen(file, "r");
    if (!fp)
    {
        PyErr_SetString(PyExc_IOError, "Error opening file");
        return NULL;
    }

    /* READ THE HEADER */
    read_header(fp, &info);

    if (info.npages == -1)
    {
        PyErr_SetString(PyExc_IOError, "Cannot read number of blocks");
        return NULL;
    }

    /* DIMENSIONS FOR RETURN VALUES */
    npy_intp dim3[2] = {info.npages * PAGE_SAMPLES, 3};
    npy_intp dim1[1] = {info.npages * PAGE_SAMPLES};

    long ndays = (long)ceil(((double)info.npages * (FPAGE_SAMPLES / info.fs)) / 86400.0);

    npy_intp dim_idx[2] = {ndays + 1, winfo.n};

    /* DATA ARRAYS */
    PyArrayObject *accel = (PyArrayObject *)PyArray_ZEROS(2, dim3, NPY_DOUBLE, 0);
    PyArrayObject *time  = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);
    PyArrayObject *light = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);
    PyArrayObject *temp  = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);

    PyArrayObject *starts = (PyArrayObject *)PyArray_ZEROS(2, dim_idx, NPY_LONG, 0);
    PyArrayObject *stops  = (PyArrayObject *)PyArray_ZEROS(2, dim_idx, NPY_LONG, 0);

    if (!accel || !time || !light || !temp || !starts || !stops)
    {
        Py_XDECREF(accel);
        Py_XDECREF(time);
        Py_XDECREF(temp);
        Py_XDECREF(light);
        Py_XDECREF(starts);
        Py_XDECREF(stops);

        free(winfo.i_start);
        free(winfo.i_stop);

        return NULL;
    }

    /* SET POINTERS */
    data.acc   = (double *)PyArray_DATA(accel);
    data.ts    = (double *)PyArray_DATA(time);
    data.light = (double *)PyArray_DATA(light);
    data.temp  = (double *)PyArray_DATA(temp);
    data.day_starts = (long *)PyArray_DATA(starts);
    data.day_stops  = (long *)PyArray_DATA(stops);
    
    /* READ FILE */
    for (int i = 0; i < info.npages; ++i)
    {
        ierr = read_block(fp, &winfo, &info, &data);

        if (ierr != READ_E_NONE)
        {
            fail = 1;
            break;
        }
    }

    fclose(fp);
    free(winfo.i_start);
    free(winfo.i_stop);

    if (fail)
    {
        Py_XDECREF(accel);
        Py_XDECREF(time);
        Py_XDECREF(temp);
        Py_XDECREF(light);
        Py_XDECREF(starts);
        Py_XDECREF(stops);

        set_error_message(ierr);
        return NULL;
    }

    return Py_BuildValue(
        "lfOOOOOO",
        info.max_n,
        info.fs,
        (PyObject *)accel,
        (PyObject *)time,
        (PyObject *)light,
        (PyObject *)temp,
        (PyObject *)starts,
        (PyObject *)stops
    );
}

static const char read_bin__doc__[] = "read_bin(file, bases, periods)\n"
"Read a Geneactiv File\n\n"
"Parameters\n"
"----------\n"
"file : str\n"
"   File name to read from\n"
"bases : numpy.ndarray\n"
"   Base times for providing windowing. Must be in [0, 23]\n"
"periods : numpy.ndarray\n"
"   Number of hours for each window. Must be in [1, 24]\n\n"
"Returns\n"
"-------\n"
"N : int\n"
"   Number of pages read.\n"
"fs : float\n"
"   Sampling frequency in Hz.\n"
"accel : numpy.ndarray\n"
"time : numpy.ndarray\n"
"light : numpy.ndarray\n"
"temp : numpy.ndarray\n"
"starts : numpy.ndarray\n"
"stops : numpy.ndarray\n";

static struct PyMethodDef methods[] = {
  {"read_bin", read_bin, 1, bin_convert__doc__},
  {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "read_bin",
  NULL,
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};


/* Initialization function for the module */
PyMODINIT_FUNC PyInit_read_bin(void){
  PyObject *m;
  m = PyModule_Create(&moduledef);
  if (m == NULL){
    return NULL;
  }

  /* import the array object */
  import_array();

  /* add constants here */

  return m;
}
