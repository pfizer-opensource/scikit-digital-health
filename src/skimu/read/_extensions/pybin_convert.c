#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "bin_convert.h"

#define STR2PY PyUnicode_FromString


void set_error_message(int ierr){
    switch(ierr){
        case TIMESTAMP_ERROR :
            PyErr_SetString(PyExc_RuntimeError, "Error reading timestamp from a data block.");
            break;
        case DATA_ERROR :
            PyErr_SetString(PyExc_RuntimeError, "Error reading data from a data block.");
            break;
        case DATA_LEN_ERROR :
            PyErr_SetString(PyExc_RuntimeError, "Data length is not as long as expected.");
            break;
        default :
            PyErr_SetString(PyExc_RuntimeError, "Error reading GeneActiv file.");
    }
}


static PyObject * read_bin(PyObject *NPY_UNUSED(self), PyObject *args){
    char *file;
    int ierr = -1, fail = 0;
    long base, period;
    
    FILE *fp;
    Info_t info;
    Data_t data;

    info.max_n = 0;  // initialize

    // get the python arguments
    if (!PyArg_ParseTuple(args, "sll:read_bin", &file, &base, &period)) return NULL;

    fp = fopen(file, "r");
    if (!fp){
        PyErr_SetString(PyExc_IOError, "Error openining file");
        return NULL;
    }

    // read the header for gain/offset information, etc
    read_header(fp, &info, 0);

    // dimensions for return values
    npy_intp dim3[2] = {info.npages * PAGE_SAMPLES, 3},
             dim1[1] = {info.npages * PAGE_SAMPLES};

    // data arrays
    PyArrayObject *accel = (PyArrayObject *)PyArray_EMPTY(2, dim3, NPY_DOUBLE, 0),
                  *time  = (PyArrayObject *)PyArray_EMPTY(1, dim1, NPY_DOUBLE, 0),
                  *light = (PyArrayObject *)PyArray_EMPTY(1, dim1, NPY_DOUBLE, 0),
                  *temp  = (PyArrayObject *)PyArray_EMPTY(1, dim1, NPY_DOUBLE, 0),
                  *index = (PyArrayObject *)PyArray_EMPTY(1, &(info.npages), NPY_LONG, 0);
    
    // set the pointer to the data objects
    data.acc   = (double *)PyArray_DATA(accel);
    data.ts    = (double *)PyArray_DATA(time);
    data.light = (double *)PyArray_DATA(light);
    data.temp  = (double *)PyArray_DATA(temp);
    data.idx   = (long *)PyArray_DATA(index);

    // initialize the index values
    for (int i = 0; i < info.npages; ++i){
        data.idx[i] = -2 * PAGE_SAMPLES * info.npages;
    }

    // read the file
    for (int i = 0; i < info.npages; ++i){
        ierr = read_block(fp, &base, &period, &info, &data);

        if (ierr != -1){
            fail = 1;
            break;
        }
    }

    fclose(fp);

    if (fail){
        Py_XDECREF(accel);
        Py_XDECREF(time);
        Py_XDECREF(temp);
        Py_XDECREF(light);
        Py_XDECREF(index);

        set_error_message(ierr);
        return NULL;
    }

    return Py_BuildValue(
        "lOOOOO",
        info.max_n,
        (PyObject *)accel,
        (PyObject *)time,
        (PyObject *)light,
        (PyObject *)temp,
        (PyObject *)index
    );
}

static const char bin_convert__doc__[] = "bin_convert(file, base, period)\n"
"Read a Geneactiv File\n\n"
"Parameters\n"
"----------\n"
"file : str\n"
"  File name to read from\n"
"base : int\n"
"  Base time for providing windowing. Must be in [0, 23]\n"
"period : int\n"
"  number of hours in each window. Must be in [1, 24]\n";


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
PyMODINIT_FUNC PyInit_bin_convert(void){
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