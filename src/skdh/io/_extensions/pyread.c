// Copyright (c) 2021. Pfizer Inc. All rights reserved.
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include "read_binary_imu.h"

#define STR2PY PyUnicode_FromString

#define NP_FROM_ANY(x) PyArray_FromAny(x, PyArray_DescrFromType(NPY_LONG), 1, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO, NULL)


void geneactiv_set_error_message(int ierr)
{
    switch (ierr)
    {
        case GN_READ_E_BLOCK_TIMESTAMP :
            PyErr_SetString(PyExc_RuntimeError, "Error reading timestamp from data block");
            break;
        case GN_READ_E_BLOCK_FS :
            PyErr_SetString(PyExc_RuntimeError, "Block sampling frequency does not match header");
            break;
        case GN_READ_E_BLOCK_DATA :
            PyErr_SetString(PyExc_RuntimeError, "Error reading data from data block");
            break;
        case GN_READ_E_BLOCK_DATA_3600 :
            PyErr_SetString(PyExc_RuntimeError, "Data length is shorter than 3600");
            break;
        default :
            PyErr_SetString(PyExc_RuntimeError, "Unknown error reading GeneActiv file");
    }
}

void axivity_set_error_message(int ierr)
{
    switch(ierr)
    {
//        case AX_READ_E_BAD_HEADER :
//            PyErr_SetString(PyExc_RuntimeError, "Bad packet header value.");
//            break;
        case AX_READ_E_MISMATCH_N_AXES :
            PyErr_SetString(PyExc_RuntimeError, "Incorrect number of axes given file configuration.");
            break;
        case AX_READ_E_INVALID_BLOCK_SAMPLES :
            PyErr_SetString(PyExc_RuntimeError, "Invalid number of samples in data block.");
            break;
        case AX_READ_E_BAD_AXES_PACKED :
            PyErr_SetString(PyExc_RuntimeError, "More than 3 axes for packed data.");
            break;
        case AX_READ_E_BAD_PACKING_CODE :
            PyErr_SetString(PyExc_RuntimeError, "Invalid packing code (not 0 or 2).");
            break;
        case AX_READ_E_BAD_CHECKSUM :
            PyErr_SetString(PyExc_RuntimeError, "Checksum not equal to 0.");
            break;
        case AX_READ_E_BAD_LENGTH_ZERO_TIMESTAMPS :
            PyErr_SetString(PyExc_RuntimeError, "Bad block of timestamps not equal to data block sample size.");
            break;
        default :
            PyErr_SetString(PyExc_RuntimeError, "Unknown error reading Axivity file");
    }
}

static PyObject *read_axivity(PyObject *NPY_UNUSED(self), PyObject *args)
{
    char *file;
    Py_ssize_t flen;
    int ierr = AX_READ_E_NONE, fail = 0;

    AX_Info_t info;

    /* READ INPUT ARGUMENTS */
    if (!PyArg_ParseTuple(args, "s:read_axivity", &file))
        return NULL;
    flen = strlen(file);

    /* INITIALIZATION */
    info.nblocks = -1;
    info.axes = -1;
    info.count = -1;

    /* read the header */
    axivity_read_header(&flen, file, &info, &ierr);

    if (ierr != AX_READ_E_NONE)
    {
        axivity_close(&info);

        axivity_set_error_message(ierr);
        return NULL;
    }

    if ((info.nblocks == -1) || (info.axes == -1) || (info.count == -1))
    {
        axivity_close(&info);
        PyErr_SetString(PyExc_IOError, "Bad read on number of blocks, axes, or samples");
        return NULL;
    }

    /* DIMENSIONS FOR RETURN VALUES */
    npy_intp dim3[2] = {(info.nblocks - 2) * info.count, info.axes};
    npy_intp dim1[1] = {(info.nblocks - 2) * info.count};

    /* DATA ARRAYS */
    PyArrayObject *imudata = (PyArrayObject *)PyArray_ZEROS(2, dim3, NPY_DOUBLE, 0);
    PyArrayObject *time  = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);
    PyArrayObject *temperature = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);

    if (!imudata || !time || !temperature)
    {   
        axivity_close(&info);

        Py_XDECREF(imudata);
        Py_XDECREF(time);
        Py_XDECREF(temperature);

        return NULL;
    }

    /* SET POINTERS */
    double *imu_p   = (double *)PyArray_DATA(imudata);
    double *ts_p    = (double *)PyArray_DATA(time);
    double *temp_p = (double *)PyArray_DATA(temperature);

    /* READ FILE */
    long pos = 0;
    for (int i=2; i < info.nblocks; ++i)
    {
        pos = 512 * i + 1;  /* +1 to account for fortran numbering */
        axivity_read_block(&info, &pos, imu_p, ts_p, temp_p, &ierr);
        
        if (ierr != 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Error reading axivity data block.");
            fail = 1;
            break;
        }
    }

    /* adjust timestamps if there were bad blocks */
    if (info.n_bad_blocks > 0)
    {
        adjust_timestamps(&info, ts_p, &ierr);
        if (ierr != 0)
        {
            fail = 1;
        }
    }

    /* set a warning for the number of bad blocks */
    if (info.n_bad_blocks > 0)
    {
        fprintf(stdout, "WARNING: %li bad blocks\n", info.n_bad_blocks);
        int err_ret = PyErr_WarnEx(PyExc_RuntimeWarning, "Bad data blocks present", 1);

        if (err_ret == -1)  /* warnings are being raised as exceptions */
        {
            fail = 1;
        }
    }

    axivity_close(&info);

    /* decrease ref count if failed */
    
    if (fail)
    {
        Py_XDECREF(imudata);
        Py_XDECREF(time);
        Py_XDECREF(temperature);

        axivity_set_error_message(ierr);
        return NULL;
    }

    return Py_BuildValue(
        "dlNNN",  /* need to use N to not increment reference counter */
        info.frequency,
        info.n_bad_blocks * info.count,
        (PyObject *)imudata,
        (PyObject *)time,
        (PyObject *)temperature
    );
}


static PyObject *read_geneactiv(PyObject *NPY_UNUSED(self), PyObject *args)
{
    char *file;
    int ierr = GN_READ_E_NONE, fail = 0;

    FILE *fp;
    GN_Info_t info;
    GN_Data_t data;

    /* INITIALIZATION */
    info.fs_err = 0;
    info.max_n = 0;
    info.npages = -1;

    /* PYTHON ARGUMENTS */
    if (!PyArg_ParseTuple(args, "s:read_geneactiv", &file))
        return NULL;  /* error is set for us */
    
    /* OPEN THE FILE */
    fp = fopen(file, "r");
    if (!fp)
    {
        PyErr_SetString(PyExc_IOError, "Error opening file");
        return NULL;
    }

    /* READ THE HEADER */
    DEBUG_PRINTF("Reading header\n");
    geneactiv_read_header(fp, &info);

    if (info.npages == -1)
    {
        fclose(fp);
        PyErr_SetString(PyExc_IOError, "Cannot read number of blocks");
        return NULL;
    }

    /* DIMENSIONS FOR RETURN VALUES */
    npy_intp dim3[2] = {info.npages * GN_SAMPLES, 3};
    npy_intp dim1[1] = {info.npages * GN_SAMPLES};

    /* DATA ARRAYS */
    PyArrayObject *accel = (PyArrayObject *)PyArray_ZEROS(2, dim3, NPY_DOUBLE, 0);
    PyArrayObject *time  = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);
    PyArrayObject *light = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);
    PyArrayObject *temp  = (PyArrayObject *)PyArray_ZEROS(1, dim1, NPY_DOUBLE, 0);

    if (!accel || !time || !light || !temp)
    {
        fclose(fp);

        Py_XDECREF(accel);
        Py_XDECREF(time);
        Py_XDECREF(temp);
        Py_XDECREF(light);

        return NULL;
    }

    /* SET POINTERS */
    data.acc   = (double *)PyArray_DATA(accel);
    data.ts    = (double *)PyArray_DATA(time);
    data.light = (double *)PyArray_DATA(light);
    data.temp  = (double *)PyArray_DATA(temp);
    
    /* READ FILE */
    DEBUG_PRINTF("Reading pages\n");
    for (int i = 0; i < info.npages; ++i)
    {
        DEBUG_PRINTF("%i\n", i);
        ierr = geneactiv_read_block(fp, &info, &data);

        /* check output of ierr */
        if (ierr == GN_READ_E_NONE)  /* most common case */
        {}
        else if (ierr == GN_READ_E_BLOCK_FS_WARN)
        {
            int err_ret = PyErr_WarnEx(PyExc_RuntimeWarning, "Block fs is not the same as header fs. Setting to block fs.", 1);

            if (err_ret == -1)  /* warnings are being raised as exceptions */
            {
                fail = 1;
                break;
            }
        }
        else if (ierr == GN_READ_E_BLOCK_MISSING_BLOCK_WARN)
        {
            int err_ret = PyErr_WarnEx(PyExc_RuntimeWarning, "Found an empty block, assuming end of recorded data.", 1);
            /* break out of the loop, but dont fail, unless raising warnings */
            if (err_ret == -1) fail = 1;

            break;
        }
        else
        {
            fail = 1;
            break;
        }
    }

    fclose(fp);

    if (fail)
    {
        Py_XDECREF(accel);
        Py_XDECREF(time);
        Py_XDECREF(temp);
        Py_XDECREF(light);

        geneactiv_set_error_message(ierr);
        return NULL;
    }

    return Py_BuildValue(
        "lfNNNN",  /* need to use N to not increment reference counter */
        (info.max_n + 1) * GN_SAMPLES,
        info.fs,
        (PyObject *)accel,
        (PyObject *)time,
        (PyObject *)light,
        (PyObject *)temp
    );
}


static const char read_axivity__doc__[] = "read_axivity(file)\n"
"Read an Axivity binary file.\n\n"
"Parameters\n"
"----------\n"
"file : str\n"
"   File name to read from\n\n"
"Returns\n"
"-------\n"
"fs : float\n"
"   Sampling frequency\n"
"imudata : numpy.ndarray\n"
"   IMU data available. Shape is (N, 3/6/9). Order of types is [Gy]Ax[Mag]. Ax (accelerometer) is\n"
"   required. Gy (gyroscope) and Mag (magnetometer) are optional.\n"
"time : numpy.ndarray\n"
"temperature : numpy.ndarray\n";

static const char read_geneactiv__doc__[] = "read_geneactiv(file)\n"
"Read a Geneactiv File\n\n"
"Parameters\n"
"----------\n"
"file : str\n"
"   File name to read from\n\n"
"Returns\n"
"-------\n"
"N : int\n"
"   Number of pages read.\n"
"fs : float\n"
"   Sampling frequency\n"
"accel : numpy.ndarray\n"
"time : numpy.ndarray\n"
"light : numpy.ndarray\n"
"temp : numpy.ndarray\n";

static struct PyMethodDef methods[] = {
  {"read_geneactiv", read_geneactiv, 1, read_geneactiv__doc__},
  {"read_axivity", read_axivity, 1, read_axivity__doc__},
  {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "read",
  NULL,
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};


/* Initialization function for the module */
PyMODINIT_FUNC PyInit_read(void){
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
