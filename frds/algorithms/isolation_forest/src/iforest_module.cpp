#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifndef IFOREST_MODULE
#define IFOREST_MODULE
#endif
#include "iforest.cpp"
#include "numpy/arrayobject.h"

static PyObject *iforest_wrapper(PyObject *self, PyObject *args)
{
    PyObject *in = NULL, *out = NULL;
    PyObject *arr = NULL, *outArr = NULL;
    int forestSize, treeSize, randomSeed;
    if (!PyArg_ParseTuple(args, "OOiii", &in, &out, &forestSize, &treeSize,
                          &randomSeed))
        return NULL;

    arr = PyArray_FROM_OTF(in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL)
        return NULL;

#if NPY_API_VERSION >= 0x0000000c
    outArr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    outArr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (outArr == NULL)
    {
        Py_XDECREF(arr);
        Py_XDECREF(outArr);
        return NULL;
    }

    // number of attributes
    auto nAttrs = PyArray_DIM(arr, 0);
    // number of obervations
    auto nObs = PyArray_DIM(arr, 1);
    // populate data
    Data data;
    for (int p = 0; p < nAttrs; p++)
    {
        std::vector<DataType> obVals;
        for (int i = 0; i < nObs; i++)
            obVals.push_back(*(DataType *)PyArray_GETPTR2(arr, p, i));
        data.push_back(obVals);
    }

    // compute anomaly scores
    auto ascores = calAnomalyScore(data, forestSize, treeSize, randomSeed);

    // write the ascores to the output array
    for (size_t i = 0; i < ascores.size(); i++)
        PyArray_SETITEM(outArr, PyArray_GETPTR1(outArr, i),
                        PyFloat_FromDouble(ascores[i]));

    Py_DECREF(arr);
    Py_DECREF(outArr);

    Py_INCREF(Py_None);
    return Py_None;
};

static PyMethodDef iforest_methods[] = {
    {
        "iforest",         // method name
        iforest_wrapper,   // wrapper function
        METH_VARARGS,      // varargs flag
        "Isolation Forest" // docstring
    },
    {NULL, NULL, 0, NULL}};

// module definition structure for python3
static struct PyModuleDef iforestModule = {PyModuleDef_HEAD_INIT, "iforest",
                                           "Isolation Forest algorithm", -1,
                                           iforest_methods};

// module initializer for python3
PyMODINIT_FUNC PyInit_iforest_ext()
{
    import_array();
    return PyModule_Create(&iforestModule);
}
