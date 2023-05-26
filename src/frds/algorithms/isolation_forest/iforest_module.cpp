#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifndef IFOREST_MODULE
#define IFOREST_MODULE
#endif
#include "IsolationForest.cpp"

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include "numpy/arrayobject.h"

static PyObject *iforest_wrapper(PyObject *self, PyObject *args) {
  PyObject *num_data = NULL, *char_data = NULL, *out = NULL;
  PyArrayObject *num_arr = NULL, *char_arr = NULL, *outArr = NULL;
  int forestSize, treeSize, randomSeed;
  if (!PyArg_ParseTuple(args, "OOOiii", &num_data, &char_data, &out,
                        &forestSize, &treeSize, &randomSeed))
    return NULL;

  num_arr = (PyArrayObject *)PyArray_FROM_OTF(num_data, NPY_DOUBLE,
                                              NPY_ARRAY_IN_ARRAY);
  if (num_arr == NULL) return NULL;
  char_arr = (PyArrayObject *)PyArray_FROM_OTF(char_data, NPY_STRING,
                                               NPY_ARRAY_IN_ARRAY);
  if (char_arr == NULL) return NULL;

#if NPY_API_VERSION >= 0x0000000c
  outArr = (PyArrayObject *)PyArray_FROM_OTF(out, NPY_DOUBLE,
                                             NPY_ARRAY_INOUT_ARRAY2);
#else
  outArr =
      (PyArrayObject *)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
  if (outArr == NULL) {
    Py_XDECREF(num_arr);
    Py_XDECREF(char_arr);
    Py_XDECREF(outArr);
    return NULL;
  }

  // number of obervations
  auto nObs = PyArray_DIM(num_arr, 1);

  auto iforest =
      IsolationForest(num_arr, char_arr, treeSize, forestSize, randomSeed);

  iforest.grow();
  iforest.calculateAnomalyScores();

  // write the ascores to the output array
  for (npy_intp i = 0; i < nObs; i++) {
    PyArray_SETITEM(outArr, (char *)PyArray_GETPTR1(outArr, i),
                    PyFloat_FromDouble(iforest.anomalyScores[i]));
  }

  Py_DECREF(num_arr);
  Py_DECREF(char_arr);
  Py_DECREF(outArr);

  Py_INCREF(Py_None);
  return Py_None;
};

static PyMethodDef iforest_methods[] = {
    {
        "iforest",          // method name
        iforest_wrapper,    // wrapper function
        METH_VARARGS,       // varargs flag
        "Isolation Forest"  // docstring
    },
    {NULL, NULL, 0, NULL}};

// module definition structure for python3
static struct PyModuleDef iforestModule = {PyModuleDef_HEAD_INIT, "iforest",
                                           "Isolation Forest algorithm", -1,
                                           iforest_methods};

// module initializer for python3
PyMODINIT_FUNC PyInit_iforest_ext() {
  import_array();
  return PyModule_Create(&iforestModule);
}