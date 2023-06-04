#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifndef DCC_MODULE
#define DCC_MODUEL
#endif
#include "dcc.hpp"

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include "numpy/arrayobject.h"

static PyObject *dcc_wrapper(PyObject *self, PyObject *args) {
  PyObject *firm_data = NULL, *mkt_data = NULL;
  PyArrayObject *firm_arr = NULL, *mkt_arr = NULL, *outArr = NULL;
  if (!PyArg_ParseTuple(args, "OO", &firm_data, &mkt_data))
    return NULL;

  firm_arr = (PyArrayObject *)PyArray_FROM_OTF(firm_data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (firm_arr == NULL) return NULL;
  mkt_arr = (PyArrayObject *)PyArray_FROM_OTF(mkt_data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (mkt_arr == NULL) return NULL;

  uint T = PyArray_DIM(firm_arr, 1);
  
  double firm[T], mkt[T];
  for (size_t i = 0; i < T; i++)
  {
    firm[i] = *(double *)PyArray_GETPTR1(firm_arr, i);
    mkt[i] = *(double *)PyArray_GETPTR1(mkt_arr, i);
  }
  
  std::tuple<double, double> ab = dcc(firm, mkt, T);

  Py_DECREF(firm_arr);
  Py_DECREF(mkt_arr);

  return PyTuple_Pack(2, PyFloat_FromDouble(std::get<0>(ab)), PyFloat_FromDouble(std::get<1>(ab)));
};


static PyObject *loss_func_wrapper(PyObject *self, PyObject *args) {
  PyObject *firm_data = NULL, *mkt_data = NULL;
  double a, b;
  PyArrayObject *firm_arr = NULL, *mkt_arr = NULL, *outArr = NULL;
  if (!PyArg_ParseTuple(args, "OOdd", &firm_data, &mkt_data, &a, &b))
    return NULL;

  firm_arr = (PyArrayObject *)PyArray_FROM_OTF(firm_data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (firm_arr == NULL) return NULL;
  mkt_arr = (PyArrayObject *)PyArray_FROM_OTF(mkt_data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (mkt_arr == NULL) return NULL;

  uint T = PyArray_DIM(firm_arr, 1);
  
  double firm[T], mkt[T];
  for (size_t i = 0; i < T; i++)
  {
    firm[i] = *(double *)PyArray_GETPTR1(firm_arr, i);
    mkt[i] = *(double *)PyArray_GETPTR1(mkt_arr, i);
  }
  
  double loss = loss_func(firm, mkt, T, a, b);

  Py_DECREF(firm_arr);
  Py_DECREF(mkt_arr);

  return PyFloat_FromDouble(loss);
};



static PyMethodDef dcc_methods[] = {
    {
        "dcc",          // method name
        dcc_wrapper,    // wrapper function
        METH_VARARGS,       // varargs flag
        "Dynamic Conditional Correlation"  // docstring
    },
    {
        "loss_func",          // method name
        loss_func_wrapper,    // wrapper function
        METH_VARARGS,       // varargs flag
        "Loss function"  // docstring
    },
    {NULL, NULL, 0, NULL}};
    

// module definition structure for python3
static struct PyModuleDef dccModule = {PyModuleDef_HEAD_INIT, "dcc",
                                           "DCC estimation", -1,
                                           dcc_methods};

// module initializer for python3
PyMODINIT_FUNC PyInit_dcc_ext() {
  import_array();
  return PyModule_Create(&dccModule);
}

