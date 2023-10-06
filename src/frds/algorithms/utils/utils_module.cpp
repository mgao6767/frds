
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifndef UTILS_MODULE
#define UTILS_MODULE
#endif
#include "garch.hpp"
#include "mgarch.hpp"

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include "numpy/arrayobject.h"

// The wrapper function to be called from Python
static PyObject *ewma_wrapper(PyObject *self, PyObject *args) {
  PyArrayObject *resids_in;
  double initial_value, lam;

  // Parse the input tuple
  if (!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &resids_in, &initial_value,
                        &lam)) {
    return NULL;
  }

  // Construct the output array
  int T = PyArray_DIM(resids_in, 0);
  npy_intp dims[1] = {T};
  PyObject *variance_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (variance_out == NULL) {
    return NULL;
  }

  // Call the core EWMA function
  ewma((double *)PyArray_DATA(resids_in),
       (double *)PyArray_DATA((PyArrayObject *)variance_out), T, initial_value,
       lam);

  return variance_out;
}

// The wrapper function to be called from Python
static PyObject *bounds_check_wrapper(PyObject *self, PyObject *args) {
  double sigma2;
  PyArrayObject *var_bounds_in;

  // Parse the input tuple
  if (!PyArg_ParseTuple(args, "dO!", &sigma2, &PyArray_Type, &var_bounds_in)) {
    return NULL;
  }

  // Extract lower and upper bounds
  double *bounds_ptr = (double *)PyArray_DATA(var_bounds_in);
  double lower = bounds_ptr[0];
  double upper = bounds_ptr[1];

  // Call the core bounds_check function
  double result = bounds_check(sigma2, lower, upper);

  return PyFloat_FromDouble(result);
}

// The wrapper function to be called from Python
static PyObject *compute_garch_variance_wrapper(PyObject *self,
                                                PyObject *args) {
  PyObject *params_obj;
  PyArrayObject *resids_in;
  double backcast;
  PyArrayObject *var_bounds_in;

  // Parse the input tuple
  if (!PyArg_ParseTuple(args, "OO!dO!", &params_obj, &PyArray_Type, &resids_in,
                        &backcast, &PyArray_Type, &var_bounds_in)) {
    return NULL;
  }

  // Check dimensions
  if (PyArray_NDIM(resids_in) != 1 || PyArray_NDIM(var_bounds_in) != 2) {
    PyErr_SetString(PyExc_ValueError, "Invalid array dimensions");
    return NULL;
  }

  // Extract parameters
  std::vector<double> params;
  PyObject *iter = PyObject_GetIter(params_obj);
  PyObject *item;
  while ((item = PyIter_Next(iter))) {
    params.push_back(PyFloat_AsDouble(item));
    Py_DECREF(item);
  }
  Py_DECREF(iter);

  // Construct the output array
  int T = PyArray_DIM(resids_in, 0);
  npy_intp dims[1] = {T};
  PyObject *sigma2_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (sigma2_out == NULL) {
    return NULL;
  }

  // Call the core compute_garch_variance function
  compute_garch_variance((double *)PyArray_DATA((PyArrayObject *)sigma2_out),
                         (double *)PyArray_DATA(resids_in),
                         (double *)PyArray_DATA(var_bounds_in), T, params,
                         backcast);

  return sigma2_out;
}

// The wrapper function to be called from Python
static PyObject *compute_gjrgarch_variance_wrapper(PyObject *self,
                                                   PyObject *args) {
  PyObject *params_obj;
  PyArrayObject *resids_in;
  double backcast;
  PyArrayObject *var_bounds_in;

  // Parse the input tuple
  if (!PyArg_ParseTuple(args, "OO!dO!", &params_obj, &PyArray_Type, &resids_in,
                        &backcast, &PyArray_Type, &var_bounds_in)) {
    return NULL;
  }

  // Check dimensions
  if (PyArray_NDIM(resids_in) != 1 || PyArray_NDIM(var_bounds_in) != 2) {
    PyErr_SetString(PyExc_ValueError, "Invalid array dimensions");
    return NULL;
  }

  // Extract parameters
  std::vector<double> params;
  PyObject *iter = PyObject_GetIter(params_obj);
  PyObject *item;
  while ((item = PyIter_Next(iter))) {
    params.push_back(PyFloat_AsDouble(item));
    Py_DECREF(item);
  }
  Py_DECREF(iter);

  // Construct the output array
  int T = PyArray_DIM(resids_in, 0);
  npy_intp dims[1] = {T};
  PyObject *sigma2_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (sigma2_out == NULL) {
    return NULL;
  }

  // Call the core compute_gjrgarch_variance function
  compute_gjrgarch_variance((double *)PyArray_DATA((PyArrayObject *)sigma2_out),
                            (double *)PyArray_DATA(resids_in),
                            (double *)PyArray_DATA(var_bounds_in), T, params,
                            backcast);

  return sigma2_out;
}

static PyObject *dcc_conditional_correlations_wrapper(PyObject *self,
                                                      PyObject *args) {
  double a, b;
  PyArrayObject *resids1;
  PyArrayObject *resids2;
  PyArrayObject *sigma2_1;
  PyArrayObject *sigma2_2;

  if (!PyArg_ParseTuple(args, "ddO!O!O!O!", &a, &b, &PyArray_Type, &resids1,
                        &PyArray_Type, &resids2, &PyArray_Type, &sigma2_1,
                        &PyArray_Type, &sigma2_2)) {
    return NULL;
  }

  std::vector<double> rho =
      DCC::conditional_correlations(a, b, resids1, resids2, sigma2_1, sigma2_2);

  npy_intp dims[1] = {static_cast<npy_intp>(rho.size())};
  PyArrayObject *py_rho =
      (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  std::copy(rho.begin(), rho.end(), (double *)PyArray_DATA(py_rho));

  return (PyObject *)py_rho;
}

// Method table
static PyMethodDef methods[] = {
    {"ewma", ewma_wrapper, METH_VARARGS, "Calculate EWMA"},

    {"bounds_check", bounds_check_wrapper, METH_VARARGS,
     "Adjust the conditional variance based on its bounds"},

    {"compute_garch_variance", compute_garch_variance_wrapper, METH_VARARGS,
     "Computes the variances conditional on given parameters"},

    {"compute_gjrgarch_variance", compute_gjrgarch_variance_wrapper,
     METH_VARARGS, "Computes the variances conditional on given parameters"},

    {"dcc_conditional_correlation", dcc_conditional_correlations_wrapper,
     METH_VARARGS,
     "Computes the DCC conditional correlation based on given parameters"},

    {NULL, NULL, 0, NULL}};

// module definition structure for python3
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "utils", "utils", -1,
                                    methods};

// module initializer for python3
PyMODINIT_FUNC PyInit_utils_ext() {
  import_array();
  return PyModule_Create(&module);
}
