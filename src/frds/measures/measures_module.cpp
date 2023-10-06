#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifndef UTILS_MODULE
#define UTILS_MODULE
#endif
#include "lrmes.hpp"

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include "numpy/arrayobject.h"

// Forward declaration of the LRMES simulation function
std::tuple<double, bool>
simulation(PyArrayObject *innovation, double C, double firm_var, double mkt_var,
           double firm_resid, double mkt_resid, double a, double b, double rho,
           PyArrayObject *Q_bar, double mu_i, double omega_i, double alpha_i,
           double gamma_i, double beta_i, double mu_m, double omega_m,
           double alpha_m, double gamma_m, double beta_m);

static PyObject *lrmes_simulation_wrapper(PyObject *self, PyObject *args) {
  PyArrayObject *innovation;
  double C, firm_var, mkt_var, firm_resid, mkt_resid, a, b, rho;
  PyArrayObject *Q_bar;
  double mu_i, omega_i, alpha_i, gamma_i, beta_i;
  double mu_m, omega_m, alpha_m, gamma_m, beta_m;

  if (!PyArg_ParseTuple(args, "OddddddddOdddddddddd", &innovation, &C,
                        &firm_var, &mkt_var, &firm_resid, &mkt_resid, &a, &b,
                        &rho, &Q_bar, &mu_i, &omega_i, &alpha_i, &gamma_i,
                        &beta_i, &mu_m, &omega_m, &alpha_m, &gamma_m,
                        &beta_m)) {
    return NULL;
  }

  double firmret;
  bool systemic_event;
  std::tie(firmret, systemic_event) =
      simulation(innovation, C, firm_var, mkt_var, firm_resid, mkt_resid, a, b,
                 rho, Q_bar, mu_i, omega_i, alpha_i, gamma_i, beta_i, mu_m,
                 omega_m, alpha_m, gamma_m, beta_m);

  return Py_BuildValue("db", firmret, systemic_event);
}

// Method table
static PyMethodDef methods[] = {{"simulation", lrmes_simulation_wrapper,
                                 METH_VARARGS, "Run the LRMES simulation"},
                                {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "measures", NULL, -1,
                                    methods};

PyMODINIT_FUNC PyInit_measures_ext(void) {
  import_array();
  return PyModule_Create(&module);
}
