#ifndef FRDS_ALGO_UTILS_MGARCH
#define FRDS_ALGO_UTILS_MGARCH

#include <cmath>
#include <numpy/arrayobject.h>
#include <vector>

namespace DCC {

std::vector<double> conditional_correlations(double a, double b,
                                             PyArrayObject *resids1,
                                             PyArrayObject *resids2,
                                             PyArrayObject *sigma2_1,
                                             PyArrayObject *sigma2_2) {
  int T = PyArray_DIM(resids1, 0);
  std::vector<double> rho(T, 0.0);

  // Compute z1 and z2 (standardized residuals)
  std::vector<double> z1(T);
  std::vector<double> z2(T);
  for (int t = 0; t < T; ++t) {
    z1[t] = *((double *)PyArray_GETPTR1(resids1, t)) /
            std::sqrt(*((double *)PyArray_GETPTR1(sigma2_1, t)));
    z2[t] = *((double *)PyArray_GETPTR1(resids2, t)) /
            std::sqrt(*((double *)PyArray_GETPTR1(sigma2_2, t)));
  }

  // Compute Q_bar (correlation matrix)
  // Since z1 z2 have zero mean, covariance matrix is same as correlation matrix
  double q_11_bar = 0.0, q_12_bar = 0.0, q_22_bar = 0.0;
  for (int t = 0; t < T; ++t) {
    q_11_bar += z1[t] * z1[t];
    q_12_bar += z1[t] * z2[t];
    q_22_bar += z2[t] * z2[t];
  }
  q_11_bar /= T;
  q_12_bar /= T;
  q_22_bar /= T;

  // Initialize q11, q12, q22
  std::vector<double> q11(T);
  std::vector<double> q12(T);
  std::vector<double> q22(T);
  q11[0] = q_11_bar;
  q22[0] = q_22_bar;
  q12[0] = q_12_bar;
  rho[0] = q12[0] / std::sqrt(q11[0] * q22[0]);

  // Main loop
  for (int t = 1; t < T; ++t) {
    q11[t] =
        (1 - a - b) * q_11_bar + a * std::pow(z1[t - 1], 2) + b * q11[t - 1];
    q22[t] =
        (1 - a - b) * q_22_bar + a * std::pow(z2[t - 1], 2) + b * q22[t - 1];
    q12[t] =
        (1 - a - b) * q_12_bar + a * z1[t - 1] * z2[t - 1] + b * q12[t - 1];
    rho[t] = q12[t] / std::sqrt(q11[t] * q22[t]);
  }

  return rho;
}

} // namespace DCC
#endif // FRDS_ALGO_UTILS_MGARCH