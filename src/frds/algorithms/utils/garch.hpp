#ifndef FRDS_ALGO_UTILS_GARCH
#define FRDS_ALGO_UTILS_GARCH

#include <cmath>
#include <vector>

void ewma(const double resids[], double variance[], int T, double initial_value,
          double lam = 0.94) {
  // Compute the conditional variance estimates using Exponentially Weighted
  // Moving Average (EWMA).

  variance[0] = initial_value; // Set the initial value

  // Compute the EWMA using the decay factors and squared residuals
  for (int t = 1; t < T; ++t) {
    double squared_resid = resids[t - 1] * resids[t - 1];
    variance[t] = lam * variance[t - 1] + (1 - lam) * squared_resid;
  }
}

inline double bounds_check(double sigma2, double lower, double upper) {
  sigma2 = lower > sigma2 ? lower : sigma2;
  if (sigma2 > upper) {
    if (!std::isinf(sigma2)) {
      sigma2 = upper + std::log(sigma2 / upper);
    } else {
      sigma2 = upper + 1000;
    }
  }
  return sigma2;
}

void compute_garch_variance(double *sigma2, double *resids, double *var_bounds,
                            int T, std::vector<double> params,
                            double backcast) {
  double omega = params[0];
  double alpha = params[1];
  double beta = params[2];

  sigma2[0] = omega + (alpha + beta) * backcast;

  for (int t = 1; t < T; ++t) {
    sigma2[t] =
        omega + alpha * std::pow(resids[t - 1], 2) + beta * sigma2[t - 1];
    sigma2[t] =
        bounds_check(sigma2[t], var_bounds[t * 2], var_bounds[t * 2 + 1]);
  }
}

void compute_gjrgarch_variance(double *sigma2, double *resids,
                               double *var_bounds, int T,
                               std::vector<double> params, double backcast) {
  double omega = params[0];
  double alpha = params[1];
  double gamma = params[2];
  double beta = params[3];

  sigma2[0] = omega + (alpha + gamma / 2 + beta) * backcast;

  for (int t = 1; t < T; ++t) {
    sigma2[t] =
        omega + alpha * std::pow(resids[t - 1], 2) + beta * sigma2[t - 1];
    if (resids[t - 1] < 0) {
      sigma2[t] += gamma * std::pow(resids[t - 1], 2);
    }
    sigma2[t] =
        bounds_check(sigma2[t], var_bounds[t * 2], var_bounds[t * 2 + 1]);
  }
}

#endif // FRDS_ALGO_UTILS_GARCH
