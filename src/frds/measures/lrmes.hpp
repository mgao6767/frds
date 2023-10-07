#ifndef FRDS_MEASURES_LRMES
#define FRDS_MEASURES_LRMES

#include <cmath>
#include <numpy/arrayobject.h>
#include <tuple>
#include <vector>

std::tuple<double, bool>
simulation(PyArrayObject *innovation, double C, double firm_var, double mkt_var,
           double firm_resid, double mkt_resid, double a, double b, double rho,
           PyArrayObject *Q_bar, double mu_i, double omega_i, double alpha_i,
           double gamma_i, double beta_i, double mu_m, double omega_m,
           double alpha_m, double gamma_m, double beta_m) {
  // Extract Q_bar values
  double q_i_bar = *((double *)PyArray_GETPTR2(Q_bar, 0, 0));
  double q_m_bar = *((double *)PyArray_GETPTR2(Q_bar, 1, 1));
  double q_im_bar = *((double *)PyArray_GETPTR2(Q_bar, 1, 0));

  // Initialize variables
  double q_i = 1.0, q_m = 1.0, q_im = rho;

  // Initialize vectors to hold firm and market returns
  std::vector<double> firm_return(PyArray_DIM(innovation, 0));
  std::vector<double> mkt_return(PyArray_DIM(innovation, 0));

  double epsilon_i, epsilon_m;
  bool systemic_event = true;
  double firmret = 0.0;

  // lagged residuals
  double resid_i_tm1 = firm_resid;
  double resid_m_tm1 = mkt_resid;
  // lagged conditonal variance
  double firm_var_tm1 = firm_var;
  double mkt_var_tm1 = mkt_var;
  // lagged standarized residuals
  double firm_innov_tm1 = resid_i_tm1 / std::sqrt(firm_var_tm1);
  double mkt_innov_tm1 = resid_m_tm1 / std::sqrt(mkt_var_tm1);

  // Main loop
  for (int h = 0; h < PyArray_DIM(innovation, 0); ++h) {

    double firm_var_t =
        omega_i + alpha_i * std::pow(resid_i_tm1, 2) + beta_i * firm_var_tm1;
    if (resid_i_tm1 < 0) {
      firm_var_t += gamma_i * std::pow(resid_i_tm1, 2);
    }

    double mkt_var_t =
        omega_m + alpha_m * std::pow(resid_m_tm1, 2) + beta_m * mkt_var_tm1;
    if (resid_m_tm1 < 0) {
      mkt_var_t += gamma_m * std::pow(resid_m_tm1, 2);
    }

    q_i = (1 - a - b) * q_i_bar + a * firm_innov_tm1 * firm_innov_tm1 + b * q_i;
    q_m = (1 - a - b) * q_m_bar + a * mkt_innov_tm1 * mkt_innov_tm1 + b * q_m;
    q_im =
        (1 - a - b) * q_im_bar + a * firm_innov_tm1 * mkt_innov_tm1 + b * q_im;
    double rho_h = q_im / std::sqrt(q_i * q_m);

    double firm_innov = *((double *)PyArray_GETPTR2(innovation, h, 0));
    double mkt_innov = *((double *)PyArray_GETPTR2(innovation, h, 1));

    epsilon_m = std::sqrt(mkt_var_t) * mkt_innov;
    epsilon_i =
        std::sqrt(firm_var_t) *
        (rho_h * mkt_innov + std::sqrt(1 - std::pow(rho_h, 2)) * firm_innov);

    mkt_return[h] = mu_m + epsilon_m;
    firm_return[h] = mu_i + epsilon_i;

    firm_var_tm1 = firm_var_t;
    mkt_var_tm1 = mkt_var_t;
    resid_i_tm1 = epsilon_i;
    resid_m_tm1 = epsilon_m;
    firm_innov_tm1 = resid_i_tm1 / std::sqrt(firm_var_tm1);
    mkt_innov_tm1 = mkt_innov;
  }

  // Convert back to original scale
  for (double &ret : mkt_return) {
    ret /= 100;
  }
  for (double &ret : firm_return) {
    ret /= 100;
  }

  // Check for systemic event
  double mkt_sum = 0.0;
  for (const double &ret : mkt_return) {
    mkt_sum += ret;
  }
  systemic_event = std::exp(mkt_sum) - 1 < C;

  if (!systemic_event) {
    return std::make_tuple(0.0, false);
  }

  // Calculate firm return over the horizon
  double firm_sum = 0.0;
  for (const double &ret : firm_return) {
    firm_sum += ret;
  }
  firmret = std::exp(firm_sum) - 1;

  return std::make_tuple(firmret, true);
}

#endif // FRDS_MEASURES_LRMES