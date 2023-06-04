#ifndef FRDS_ALGO_DCC_H
#define FRDS_ALGO_DCC_H

#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <iostream>
#include <functional>
#include <algorithm>

struct Point {
    double x;
    double y;
};

inline void calc_Q_avg(double Q_avg[], double firm[], double mkt[], uint T) {
  for (int i = 0; i < 4; i++)
    Q_avg[i] = 0;
  for (size_t i = 0; i < T; i++) {
    auto fret = firm[i];
    auto mret = mkt[i];
    Q_avg[0] += fret*fret; Q_avg[1] += fret*mret;
    Q_avg[2] += mret*fret; Q_avg[3] += mret*mret;
  }
  for (int i = 0; i < 4; i++)
    Q_avg[i] /= T;
}

inline void calc_Q(double Qs[], double firm[], double mkt[], uint T, double a, double b) {
  double Q_avg[4];
  calc_Q_avg(Q_avg, firm, mkt, T);
  for (int i = 0; i < 4; i++)
    Qs[i] = Q_avg[i];
  
  double omega[4];
  for (int i = 0; i < 4; i++)
    omega[i] = (1.0 - a - b) * Q_avg[i];

  for (uint i = 1; i < T; i++) {
    double Qt_1[4] = {Qs[i*4], Qs[i*4+1], Qs[i+4+2], Qs[i*4+3]};
    double et_1_outer[4] = {firm[i]*firm[i], firm[i]*mkt[i], mkt[i]*firm[i], mkt[i]*mkt[i]};
    for (int j = 0; j < 4; j++)
      Qs[(i+0)*4+j] = omega[j] + a * et_1_outer[j] + b * Qt_1[j];
  }
} 

inline void calc_R(double Rs[], double firm[], double mkt[], uint T, double a, double b) {
  double Q[4 * (T+0)];
  calc_Q(Q, firm, mkt, T, a, b);
  for (uint i = 0; i < T; i++)
  { 
    double q00 = Q[i*4],   q01 = Q[i*4+1]; 
    double q10 = Q[i*4+2], q11 = Q[i*4+3]; 
    
    double t00 = 1.0 / sqrt(abs(q00)),
           t11 = 1.0 / sqrt(abs(q11)); 

    // np.dot(np.dot(tmp, q), tmp)
    double r00 = t00*q00*t00, r01 = t00*q01*t11;
    double r10 = t11*q10*t00, r11 = t11*q11*t11;
    
    if (abs(r01) >= 1.) {
      r01 = 0.9999 * (r01 > 0 ? 1 : -1);
      r10 = r01;
    } 

    Rs[i*4]   = r00; Rs[i*4+1] = r01;
    Rs[i*4+2] = r10; Rs[i*4+3] = r11;
  }
  
}

double loss_func(double firm[], double mkt[], uint T, double a, double b) {
  // if (a<0 | b<0 | a>1 | b>1 | a+b>1) return std::numeric_limits<double>::max();
  double R[4 * (T+0)]; // 2*2 matrix * T
  calc_R(R, firm, mkt, T, a, b);
  double loss = 0.0;
  for (uint i = 0; i < T; i++)
  { 
    // Rt
    double r00 = R[i*4],   r01 = R[i*4+1]; 
    double r10 = R[i*4+2], r11 = R[i*4+3]; 
    // determinant of Rt
    double det = r00*r11 - r01*r10;
    if (det <= 0) return std::numeric_limits<double>::max();
    // inverse of Rt
    double inv00 =  r11 / det, inv01 = -r01 / det; 
    double inv10 = -r10 / det, inv11 =  r00 / det;
    // et
    double e00 = firm[i],   e01 = mkt[i]; 
    loss += log(det) + (e00*inv00+e01*inv10)*e00 + (e00*inv01+e01*inv11)*e01;
  }
  std::cout << "loss " << loss <<std::endl;
  return loss;
}


Point NelderMeadMinimize(const std::function<double(double, double)>& func, Point initialPoint, double tolerance = 1e-6f, int maxIterations = 1000) {
    const double alpha = 1.0f;
    const double beta = 0.2f;
    const double gamma = 1.1f;

    std::vector<Point> simplex{
        {initialPoint.x, initialPoint.y},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
    };

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Sort the simplex by function values
        std::sort(simplex.begin(), simplex.end(), [&](const Point& p1, const Point& p2) {
            return func(p1.x, p1.y) < func(p2.x, p2.y);
        });

        // Calculate the centroid of the best (n-1) points
        Point centroid{0.0f, 0.0f};
        for (int i = 0; i < simplex.size() - 1; ++i) {
            centroid.x += simplex[i].x;
            centroid.y += simplex[i].y;
        }
        centroid.x /= (simplex.size() - 1);
        centroid.y /= (simplex.size() - 1);

        // Reflection
        Point reflectedPoint{
            centroid.x + alpha * (centroid.x - simplex.back().x),
            centroid.y + alpha * (centroid.y - simplex.back().y)
        };

        if (func(reflectedPoint.x, reflectedPoint.y) < func(simplex[0].x, simplex[0].y)) {
            // Expansion
            Point expandedPoint{
                centroid.x + gamma * (reflectedPoint.x - centroid.x),
                centroid.y + gamma * (reflectedPoint.y - centroid.y)
            };

            if (func(expandedPoint.x, expandedPoint.y) < func(reflectedPoint.x, reflectedPoint.y)) {
                simplex.back() = expandedPoint;
            } else {
                simplex.back() = reflectedPoint;
            }
        } else if (func(reflectedPoint.x, reflectedPoint.y) < func(simplex[simplex.size() - 2].x, simplex[simplex.size() - 2].y)) {
            simplex.back() = reflectedPoint;
        } else {
            // Contraction
            Point contractedPoint{
                centroid.x + beta * (simplex.back().x - centroid.x),
                centroid.y + beta * (simplex.back().y - centroid.y)
            };

            if (func(contractedPoint.x, contractedPoint.y) < func(simplex.back().x, simplex.back().y)) {
                simplex.back() = contractedPoint;
            } else {
                // Shrink
                for (int i = 1; i < simplex.size(); ++i) {
                    simplex[i].x = simplex[0].x + 0.5f * (simplex[i].x - simplex[0].x);
                    simplex[i].y = simplex[0].y + 0.5f * (simplex[i].y - simplex[0].y);
                }
            }
        }

        // Check for convergence
        const double maxError = std::max(std::abs(func(simplex[0].x, simplex[0].y) - func(simplex.back().x, simplex.back().y)), std::max(std::abs(simplex[0].x - simplex.back().x), std::abs(simplex[0].y - simplex.back().y)));
        if (maxError < tolerance)
            break;
    }

    return simplex[0];
}


std::tuple<double, double> dcc(double firm[], double mkt[], uint T) {
  double a = 0.5, b = 0.5;
  Point initialPoint{a, b}; // Initial guess for x and y

  auto func = [&](double a, double b) -> double {
    return loss_func(firm, mkt, T, a, b);
  };

  Point minimum = NelderMeadMinimize(func, initialPoint, .00000001f, 1000);

  // std::cout << "Optimized parameters: x = " << minimum.x << ", y = " << minimum.y << std::endl;
  // std::cout << "Minimum function value: " << loss_func(minimum.x, minimum.y) << std::endl;

  return std::make_tuple(minimum.x, minimum.y);
}

#endif  // FRDS_ALGO_DCC_H