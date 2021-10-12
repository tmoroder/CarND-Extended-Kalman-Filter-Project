#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Variable creation
  VectorXd rmse(4);
  rmse << -1, -1, -1, -1;

  // Input validation
  if (estimations.size() != ground_truth.size()) {
    std::cout << "ERROR - CalculateRMSE() - Unequal sizes." << std::endl;
    return rmse;
  }
  if (estimations.size() == 0) {
    std::cout << "ERROR - CalculateRMSE() - No elements." << std::endl;
    return rmse;
  }

  // Compute RMSE
  rmse << 0, 0, 0, 0;
  for (unsigned int i=0; i < estimations.size(); ++i) {
    VectorXd r = estimations[i] - ground_truth[i];
    r = r.array() * r.array();
    rmse += r;
  }
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Variable creation
  MatrixXd J(3, 4);
  J << 0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0;
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Compute intermediate values
  float c2 = px * px + py * py;
  float c1 = std::sqrt(c2);
  float c3 = c1 * c2;

  // Input validation
  if (std::fabs(c2) < 0.00001) {
    std::cout << "ERROR - CalculateJacobian()- Division by a too small number." << std::endl;
    return J;
  }

  // Set Jacobian elementwise
  J(0, 0) = px / c1;
  J(0, 1) = py / c1;
  J(1, 0) = -py / c2;
  J(1, 1) = px / c2;
  J(2, 0) = py * (vx * py - vy * px) / c3;
  J(2, 1) = px * (vy * px - vx * py) / c3;
  J(2, 2) = px / c1;
  J(2, 3) = py / c1;

  return J;
}
