#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // New state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Intermediate variables
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // New state
  x_ = x_ + K * y;
  P_ = P_ - K * H_ * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Compute h(x)
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float rho = std::sqrt(px * px + py * py);
  VectorXd hx(3);
  hx << rho,
        std::atan2(py, px),
        (px * vx + py * vy) / rho;

  // Intermediate variables
  VectorXd y = z - hx;
  // angle difference y(1) should be in interval [-pi, pi)
  y(1) =  std::fmod(y(1) + M_PI, 2 * M_PI) - M_PI;;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // New state
  x_ = x_ + K * y;
  P_ = P_ - K * H_ * P_;
}
