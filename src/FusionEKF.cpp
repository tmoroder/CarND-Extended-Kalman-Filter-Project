#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  Q_nu_ = MatrixXd(2, 2);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // tools object
  tools = Tools();

  // Internal covariance matrix Q_nu - process
  Q_nu_ << 9, 0,
           0, 9;

  // Extended Kalman Filter
  ekf_ = KalmanFilter();

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */

  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar data from polar to Cartesian coordinates
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      ekf_.x_ << rho * std::cos(phi),
                 rho * std::sin(phi),
                 0,
                 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0),
                 measurement_pack.raw_measurements_(1),
                 0,
                 0;
    }
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }


  /**
   * Prediction
   */

  // Elapsed time in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1E+6;
  previous_timestamp_ = measurement_pack.timestamp_;
  // State transition matrix
  MatrixXd F(4, 4);
  F << 1, 0, dt, 0,
       0, 1, 0, dt,
       0, 0, 1, 0,
       0, 0, 0, 1;
  ekf_.F_ = F;
  // Process noise covariance matrix
  MatrixXd G(4, 2);
  G << dt * dt / 2, 0,
       0, dt * dt / 2,
       dt, 0,
       0, dt;
  ekf_.Q_ = G * Q_nu_ * G.transpose();
  // Predict using linear motion model of KF
  ekf_.Predict();


  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates:
    // - H_ is Jacobian at x
    // - h(x) is computed in ekf_.UpdateEKF
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates is default linear Kalman Filter update
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
