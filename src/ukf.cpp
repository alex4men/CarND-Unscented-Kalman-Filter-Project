#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  time_us_ = 0;
  
  // set state dimension
  n_x_ = 5;
  
  // set augmented dimension
  n_aug_ = n_x_ + 2;
  
  // set number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // define spreading parameter
  lambda_ = 3 - n_aug_;
  
  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;
  
  // Process covariance matrix Q
  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  // initial vector for weights
  weights_ = VectorXd(n_sig_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    /**
     * Initialize the state x_ with the first measurement.
     * Create the covariance matrix.
     * Convert radar from polar to cartesian coordinates.
     */
    
    // first measurement
    cout << "EKF: " << meas_package.sensor_type_ << endl;
    cout << meas_package.raw_measurements_ << endl;
    
    if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_) {
      // Convert radar from polar to cartesian coordinates
      //         and initialize state.
      float x = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      float y = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
      x_ << x, y, 0, 0, 0;
      
      P_(0,0) = 1;
      P_(1,1) = 1;
      
      // done initializing
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
      
    } else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_) {
      // Initialize state.
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0,
            0,
            0;
      
      P_(0,0) = std_laspx_ * std_laspx_;
      P_(1,1) = std_laspy_ * std_laspy_;
      
      // done initializing
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    }
    
    return;
  }
  
  /**
   * Prediction
   */
  
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  
  UKF::Prediction(dt);
  
  /**
   * Update
   */
  
  /**
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */
  
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_) {
    // Radar updates
    UKF::UpdateRadar(meas_package);
    
  } else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_) {
    // Laser updates
    UKF::UpdateLidar(meas_package);
  }
  
  // print the output
  cout << "Posterior: " << endl;
  cout << "x_ = " << endl << x_ << endl;
  cout << "P_ = " << endl << P_ << endl;
  
}

void UKF::Prediction(double delta_t) {
  /**
   * Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  /**
   * Generate augmented sigma points
   */
  
  // create augmented mean state vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;
  
  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  
  // create augmented covariance matrix
  P_aug << P_, MatrixXd::Zero(n_x_, 2), MatrixXd::Zero(2,n_x_), Q_;
  
  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  
  // create augmented sigma points
  Xsig_aug << x_aug, (sqrt(lambda_ + n_aug_) * A).colwise() + x_aug, -((sqrt(lambda_ + n_aug_) * A).colwise() - x_aug);
  
  /**
   * Predict sigma points (insert them into the process model)
   */
  
  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  
  // predict sigma points
  for (int i=0; i<Xsig_aug.cols(); ++i) {
    VectorXd sig_pt = Xsig_aug.col(i);
    VectorXd sig_pt_pred(n_x_);
    
    // process noise vector
    VectorXd nu_k(n_x_);
    
    // avoid division by zero
    if (sig_pt[4] == 0) {
      // calculate the integral part
      sig_pt_pred <<  sig_pt[2] * cos(sig_pt[3]) * delta_t,
                      sig_pt[2] * sin(sig_pt[3]) * delta_t,
                      0,
                      0,
                      0;
      
    } else {
      sig_pt_pred <<  sig_pt[2] * (sin(sig_pt[3] + sig_pt[4] * delta_t) - sin(sig_pt[3])) / sig_pt[4],
                      sig_pt[2] * (-cos(sig_pt[3] + sig_pt[4] * delta_t) + cos(sig_pt[3])) / sig_pt[4],
                      0,
                      sig_pt[4] * delta_t,
                      0;
    }
    
    nu_k << delta_t * delta_t * cos(sig_pt[3]) * sig_pt[5] / 2,
            delta_t * delta_t * sin(sig_pt[3]) * sig_pt[5] / 2,
            delta_t * sig_pt[5],
            delta_t * delta_t * sig_pt[6] / 2,
            delta_t * sig_pt[6];
    
    sig_pt_pred = sig_pt.head(5) + sig_pt_pred + nu_k;
    
    // write predicted sigma points into right column
    Xsig_pred_.col(i) = sig_pt_pred;
  }
  
  /**
   * Predict Mean and Covariance
   */
  
  // set weights
  weights_ << lambda_ / (lambda_ + n_aug_), VectorXd::Constant(2*n_aug_, 1 / (2 * (lambda_ + n_aug_)));
  
  // predict state mean
  x_ = (Xsig_pred_ * weights_.asDiagonal()).rowwise().sum();
  
  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {  // iterate over sigma points
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff[3] = remainder(x_diff[3], 2*M_PI);
    P_ = P_ + weights_[i] * x_diff * x_diff.transpose();
  }
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  /**
   * Predict Measurement
   */
  
  // set measurement dimension, lidar can measure px, py
  int n_z = 2;
  
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  // transform sigma points into measurement space
  for (int i=0; i<Xsig_pred_.cols(); ++i) {
    float px = Xsig_pred_(0, i);
    float py = Xsig_pred_(1, i);
    Zsig.col(i) << px, py;
  }
  
  // calculate mean predicted measurement
  z_pred = (Zsig * weights_.asDiagonal()).rowwise().sum();
  
  // calculate innovation covariance matrix S
//  S.fill(0.0);
  S = (Zsig.colwise() - z_pred) * weights_.asDiagonal() * (Zsig.colwise() - z_pred).transpose();
//  for (int i = 0; i < Zsig.cols(); ++i) {  // iterate over sigma points
//    VectorXd z_diff = Zsig.col(i) - z_pred;
//    S = S + weights_[i] * z_diff * z_diff.transpose();
//  }
  
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R.diagonal() << std_laspx_ * std_laspx_, std_laspy_ * std_laspy_;
  
  S = S + R;
  
  /**
   * Update State
   */
  
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  // vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;
  //  cout << "z = " << endl << z << endl;
  
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {  // iterate over sigma points
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // Normalize angles
    x_diff[3] = remainder(x_diff[3], 2*M_PI);
    
    Tc = Tc + weights_[i] * x_diff * z_diff.transpose();
  }
  
  // calculate Kalman gain K;
  MatrixXd K;
  K = Tc * S.inverse();
  
  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;
  
  P_ = P_ - K * S * K.transpose();
  
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  
  /**
   * Predict Measurement
   */
  
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  // transform sigma points into measurement space
  for (int i=0; i<Xsig_pred_.cols(); ++i) {
    float px = Xsig_pred_(0, i);
    float py = Xsig_pred_(1, i);
    float v = Xsig_pred_(2, i);
    float psi = Xsig_pred_(3, i);
    
    float rho = sqrt(px * px + py * py);
    float phi = atan2(py, px);
    if (rho == 0) {
      cout << "Division by zero!" << endl;
      return;
    }
    float rhodot = (px * cos(psi) * v + py * sin(psi) * v) / rho;
    Zsig.col(i) << rho, phi, rhodot;
  }
  
  // calculate mean predicted measurement
  z_pred = (Zsig * weights_.asDiagonal()).rowwise().sum();
  
  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {  // iterate over sigma points
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff[1] = remainder(z_diff[1], 2*M_PI);
    S = S + weights_[i] * z_diff * z_diff.transpose();
  }
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R.diagonal() << std_radr_ * std_radr_, std_radphi_ * std_radphi_, std_radrd_ * std_radrd_;
  
  S = S + R;
  
  /**
   * Update State
   */
  
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  // vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;
//  cout << "z = " << endl << z << endl;
  
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {  // iterate over sigma points
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // Normalize angles
    x_diff[3] = remainder(x_diff[3], 2*M_PI);
    z_diff[1] = remainder(z_diff[1], 2*M_PI);
    
    Tc = Tc + weights_[i] * x_diff * z_diff.transpose();
  }
  
  // calculate Kalman gain K;
  MatrixXd K;
  K = Tc * S.inverse();
  
  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  z_diff[1] = remainder(z_diff[1], 2*M_PI);
  x_ = x_ + K * z_diff;
  
  P_ = P_ - K * S * K.transpose();
}
