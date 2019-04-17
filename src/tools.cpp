#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */
  int n_gt = 4;
  
  VectorXd rmse(n_gt);
  rmse.fill(0.0);
  
  if (estimations.size() == 0) {
    cout << "Estimations are empty." << endl;
    return rmse;
  }
  if (estimations.size() != ground_truth.size()) {
    cout << "Estimations and ground_truth should be of equal size." << endl;
    return rmse;
  }
  
  // accumulate squared residuals
  VectorXd sum(n_gt);
  sum.fill(0.0);
  
  for (int i=0; i < estimations.size(); ++i) {
    VectorXd resid(n_gt);
    VectorXd sq_resid(n_gt);
    resid = estimations[i] - ground_truth[i];
    
    sq_resid = resid.array()*resid.array();
    sum = sum + sq_resid;
  }
  
  // calculate the mean
  VectorXd mean(n_gt);
  mean = sum / estimations.size();
  
  // calculate the squared root
  rmse = mean.array().sqrt();
  
  return rmse;
}
