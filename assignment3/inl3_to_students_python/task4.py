#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

def tlsFit(x, y):

    # Taken from documentation
    def f(B, x):
        '''Linear function y = m*x + b'''
        # B is a vector of the parameters.
        # x is an array of the current x values.
        # x is in the same format as the x passed to Data or RealData.
        #
        # Return an array in the same format as y passed to Data or RealData.
        return B[0]*x + B[1]
    
    # Perform required setup required.
    model = scipy.odr.Model(f)
    data = scipy.odr.Data(x, y)
    odr_obj = scipy.odr.ODR(data, model, beta0=[1.0, 0.0])
    
    # Run get slope and intercept
    out = odr_obj.run()
    slope, int = out.beta
    return slope, int

def ransacFit(x, y):
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    
    # Use built in RANSAC from sklearn
    ransac = RANSACRegressor()
    ransac.fit(X, Y)
    
    # For errors later
    inlier = ransac.inlier_mask_

    slope = ransac.estimator_.coef_[0][0]
    int = ransac.estimator_.intercept_[0]
    
    return slope, int, inlier

def calculate_errors(xm, ym, tls_slope, tls_intercept, ransac_slope, ransac_intercept, inlier_mask):
    # Predictions
    tls_pred = tls_slope * xm + tls_intercept
    ransac_pred = ransac_slope * xm + ransac_intercept
    
    # LS errors or squared differences
    ls_tls_error = np.sum((ym - tls_pred) ** 2)
    ls_ransac_error = np.sum((ym - ransac_pred) ** 2)
    
    # Orthogonal distance error
    tls_tls_error = np.sum(np.abs(tls_slope * xm - ym + tls_intercept) / np.sqrt(tls_slope ** 2 + 1))
    
    # Only on inliers
    inliers_xm = xm[inlier_mask]
    inliers_ym = ym[inlier_mask]
    tls_ransac_error = np.sum(np.abs(ransac_slope * inliers_xm - inliers_ym + ransac_intercept) / np.sqrt(ransac_slope ** 2 + 1))
    
    return ls_tls_error, tls_tls_error, ls_ransac_error, tls_ransac_error

if __name__ == "__main__":
    # Load data, change datadir path if your data is elsewhere
    datadir = './'
    data = scipy.io.loadmat(datadir + 'linedata.mat')
    xm = data['xm'].flatten()
    ym = data['ym'].flatten()
    
    # Plot the raw data
    plt.scatter(xm, ym, color='gray', label='Data Points')
    
    tls_slope, tls_intercept = tlsFit(xm, ym)
    ransac_slope, ransac_intercept, inlier_mask = ransacFit(xm, ym)

    # Calculate errors
    ls_tls_error, tls_tls_error, ls_ransac_error, tls_ransac_error = calculate_errors(
        xm, ym, tls_slope, tls_intercept, ransac_slope, ransac_intercept, inlier_mask
    )
    
    print(f"LS TLS Error: {ls_tls_error}")
    print(f"TLS TLS Error: {tls_tls_error}")
    print(f"LS RANSAC Error: {ls_ransac_error}")
    print(f"TLS RANSAC Error: {tls_ransac_error}")

    # Plot
    plt.plot(xm, tls_slope * xm + tls_intercept, color='orange', label='TLS Line Fit')
    plt.plot(xm, ransac_slope * xm + ransac_intercept, color='blue', label='RANSAC Line Fit')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Fitting: TLS vs RANSAC')
    plt.grid(True)
    plt.show()