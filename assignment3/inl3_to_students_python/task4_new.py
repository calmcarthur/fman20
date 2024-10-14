#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy


def tlsFit(x, y):
    # Stack points, and center
    points = np.column_stack((x, y))
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    
    # SVD! This is the key. This gives us the transpose right singular vectors.
    _, _, vh = np.linalg.svd(centered_points)
    line_direction = vh[-1]
    
    # Division of x and y to get the slope
    slope = -line_direction[0] / line_direction[1]
    
    # Standard intercept formula
    intercept = mean[1] - slope * mean[0]
    
    return slope, intercept

def ransacFit(x, y):

    best_slope = 0
    best_intercept = 0
    best_inliers = 0
    max_inliers = 0

    iterations = 100
    threshold = 1.0

    for _ in range(iterations):

        # First, take two random points
        sample = np.random.choice(len(x), 2, replace=False)
        x_sample = x[sample]
        y_sample = y[sample]
        
        # Fit a line using numpy built in polyfit, and get distances to line with numpy
        slope, intercept = np.polyfit(x_sample, y_sample, 1)
        distances = np.abs(y - (slope * x + intercept)) / np.sqrt(slope**2 + 1)
        
        # INLINERS
        inliers = distances < threshold
        n_inliers = np.sum(inliers)
        
        # Update if # of inliers is larger, then we know its the best model so far
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_slope = slope
            best_intercept = intercept
            best_inliers = inliers
    
    return best_slope, best_intercept, best_inliers

def calculate_errors(xm, ym, tls_slope, tls_intercept, ransac_slope, ransac_intercept, inlier_mask):
    # Predictions
    tls_pred = tls_slope * xm + tls_intercept
    ransac_pred = ransac_slope * xm + ransac_intercept
    
    # LS, TLS error
    ls_tls_error = np.sum((ym - tls_pred) ** 2)
    tls_tls_error = np.sum(np.abs(tls_slope * xm - ym + tls_intercept) / np.sqrt(tls_slope ** 2 + 1))
    
    # Now only on inliers for RANSAC
    inliers_xm = xm[inlier_mask]
    inliers_ym = ym[inlier_mask]
    
    # TLS, LS error
    ls_ransac_error = np.sum((inliers_ym - ransac_pred[inlier_mask]) ** 2)
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
    
    # Perform TLS Fit
    tls_slope, tls_intercept = tlsFit(xm, ym)
    
    # Perform RANSAC Fit
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