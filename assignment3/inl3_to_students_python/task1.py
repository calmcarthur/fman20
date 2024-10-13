#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

# NEAREST NEIGHBOUR CLASSIFICATION
def class_train(X, Y):
    classification_data = {
        'X_train': X,  # Store the training feature vectors
        'Y_train': Y   # Store the corresponding labels
    }
    return classification_data

# Function to classify using nearest neighbor method
def classify(x, classification_data):
    X_train = classification_data['X_train']
    Y_train = classification_data['Y_train']
    
    # Calculate Euclidean distances from test point x to all training points
    distances = distance.cdist(X_train, x.reshape(1, -1), 'euclidean')
    
    # Find the index of the nearest neighbor
    nearest_neighbor_idx = np.argmin(distances)
    
    # Return the label of the nearest neighbor
    return Y_train[nearest_neighbor_idx]


if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = './' 
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')
    X = data['X'].transpose()  # Transpose to make examples rows
    Y = data['Y'].transpose()  # Transpose to match X dimensions
    nbr_examples = np.size(Y, 0)
    
    # This outer loop will run 100 times, so that you get a mean error for your
    # classifier (the error will become different each time due to the
    # randomness of train_test_split, which you may verify if you wish).
    nbr_trials = 100
    err_rates_test = np.zeros((nbr_trials, 1))
    err_rates_train = np.zeros((nbr_trials, 1))
    
    for i in range(nbr_trials):
        
        # First split data into training / testing (80% train, 20% test)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        nbr_train_examples = np.size(Y_train, 0)
        nbr_test_examples = np.size(Y_test, 0)
        
        # Now we can train our model!
        classification_data = class_train(X_train, Y_train)
            
        # Classify the test data
        predictions_test = np.zeros((nbr_test_examples, 1))
        for j in range(nbr_test_examples):
            predictions_test[j] = classify(X_test[j, :], classification_data)
        
        # Classify the training data
        predictions_train = np.zeros((nbr_train_examples, 1))
        for j in range(nbr_train_examples):
            predictions_train[j] = classify(X_train[j, :], classification_data)
            
        # Compute the respective error rates
        pred_test_diff = predictions_test - Y_test
        pred_train_diff = predictions_train - Y_train
        err_rate_test = np.count_nonzero(pred_test_diff) / nbr_test_examples
        err_rate_train = np.count_nonzero(pred_train_diff) / nbr_train_examples
        
        # Store them in the containers
        err_rates_test[i] = err_rate_test
        err_rates_train[i] = err_rate_train
    
    # Print mean error rates across all trials
    print(f"Mean Test Error Rate: {np.mean(err_rates_test):.4f}")
    print(f"Mean Train Error Rate: {np.mean(err_rates_train):.4f}")