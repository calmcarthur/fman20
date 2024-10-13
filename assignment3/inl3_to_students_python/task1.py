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
import matplotlib.pyplot as plt

def class_train(X, Y):
    # Nearest neighbour classification so no training needed
    classification_data = {
        'X_features': X,
        'Y_labels': Y
    }
    return classification_data

def classify(x, classification_data):
    X_train = classification_data['X_features']
    Y_train = classification_data['Y_labels']
    
    # Calculate Euclidean distances with scipy distance function
    # Change x to be a row vector with reshape
    distances = distance.cdist(X_train, x.reshape(1, -1), 'euclidean')
    # print(distances.shape)
    
    # Get index
    nearest_neighbor_index = np.argmin(distances)
    
    return Y_train[nearest_neighbor_index]

def test_impemenetation(X,Y):
    # Reshape into proper 19x19 images, need to transpose
    image_0 = np.reshape(X[0], (19, 19)).T
    image_4 = np.reshape(X[4], (19, 19)).T

    plt.figure(figsize=(5, 3))

    # Image 1
    plt.subplot(1, 2, 1)
    plt.imshow(image_0, cmap='gray')
    plt.title('Face Image (Correctly Classified)')
    plt.axis('off')

    # Image 2
    plt.subplot(1, 2, 2)
    plt.imshow(image_4, cmap='gray')
    plt.title('Non-Face Image (Correctly Classified)]')
    plt.axis('off')

    plt.show()

    print(classify(X[0], class_train(X, Y)))
    print(classify(X[4], class_train(X, Y)))
    return None

if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = './' 
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')
    X = data['X'].transpose()  # Transpose to make examples rows
    Y = data['Y'].transpose()  # Transpose to match X dimensions
    nbr_examples = np.size(Y, 0)
    
    test_impemenetation(X,Y)

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