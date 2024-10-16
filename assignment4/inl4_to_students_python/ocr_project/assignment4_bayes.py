#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.feature import hog
import cv2  # For SIFT and ORB
from skimage.measure import label, regionprops
from benchmarking.benchmark_assignment3 import benchmark_assignment3
from scipy.stats import norm  # For Gaussian distribution
import time

def im2segment(im):
    # Thresholding process with np.where function.
    threshold = 40  # This value was determined through trial and error.
    thresholded_im = np.where(im > threshold, 1, 0)

    # Removing small objects below 5 pixels, with built-in skimage.morphology function.
    cleaned_image = morphology.remove_small_objects(thresholded_im.astype(bool), min_size=5)

    # Labelling the digits, using 8-connectivity not 4 because this gave better results.
    labels = label(cleaned_image, connectivity=2)
    regions = regionprops(labels)

    segments = []

    # This is performed for each region that we have.
    for region in regions:
        # Extract coordinates of bounding box with built-in region property bbox.
        min_row, min_col, max_row, max_col = region.bbox

        # Create same size image.
        segment = np.zeros_like(im)

        # Applies the label mask to set all other regions EXCEPT where the label matches to 0.
        segment[min_row:max_row, min_col:max_col] = thresholded_im[min_row:max_row, min_col:max_col] * (labels[min_row:max_row, min_col:max_col] == region.label)

        # Stores x-coordinate for reorganization since they are not in the same order.
        segments.append((min_col, segment))

    # Reorganizes by x-coordinate, and removes tuple.
    segments.sort(key=lambda x: x[0])
    segments = [segment for _, segment in segments]

    return segments

FEATURE_LENGTH = 7357  # Define a fixed length for feature vectors

def segment2feature(segment):
    # Translate segment to center based on center of mass (C.O.M)
    rows, columns = np.nonzero(segment)
    center_of_mass_x = np.mean(columns)
    center_of_mass_y = np.mean(rows)

    # Calculate the shift in x and y directions to align with the center
    shift_x = (segment.shape[1] // 2) - center_of_mass_x
    shift_y = (segment.shape[0] // 2) - center_of_mass_y

    # Perform the translation using a transformation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(segment, M, (segment.shape[1], segment.shape[0]))

    features = []

    # Feature 1: Image width (number of columns with non-zero pixels)
    width = np.sum(np.any(translated, axis=0))
    features.append(width)

    # Feature 2: Top-heaviness (upper half pixel sum / total pixel sum)
    height = translated.shape[0]
    top_heaviness = np.sum(translated[:height//2]) / np.sum(translated)
    features.append(top_heaviness)

    # Feature 3: Right-heaviness (right half pixel sum / total pixel sum)
    right_heaviness = np.sum(translated[:, translated.shape[1]//2:]) / np.sum(translated)
    features.append(right_heaviness)

    # Feature 4: Number of holes (connected components in inverted image)
    inverted_segment = cv2.bitwise_not(translated)
    num_holes, _ = cv2.connectedComponents(inverted_segment)
    features.append(num_holes)

    # Feature 5: Vertical symmetry (symmetric pixels along the y-axis)
    vertical_symmetry = np.sum(translated == np.flip(translated, axis=0)) / translated.size
    features.append(vertical_symmetry)

    # Feature 6: Horizontal symmetry (symmetric pixels along the x-axis)
    horizontal_symmetry = np.sum(translated == np.flip(translated, axis=1)) / translated.size
    features.append(horizontal_symmetry)

    # Feature 7: Hu Moments (7 moments)
    moments = cv2.HuMoments(cv2.moments(translated)).flatten()
    features.extend(moments)

    # Feature 8: HOG (Histogram of Oriented Gradients)
    fd, _ = hog(translated, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    features.extend(fd)

    # Normalize all features together
    features = np.array(features)

    # Ensure features have the same length (truncate or pad with zeros)
    if len(features) > FEATURE_LENGTH:
        features = features[:FEATURE_LENGTH]  # Truncate
    else:
        features = np.pad(features, (0, FEATURE_LENGTH - len(features)), 'constant')  # Pad with zeros

    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

    # Return final normalized feature array
    return normalized_features.reshape(-1, 1)

def feature2class(x, classification_data):
    # Extract class statistics from the classification data
    means = classification_data['means']
    variances = classification_data['variances']
    priors = classification_data['priors']
    classes = classification_data['classes']
    
    likelihoods = []

    # Loop through each class and calculate posterior probabilities
    for i, cls in enumerate(classes):
        # Assume each feature follows a Gaussian distribution, compute log likelihood
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variances[i])) - \
                         0.5 * np.sum(((x.flatten() - means[i]) ** 2) / variances[i])
        posterior = np.log(priors[i]) + log_likelihood
        likelihoods.append(posterior)
    
    # Return the class with the highest posterior probability
    return classes[np.argmax(likelihoods)]

def class_train(X, Y):
    # Calculate mean, variance, and prior for each class
    classes = np.unique(Y)
    means = []
    variances = []
    priors = []

    for cls in classes:
        X_cls = X[Y.flatten() == cls]
        means.append(np.mean(X_cls, axis=0))
        variances.append(np.var(X_cls, axis=0) + 1e-8)  # Add small value to prevent division by zero
        priors.append(len(X_cls) / len(X))

    classification_data = {
        'means': np.array(means),
        'variances': np.array(variances),
        'priors': np.array(priors),
        'classes': classes
    }
    
    return classification_data

if __name__ == "__main__": 
    
    start_time = time.time()
    
    # Loading training data
    segment_train = np.load('ocrsegmentdata.npy')
    Y_train = np.load('ocrsegmentgt.npy')

    # Transforming training data into features
    X_train = []
    for i in range(len(segment_train)):
        features = segment2feature(segment_train[i])
        X_train.append(features.flatten())
    X_train = np.array(X_train)

    # Training the classifier using Gaussian distributions
    classification_data = class_train(X_train, Y_train)
    
    # Run benchmark on all datasets
    datasets = ['short1', 'short2', 'home1', 'home2', 'home3']
    
    mode = 0  # debug modes 
    for ds in datasets:
        datadir = os.path.join('datasets', ds)
        hitrate, confmat, allres, alljs, alljfg, allX, allY = benchmark_assignment3(im2segment, segment2feature, feature2class, classification_data, datadir, mode)
        print(ds + f', Hitrate = {hitrate*100:0.5}%')

    end_time = time.time()
    print(end_time - start_time)
