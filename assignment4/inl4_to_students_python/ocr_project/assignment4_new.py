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
from scipy.spatial import distance
import time


def im2segment(im):
    # Thresholding process with np.where function.
    threshold = 40 # This value was determined through trial and error.
    thresholded_im = np.where(im > threshold, 1, 0)

    # Removing small objects below 5 pixels, with built in skimage.morphology function.
    cleaned_image = morphology.remove_small_objects(thresholded_im.astype(bool), min_size=5)

    # Labelling the digits, using 8-connectivity not 4 because this gave better results. 
    # Then extracting properties of labeled image regions with skimage.measure.regionprops function.
    labels = label(cleaned_image, connectivity=2)
    regions = regionprops(labels)

    segments = []

    # This is performed for each region that we have. 
    for region in regions:
        # Extract coordinates of bounding box with built in region property bbox.
        min_row, min_col, max_row, max_col = region.bbox

        # Create same size image.
        segment = np.zeros_like(im)

        # Applies the label mask to set all other regions EXCEPT where the label matches to 0.
        segment[min_row:max_row, min_col:max_col] = thresholded_im[min_row:max_row, min_col:max_col] * (labels[min_row:max_row, min_col:max_col] == region.label)
        
        # Stores x-coordinate for reorganization since they are not in the same order.
        segments.append((min_col, segment))

    # Reorganizies by x-coordinate, and removes tuple.
    segments.sort(key=lambda x: x[0])
    segments = [segment for _, segment in segments]

    return segments

def segment2feature(segment):
    # Resize all segments to a fixed size (e.g., 64x64) to ensure uniform HOG feature dimensions

    fixed_size = (300, 300)
    resized_segment = cv2.resize(segment, fixed_size)

    features = []

    # Feature 1: Image width (number of columns with non-zero pixels)
    width = np.sum(np.any(resized_segment, axis=0))
    features.append(width)

    # Feature 2: Top-heaviness (upper half pixel sum / total pixel sum)
    height = resized_segment.shape[0]
    top_heaviness = np.sum(resized_segment[:height//2]) / np.sum(resized_segment)
    features.append(top_heaviness)

    # Feature 3: Right-heaviness (right half pixel sum / total pixel sum)
    right_heaviness = np.sum(resized_segment[:, resized_segment.shape[1]//2:]) / np.sum(resized_segment)
    features.append(right_heaviness)

    # Feature 4: Number of holes (connected components in inverted image)
    inverted_segment = cv2.bitwise_not(resized_segment)
    num_holes, _ = cv2.connectedComponents(inverted_segment)
    features.append(num_holes)

    # Feature 5: Vertical symmetry (symmetric pixels along the y-axis)
    vertical_symmetry = np.sum(resized_segment == np.flip(resized_segment, axis=0)) / resized_segment.size
    features.append(vertical_symmetry)

    # Feature 6: Horizontal symmetry (symmetric pixels along the x-axis)
    horizontal_symmetry = np.sum(resized_segment == np.flip(resized_segment, axis=1)) / resized_segment.size
    features.append(horizontal_symmetry)

    # Feature 7: Hu Moments (7 moments)
    moments = cv2.HuMoments(cv2.moments(resized_segment)).flatten()
    features.extend(moments)

    # Feature 8: HOG (Histogram of Oriented Gradients)
    fd, _ = hog(resized_segment, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=True)
    features.extend(fd)

    # Normalize all features together
    features = np.array(features)
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

    # Return final normalized feature array
    return normalized_features.reshape(-1, 1)

def feature2class(x, classification_data):
    X_train = classification_data['X_features']
    Y_train = classification_data['Y_labels']
    
    # Calculate Euclidean distances with scipy distance function
    # Change x to be a row vector with reshape
    distances = distance.cdist(X_train, x.reshape(1, -1), 'euclidean')
    # print(distances.shape)
    
    # Get index
    nearest_neighbor_index = np.argmin(distances)
 
    Y_train = Y_train.reshape(-1,1)

    return Y_train[nearest_neighbor_index]


def class_train(X, Y):
    # Nearest neighbour classification so no training needed
    classification_data = {
        'X_features': X,
        'Y_labels': Y
    }
    return classification_data

if __name__ == "__main__": 
    
    start_time = time.time()
    
    # Loading training data
    segment_train = np.load('ocrsegmentdata.npy')
    Y_train = np.load('ocrsegmentgt.npy')

    #Transforming training data into features
    X_train = []
    for i in range(len(segment_train)):
        features = segment2feature(segment_train[i])
        X_train.append(features.flatten())
    X_train = np.array(X_train)

    # Training the classifier
    classification_data = class_train(X_train,Y_train)
    
    # Run benchmark on all datasets
    
    datasets = ['short1','short2','home1','home2','home3']
    
    
    mode = 0 # debug modes 
    # 0 with no plots
    # 1 with some plots
   
    for ds in datasets:
        datadir = os.path.join('datasets',ds)
        hitrate,confmat,allres,alljs,alljfg,allX,allY = benchmark_assignment3(im2segment, segment2feature, feature2class,classification_data,datadir,mode)        
        print(ds + f', Hitrate = {hitrate*100:0.5}%')

    end_time = time.time()
    print(end_time - start_time)