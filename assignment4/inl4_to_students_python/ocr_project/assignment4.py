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
import cv2
from skimage.measure import label, regionprops
from benchmarking.benchmark_assignment3 import benchmark_assignment3
from scipy.spatial import distance
import time
from skimage import morphology
from skimage.feature import local_binary_pattern
from skimage.transform import radon


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

def segment2feature(segment, n_coeffs=20):
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

    # Feature 9: Bounding box aspect ratio
    min_row, min_col, max_row, max_col = cv2.boundingRect(np.column_stack(np.where(translated > 0)))
    aspect_ratio = (max_row - min_row) / (max_col - min_col) if (max_col - min_col) > 0 else 0
    features.append(aspect_ratio)

    # Feature 10: Histogram of pixel intensities
    histogram, _ = np.histogram(translated, bins=10, range=(0, 1))
    features.extend(histogram)

    # Feature 11: Contour perimeter and area
    contours, _ = cv2.findContours(translated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        area = cv2.contourArea(contours[0])
        features.append(perimeter)
        features.append(area)

    # Feature 12: Local Binary Pattern (LBP) Histogram
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(translated, n_points, radius, method='uniform')
    lbp_hist = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))[0]
    features.extend(lbp_hist)

    # Feature 13: Radial Symmetry (using Radon transform)
    theta = np.linspace(0., 180., max(translated.shape), endpoint=False)
    sinogram = radon(translated, theta=theta)
    radial_symmetry = np.sum(np.var(sinogram, axis=0))
    features.append(radial_symmetry)

    # Feature 14: Compactness
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    area = cv2.contourArea(contours[0]) if contours else 0
    compactness = perimeter ** 2 / (4 * np.pi * area) if area > 0 else 0
    features.append(compactness)

    # Feature 15: Centroid distance from border
    dist_from_border = min(min(min_row, translated.shape[0] - max_row), min(min_col, translated.shape[1] - max_col))
    features.append(dist_from_border)

    # Feature 17: Skeletonization features (e.g., number of skeleton pixels)
    skeleton = morphology.skeletonize(translated)
    num_skeleton_pixels = np.sum(skeleton)
    features.append(num_skeleton_pixels)

    # Feature 18: Run-Length Encoding (RLE) - Number of runs (horizontal)
    rle = []
    prev_pixel = translated[0, 0]
    run_length = 1
    for i in range(1, translated.shape[1]):
        if translated[0, i] == prev_pixel:
            run_length += 1
        else:
            rle.append(run_length)
            prev_pixel = translated[0, i]
            run_length = 1
    rle.append(run_length)
    features.append(len(rle))

    # Normalize all features together
    features = np.array(features)
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

    return normalized_features.reshape(-1, 1)


def feature2class(x, classification_data, k=5):
    X_train = classification_data['X_features']
    Y_train = classification_data['Y_labels']
    
    # Calculate Euclidean distances with scipy distance function
    distances = distance.cdist(X_train, x.reshape(1, -1), 'euclidean')
    
    # Get the indices of the k nearest neighbors
    nearest_neighbor_indices = np.argsort(distances, axis=0)[:k].flatten()
    Y_train = Y_train.reshape(-1,1)

    # Get the labels of the k nearest neighbors
    nearest_labels = Y_train[nearest_neighbor_indices]
    
    # Count the most frequent label among the k neighbors
    unique, counts = np.unique(nearest_labels, return_counts=True)
    
    # Return the label with the highest count
    return unique[np.argmax(counts)]

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