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

# part 2
from skimage.feature import corner_harris, corner_peaks
from skimage.feature import hog
from skimage.measure import moments_central, moments_hu
import cv2  # For SIFT and ORB
from skimage.measure import label, regionprops
from benchmarking.benchmark_assignment3 import benchmark_assignment3

# part 3
from scipy.spatial import distance


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
    width = translated.shape[1]
    right_heaviness = np.sum(translated[:, width//2:]) / np.sum(translated)
    features.append(right_heaviness)

    # Feature 4: Number of holes (connected components in inverted image)
    inverted_segment = cv2.bitwise_not(translated)
    num_holes, _ = cv2.connectedComponents(inverted_segment)
    features.append(num_holes)

    # Feature 5: Holes inverse (1 / number of holes)
    holes_inverse = 1 / num_holes
    features.append(holes_inverse)

    # Feature 6: Sum of pixel values in four quadrants (Top-left, Top-right, Bottom-left, Bottom-right)
    half_h, half_w = height // 2, width // 2
    top_left = np.sum(translated[:half_h, :half_w])
    top_right = np.sum(translated[:half_h, half_w:])
    bottom_left = np.sum(translated[half_h:, :half_w])
    bottom_right = np.sum(translated[half_h:, half_w:])
    features.extend([top_left, top_right, bottom_left, bottom_right])

    # Feature 7: Vertical symmetry (symmetric pixels along the y-axis)
    vertical_symmetry = np.sum(translated == np.flip(translated, axis=0)) / translated.size
    features.append(vertical_symmetry)

    # Feature 8: Horizontal symmetry (symmetric pixels along the x-axis)
    horizontal_symmetry = np.sum(translated == np.flip(translated, axis=1)) / translated.size
    features.append(horizontal_symmetry)

    # Feature 9: Mean of pixel values in the middle row
    middle_row_mean = np.mean(translated[translated.shape[0] // 2, :])
    features.append(middle_row_mean)

    # Feature 10: Mean of pixel values in the middle column
    middle_col_mean = np.mean(translated[:, translated.shape[1] // 2])
    features.append(middle_col_mean)

    # Feature 11: Division after erosion (1 / number of connected components after erosion)
    eroded = cv2.erode(translated.astype(np.uint8), None, iterations=1)
    num_components, _ = cv2.connectedComponents(eroded)
    division_after_erosion = 1 / num_components
    features.append(division_after_erosion)

    # Feature 12: Aspect ratio of the bounding box (width / height)
    x, y, w, h = cv2.boundingRect(np.column_stack((columns, rows)))
    box_aspect_ratio = w / h
    features.append(box_aspect_ratio)

    # Feature 13: Area cover (pixel sum / bounding box area)
    area_cover_total = np.sum(segment) / (w * h)
    features.append(area_cover_total)

    # Feature 14: Area to convex hull ratio (pixel sum / convex hull area)
    convex_hull = cv2.convexHull(np.column_stack((columns, rows)))
    convex_area = cv2.contourArea(convex_hull)
    area_convex_ratio = np.sum(segment) / convex_area
    features.append(area_convex_ratio)

    # Normalize all features together
    features = np.array(features)
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)  # Normalize all at once

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
    
    # Choose dataset
    datadir = os.path.join('datasets','short1')  # Which folder of examples are you going to test it on?
    # datadir = os.path.join('datasets','home1')  # Which folder of examples are you going to test it on?

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

    # Benchmark and visualize
    
    mode = 0 # debug modes 
    # 0 with no plots
    # 1 with some plots
    # 2 with the most plots || We recommend setting mode = 2 if you get bad
    # results, where you can now step-by-step check what goes wrong. You will
    # get a plot showing some letters, and it will step-by-step show you how
    # the segmentation worked, and what your classifier classified the letter
    # as. Press any button to go to the next letter, and so on.
       
   
    hitrate,confmat,allres,alljs,alljfg,allX,allY = benchmark_assignment3(im2segment, segment2feature, feature2class,classification_data,datadir,mode)
    print('Hitrate = ' + str(hitrate*100) + '%')

