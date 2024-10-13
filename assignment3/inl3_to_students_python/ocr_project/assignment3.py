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


# def segment2feature(segment):
    
#     # FIRST TRANSLATING EACH SEGMENT TO REMOVE VARIANCE FROM POSITION BASED ON C.O.M
#     # Returns tuple with indices of non-zero elements
#     rows, columns = np.nonzero(segment) 

#     # x corresponds to columns
#     center_of_mass_x = np.mean(columns)
#     # y corresponds to rows
#     center_of_mass_y = np.mean(rows)

#     # Take overall size and divide by two then subtract the centre of mass to get shift index
#     shift_x = (segment.shape[1] // 2) - center_of_mass_x
#     shift_y = (segment.shape[0] // 2) - center_of_mass_y

#     # 0 axis is shifting by rows
#     # 1 axis is shifting by columns
#     M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
#     # Built in function suggested online for doing this shift in one clean step
#     translated = cv2.warpAffine(segment, M, (segment.shape[1], segment.shape[0]))
#     # plt.imshow(translated)
#     # plt.show()

#     # NOW CALCULATING FEATURES AND APPENDING TO LIST
#     features = []

#     # Feature 1: Corners with Harris
#     # corners = corner_peaks(corner_harris(translated), min_distance=1)
#     # corner_count = len(corners)
#     # features.append(corner_count)
#     # # print("Corners: ", corner_count)

#     # Feature 2: Scale-Invariant Feature Transform
#     sift = cv2.SIFT_create()
#     keypoints, _ = sift.detectAndCompute(translated.astype('uint8') * 255, None)
#     sift_keypoint_count = len(keypoints)
#     features.append(sift_keypoint_count)
#     # print("SIFT: ", sift_keypoint_count)

#     # Feature 3: Histogram of oriented gradients
#     hog_feature, _ = hog(translated, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
#     hog_sum = np.sum(hog_feature)
#     features.append(hog_sum)
#     # print("Histogram of oriented gradients: ", hog_sum)

#     # Feature 4: Central Moments to calculate Hu moments
#     c_moments = moments_central(translated)
#     hu_moments = moments_hu(c_moments)
#     features.extend(hu_moments)
#     # print("Hu Moments: ", hu_moments)

#     # Feature 5: Sum of pixel values in subregions
#     height, width = translated.shape
#     half_h, half_w = height // 2, width // 2
#     top_left = np.sum(translated[:half_h, :half_w])
#     top_right = np.sum(translated[:half_h, half_w:])
#     bottom_left = np.sum(translated[half_h:, :half_w])
#     bottom_right = np.sum(translated[half_h:, half_w:])
#     features.extend([top_left, top_right, bottom_left, bottom_right])
#     # print("Sum in subregions: ", top_left, top_right, bottom_left, bottom_right)

#     # Calculate the middle index for rows and columns
#     middle_row_index = translated.shape[0] // 2
#     middle_col_index = translated.shape[1] // 2

#     # Feature 6: Mean of pixel values in the middle row
#     middle_row_mean = np.mean(translated[middle_row_index, :])
#     features.append(middle_row_mean)
#     # print("Mean of the middle row: ", middle_row_mean)

#     # Feature 7: Mean of pixel values in the middle column
#     middle_col_mean = np.mean(translated[:, middle_col_index])
#     features.append(middle_col_mean)
#     # print("Mean of the middle column: ", middle_col_mean)

#     # Normalizing all at once
#     normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
#     features_final = normalized_features.reshape(-1, 1)  # Reshape to proper form (transpose)

#     return features_final

def segment2feature(segment):
    
    # FIRST TRANSLATING EACH SEGMENT TO REMOVE VARIANCE FROM POSITION BASED ON C.O.M
    rows, columns = np.nonzero(segment)

    center_of_mass_x = np.mean(columns)
    center_of_mass_y = np.mean(rows)

    shift_x = (segment.shape[1] // 2) - center_of_mass_x
    shift_y = (segment.shape[0] // 2) - center_of_mass_y

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(segment, M, (segment.shape[1], segment.shape[0]))

    features = []

    # Feature 1: Skeleton on perimeter ratio
    perimeter = np.sum(segment[:, 0]) + np.sum(segment[:, -1]) + np.sum(segment[0, :]) + np.sum(segment[-1, :])
    skeleton = np.sum(translated)  # Simplified skeleton approximation as non-zero elements
    skeleton_perimeter_ratio = skeleton / perimeter
    features.append(skeleton_perimeter_ratio)

    # Feature 2: Area on convex hull ratio
    convex_hull = cv2.convexHull(np.column_stack((columns, rows)))
    convex_area = cv2.contourArea(convex_hull)
    area_convex_ratio = np.sum(segment) / convex_area
    features.append(area_convex_ratio)

    # Feature 3: Bounding box width/height ratio
    x, y, w, h = cv2.boundingRect(np.column_stack((columns, rows)))
    box_aspect_ratio = w / h
    features.append(box_aspect_ratio)

    # Feature 4: Center of mass (x and y) position relative to the bounding box
    cMassXPos = center_of_mass_x / w
    cMassYPos = center_of_mass_y / h
    features.append(cMassXPos)
    features.append(cMassYPos)

    # Feature 5: Area covered by the image in the bounding box (normalized)
    area_cover_total = np.sum(segment) / (w * h)
    features.append(area_cover_total)

    # Feature 6: Area covered in each of the four corners of the bounding box
    upper_left = np.sum(translated[:h//2, :w//2]) / (w//2 * h//2)
    upper_right = np.sum(translated[:h//2, w//2:]) / (w//2 * h//2)
    lower_left = np.sum(translated[h//2:, :w//2]) / (w//2 * h//2)
    lower_right = np.sum(translated[h//2:, w//2:]) / (w//2 * h//2)
    features.extend([upper_left, upper_right, lower_left, lower_right])

    # Feature 7: Division after erosion (inverse of the number of connected components after erosion)
    eroded = cv2.erode(translated.astype(np.uint8), None, iterations=1)
    num_components, _ = cv2.connectedComponents(eroded)
    division_after_erosion = 1 / num_components
    features.append(division_after_erosion)

    # Feature 8: Holes inverse (inverse of number of holes in the image)
    inverted_segment = cv2.bitwise_not(translated)
    num_holes, _ = cv2.connectedComponents(inverted_segment)
    holes_inverse = 1 / num_holes
    features.append(holes_inverse)

    # Feature 9: Horizontal symmetry rate (rate of symmetric pixels along the x-axis)
    horizontal_symmetry = np.sum(translated == np.flip(translated, axis=1)) / translated.size
    features.append(horizontal_symmetry)

    # Feature 10: Vertical symmetry rate (rate of symmetric pixels along the y-axis)
    vertical_symmetry = np.sum(translated == np.flip(translated, axis=0)) / translated.size
    features.append(vertical_symmetry)

    # Normalizing all at once
    normalized_features = (np.array(features) - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    features_final = normalized_features.reshape(-1, 1)  # Reshape to proper form

    return features_final


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
    
    # datadir = os.path.join('datasets','short1')  # Which folder of examples are you going to test it on?
    datadir = os.path.join('datasets','home1')  # Which folder of examples are you going to test it on?

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

