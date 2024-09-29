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
from skimage.feature import blob_dog
from skimage.feature import hog
from skimage.measure import moments_central, moments_hu
import cv2  # For SIFT and ORB
from skimage.measure import label, regionprops

from benchmarking.benchmark_assignment2 import benchmark_assignment2


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


def is_black_and_white(image_array):
    return np.all(np.isin(image_array, [0, 1]))

def segment2feature(segment):
    
    # FIRST TRANSLATING EACH SEGMENT TO REMOVE VARIANCE FROM POSITION BASED ON C.O.M
    # Returns tuple with indices of non-zero elements
    rows, columns = np.nonzero(segment) 

    # x corresponds to columns
    center_of_mass_x = np.mean(columns)
    # y corresponds to rows
    center_of_mass_y = np.mean(rows)

    # Take overall size and divide by two then subtract the centre of mass to get shift index
    shift_x = (segment.shape[1] // 2) - center_of_mass_x
    shift_y = (segment.shape[0] // 2) - center_of_mass_y

    # 0 axis is shifting by rows
    # 1 axis is shifting by columns
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # Built in function suggested online for doing this shift in one clean step
    translated = cv2.warpAffine(segment, M, (segment.shape[1], segment.shape[0]))

    # NOW CALCULATING FEATURES AND APPENDING TO LIST
    features = []

    # Feature 1: Corners with Harris
    corners = corner_peaks(corner_harris(translated), min_distance=1)
    corner_count = len(corners)
    features.append(corner_count)
    # print("Corners: ", corner_count)

    # Feature 2: Scale-Invariant Feature Transform
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(translated.astype('uint8') * 255, None)
    sift_keypoint_count = len(keypoints)
    features.append(sift_keypoint_count)
    # print("SIFT: ", sift_keypoint_count)

    # Feature 3: Histogram of oriented gradients
    hog_feature, _ = hog(translated, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_sum = np.sum(hog_feature)
    features.append(hog_sum)
    # print("Histogram of oriented gradients: ", hog_sum)

    # Feature 4: Central Moments to calculate Hu moments
    c_moments = moments_central(translated)
    hu_moments = moments_hu(c_moments)
    features.extend(hu_moments)
    # print("Hu Moments: ", hu_moments)

    # Feature 5: Sum of pixel values in subregions
    height, width = translated.shape
    half_h, half_w = height // 2, width // 2
    top_left = np.sum(translated[:half_h, :half_w])
    top_right = np.sum(translated[:half_h, half_w:])
    bottom_left = np.sum(translated[half_h:, :half_w])
    bottom_right = np.sum(translated[half_h:, half_w:])
    features.extend([top_left, top_right, bottom_left, bottom_right])
    # print("Sum in subregions: ", top_left, top_right, bottom_left, bottom_right)

    # Feature 6: Sum of pixel values in each row
    row_sums = np.sum(segment, axis=1)
    features.extend(row_sums)
    # print("Sum over each row: ", row_sums)

    # Feature 7: Sum of pixel values in each column
    col_sums = np.sum(segment, axis=0)
    features.extend(col_sums)
    # print("Sum over each col: ", col_sums)

    # Normalizing all at once
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    features_final = normalized_features.reshape(-1, 1)  # Reshape to proper form (transpose)

    return features_final


if __name__ == "__main__": 
    # my test:
    # get all segments
    # all_segments = []
    # for x in range(1,11):
    #     image = plt.imread(f'datasets/short1/im{x}.jpg')
    #     segs = im2segment(image)
    #     all_segments.extend(segs)


    # count = 0
    # for x in all_segments:

    #     features = segment2feature(x)
    #     plt.imshow(x)
    #     plt.title(features)
    #     plt.show()
    #     print(f"Feature {count}: ",features)
    #     print(count)
    #     count += 1
        
    
    # my test:
    # ones = []
    # for x in range(1,7):
    #     im = plt.imread(f'datasets/short1/im{x}.jpg')
    #     segs = im2segment(im)

    #     with open(f'datasets/short1/im{x}.txt', 'r') as file:
    #         number = file.read().strip()
    #         indices = [i for i, digit in enumerate(number) if digit == '1']
    #     for x in indices:
    #         ones.append(segs[x])

    # for x in ones:
    #     print("ONE: ", f)
    #     f = segment2feature(x)

    # Benchmark your feature extractor routine on all images
    # The routine extracts features for all images in the dataset
    # and extracts features. It returns all features and the corresponding labels
    # im allX and allY respectively.
    # The routine also plots a 2d projection of your features
    # Hopefully the graph should separate the numbers in a good way
    
    datadir = os.path.join('datasets','short1')
    debug = True
    allX,allY = benchmark_assignment2(segment2feature,datadir,debug)