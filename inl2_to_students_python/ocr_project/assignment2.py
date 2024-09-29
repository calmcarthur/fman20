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
        segment[min_row:max_row, min_col:max_col] = im[min_row:max_row, min_col:max_col] * (labels[min_row:max_row, min_col:max_col] == region.label)
        
        # Stores x-coordinate for reorganization since they are not in the same order.
        segments.append((min_col, segment))

    # Reorganizies by x-coordinate, and removes tuple.
    segments.sort(key=lambda x: x[0])
    segments = [segment for _, segment in segments]

    return segments



def segment2feature(Si):
    # return a feature vector (array with size nx1)
    # This is a bad test implementation, please change to your code!
    nrofeatures = 7
    features = np.random.rand(nrofeatures,1)
    return features



if __name__ == "__main__": 
    # read example image
    im = plt.imread('datasets/short1/im1.jpg') 
    
    # segment example image
    S = im2segment(im)
    
    # calculate feature vector for one of the segments
    Si = S[2]
    f = segment2feature(Si)
    print(f)
    
    
    
    
    # Benchmark your feature extractor routine on all images
    # The routine extracts features for all images in the dataset
    # and extracts features. It returns all features and the corresponding labels
    # im allX and allY respectively.
    # The routine also plots a 2d projection of your features
    # Hopefully the graph should separate the numbers in a good way
    
    datadir = os.path.join('datasets','short1')
    debug = True
    allX,allY = benchmark_assignment2(segment2feature,datadir,debug)

    
