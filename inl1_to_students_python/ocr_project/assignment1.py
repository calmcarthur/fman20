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

from benchmarking.benchmark_assignment1 import benchmark_assignment1

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


if __name__ == "__main__":
    # read example image
    im = plt.imread('datasets/short1/im1.jpg') 
    
    # # read ground truth numbers
    gt_file = open('datasets/short1/im1.txt','r') 
    gt = gt_file.read()
    gt = gt[:-1] # remove newline character
    gt_file.close()
    
    # show image with ground truth
    plt.imshow(im)
    plt.title(gt)
    
    # segment example image
    S = im2segment(im)
    
    # Plot all the segments
    fig,axs = plt.subplots(len(S),1) # create n x 1 subplots 
    
    for Si,axi in zip(S,axs):  # loop over segments and subplots
        axi.imshow(Si,cmap = 'gray',vmin = 0, vmax = 1.0)
    
    # Benchmark your segmentation routine on all images
    
    datadir = os.path.join('datasets','short1')
    debug = True
    stats = benchmark_assignment1(im2segment,datadir,debug)
    if stats != 0:
        print(f'Total mean Jaccard score is {np.mean(stats[0]):.2}')
