#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from benchmarking.benchmark_assignment2 import benchmark_assignment2


def im2segment(im):
    # return a list of true/false images of the same size as im
    # This is a bad test implementation, please change to your code!
    
    nrofsegments = 5 # could vary, probably you should estimate this
    m,n = im.shape # image size
    S = []
    for kk in range(nrofsegments):
        S =  S + [np.random.rand(m,n)<0.5] # this is not a good segmentation method...
    
    return S



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

    
