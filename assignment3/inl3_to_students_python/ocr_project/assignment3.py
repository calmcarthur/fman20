#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from benchmarking.benchmark_assignment3 import benchmark_assignment3


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


def feature2class(x, classdata):
    # return a classification 0..9 for the feature x
    # This is a bad test implementation, please change to your code!
    return np.random.randint(10)

if __name__ == "__main__": 
    
    # Choose dataset
    
    datadir = os.path.join('datasets','short1')  # Which folder of examples are you going to test it on?
    # datadir = os.path.join('datasets','home1')  # Which folder of examples are you going to test it on?
    
    
    # Benchmark and visualize
    
    mode = 0 # debug modes 
    # 0 with no plots
    # 1 with some plots
    # 2 with the most plots || We recommend setting mode = 2 if you get bad
    # results, where you can now step-by-step check what goes wrong. You will
    # get a plot showing some letters, and it will step-by-step show you how
    # the segmentation worked, and what your classifier classified the letter
    # as. Press any button to go to the next letter, and so on.
       
   
    hitrate,confmat,allres,alljs,alljfg,allX,allY = benchmark_assignment3(im2segment, segment2feature, feature2class,0,datadir,mode)
    print('Hitrate = ' + str(hitrate*100) + '%')

