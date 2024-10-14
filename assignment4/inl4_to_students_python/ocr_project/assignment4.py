#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
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
    
    
    # train your classifier or load classification data
    
    # here we just set it to zero, since our classifier is nonsense
    classification_data = 0
    
    
    
    # Run benchmark on all datasets
    
    datasets = ['short1','short2','home1','home2','home3']
    
    
    mode = 0 # debug modes 
    # 0 with no plots
    # 1 with some plots
   
    for ds in datasets:
        datadir = os.path.join('datasets',ds)
        hitrate,confmat,allres,alljs,alljfg,allX,allY = benchmark_assignment3(im2segment, segment2feature, feature2class,classification_data,datadir,mode)        
        print(ds + f', Hitrate = {hitrate*100:0.5}%')

