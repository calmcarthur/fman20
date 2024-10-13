#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt






if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = './'
    data = scipy.io.loadmat(datadir + 'linedata.mat')
    xm = data['xm'].flatten()
    ym = data['ym'].flatten()
    
    plt.scatter(xm,ym)
    
    
    