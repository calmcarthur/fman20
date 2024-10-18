#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:06:01 2024

@author: magnuso
"""

import numpy as np
import scipy
from scipy.sparse.csgraph import maximum_flow
from scipy import sparse
import matplotlib.pyplot as plt

def edges4connected(height,width,only_one_dir = 0):
    #
    # Generates a 4-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    
    N = height*width
    I = np.array([])
    J = np.array([])
    
    # connect vertically (down, then up)
    iis = np.delete(np.arange(N),np.arange(height-1,N,height))
    jjs = iis+1;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    # connect horizontally (right, then left)
    
    iis = np.arange(0,N-height)
    jjs = iis+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    return I,J

def edges8connected(height,width,only_one_dir = 0):
    #
    # Generates a 8-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    
    N = height*width
    I = np.array([])
    J = np.array([])
    
    # connect vertically (down, then up)
    iis = np.delete(np.arange(N),np.arange(height-1,N,height))
    jjs = iis+1;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    # Diagonal
    jjs = iis+1+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))

    jjs = iis+1-height;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))


    ind = (I>=0) & (I<N)
    I = I[ind]
    J = J[ind]
    ind = (J>=0) & (J<N)
    I = I[ind]
    J = J[ind]
    

    # connect horizontally (right, then left)
    
    iis = np.arange(0,N-height)
    jjs = iis+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    return I,J


if __name__ == '__main__':

    
    # example of setting up a graph problem and run max flow
    
    # example with flow from node 0 to 1 = 5
    # flow from node 0 to 2 = 2
    # flow from node 1 to 3 = 3
    # flow from node 2 to 3 = 7
    
    # I are indices from nodes
    # J are indices to nodes
    # V are flow values
    I = np.array([0,0,1,2])
    J = np.array([1,2,3,3])
    V = np.array([5,2,3,7])
    
    # the max flow algorithm wants integer values so we change to that
    I = I.astype(np.int32)
    J = J.astype(np.int32)
    V = V.astype(np.int32)
    
    # construct  a graph with edges between nodes I and J with weights V
    # make it a sparse matrix in special format that the max flow algorithm uses
    F = sparse.coo_array((V,(I,J)),shape=(4,4)).tocsr()
    
    # calculate the max flow between node  0 and 3 
    mf = maximum_flow(F, 0, 3)
    
    # In this case the maximum flow is given by passing 3 from node 0 to 1 to 3 and
    # passing 2 from node 0 to 2 to 3, in total 5
    
    print(mf.flow_value)
    
    # the flow is given by
    print(mf.flow.toarray())
    
    
    
    # Now we turn to our real problem
    # Fix things below that are missing
    

    # load data
    data = scipy.io.loadmat('heart_data.mat')
    data_chamber = data['chamber_values']
    data_background = data['background_values']
    im = data['im']
    M,N = im.shape
    n = M*N # Number of image pixels

    
    # calculate means and stadard deviations
    m_chamber = np.mean(data_chamber)
    s_chamber = np.std(data_chamber)
    m_background = np.mean(data_background)
    s_background = np.std(data_background)
    print("Esimated Mean Chamber: ", m_chamber)
    print("Esimated SD Chamber: ", s_chamber)
    print("Esimated Mean Background: ", m_background)
    print("Esimated SD Background: ", m_background)
   
    # setup edge structure and regularization terms     
    Ie,Je = edges4connected(M,N)
    # Ie,Je = edges8connected(M,N)
    
    # Decide how important a short curve length is:    
    lam = 1
    # what should the weights on the edges be?
    Ve = lam * np.ones_like(Ie)
    
    # setup data terms to source s and sink t:
    Is1 = np.arange(n)
    Js1 = (n)*np.ones((n,))
    Is2 = (n)*np.ones((n,))
    Js2 = np.arange(n)
    
    # Computing negative log-likelihoods for each pixel based on Gaussian distributions
    Vs = (im.flatten() - m_chamber) ** 2 / (2 * s_chamber ** 2)
    Vt = (im.flatten() - m_background) ** 2 / (2 * s_background ** 2)
    
    It1 = np.arange(n)
    Jt1 = (n+1)*np.ones((n,))
    It2 = (n+1)*np.ones((n,))
    Jt2 = np.arange(n)
    
    # setup graph, discretize since maxflow algorithm only works on int
    I = np.hstack((Ie,Is1,Is2,It1,It2)).astype(np.int32)
    J = np.hstack((Je,Js1,Js2,Jt1,Jt2)).astype(np.int32)
    V = np.hstack((Ve,Vs,Vs,Vt,Vt))
    
    # we have real values but multiply with large number and round
    sf = 1000
    V = np.round(V*sf).astype(np.int32) 

    # setup sparse graph matrix
    F = sparse.coo_array((V,(I,J)),shape=(n+2,n+2)).tocsr()
  
    # find flow and segmentation
    mf = maximum_flow(F, n, n+1)

    seg = mf.flow
    # flow to sink (node n+1)
    imflow = seg[0:n,n+1].reshape((M,N)).toarray().astype(float)
    imseg = imflow<(V[-n:].astype(float).reshape(M,N)) # segmentation based on flow
   
    plt.imshow(np.hstack((im,imseg))) # show segmentation
    plt.show()
    
    
    
    
    