#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn import neighbors 



if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = './'
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')
    X = data['X'].transpose()
    Y = data['Y'].transpose().flatten()
    nbr_examples = np.size(Y,0)
    
    # This outer loop will run 100 times, so that you get a mean error for your
    # classifier (the error will become different each time due to the
    # randomness of train_test_split, which you may verify if you wish).
    nbr_trials = 100
    err_rates_test = np.zeros((nbr_trials,3));
    err_rates_train = np.zeros((nbr_trials,3));
    for i in range(nbr_trials):
        
        # First split data into training / testing (80% train, 20% test)
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
        nbr_train_examples = np.size(Y_train,0)
        nbr_test_examples = np.size(Y_test,0)
        
        
        
        tree_model = tree.DecisionTreeClassifier()
        tree_model = tree_model.fit(X_train,Y_train)
        
        
        # FILL IN COMMANDS FOR A LINEAR SVM USING THE SVC CLASSIFIER
        # svm_model = ...
        
        
        # FILL IN COMMANDS FOR A NEAREST NEIGHBOR CLASSIFIER
        # USING THE KNeighborsClassifier
        # nn_model = ...
        
        

    
        # Next, let's use our trained model to classify the examples in the 
        # test data. 
        predictions_test_tree  = tree_model.predict(X_test)
        
        # FILL IN SIMILAR FOR THE OTHER MODELS
        # predictions_test_svm  = ...
        # predictions_test_nn  = ...
        
        
       
        # We can now proceed to computing the respective error rates.
        pred_test_diff_tree = predictions_test_tree - Y_test
        
        # FILL IN SIMILAR FOR THE OTHER MODELS
        # pred_test_diff_svm = ...
        # pred_test_diff_nn = ...
        
        err_rate_test_tree =  np.count_nonzero(pred_test_diff_tree) / nbr_test_examples 
        
        # FILL IN SIMILAR FOR THE OTHER MODELS
        # err_rate_test_svm =  
        # err_rate_test_nn =  
        
        
        # Store them in the containers
        err_rates_test[i,0] = err_rate_test_tree
        
        # FILL IN SIMILAR FOR THE OTHER MODELS
        # err_rates_test[i,1] = ...
        # err_rates_test[i,2] = ...
            
        
        # Next, let's do the same for the train data
        
        predictions_train_tree  = tree_model.predict(X_train)
        # FILL IN REST SIMILAR TO THE TEST DATA ... 
        
        
       
        
        
        
    

    print(np.mean(err_rates_test,0))
    print(np.mean(err_rates_train,0))
    
        



