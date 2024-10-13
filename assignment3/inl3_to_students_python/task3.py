#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding = 'same')
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(8, 16, 3,padding = 'same')
        self.conv3 = nn.Conv2d(16, 32, 3,padding = 'same')
        self.fc1 = nn.Linear(4*4*32, 2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.flatten(x,1)  
        x = self.fc1(x)
        x = F.softmax(x,dim = -1)
        return x

def splitData(X,Y,split = 0.8):
    nd = X.size(0)
    ids = torch.randperm(nd)
    ts = int(split*nd)
    X_train = X[ids[:ts],:,:,:]
    Y_train = Y[ids[:ts]]
    X_test = X[ids[ts:],:,:,:]
    Y_test = Y[ids[ts:]]
    return (X_train,X_test,Y_train,Y_test)

def trainSimpleCNN(X,Y):
    net = SimpleCNN()   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 
    
    nrepochs = 100
    for epoch in range(nrepochs):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    
        # print statistics
        # print(loss.item())

    print('Finished Training')
    return net

def predictSimpleCNN(net,X):
    with torch.no_grad():
        output = net(X)
    _, Y = torch.max(output, 1)
    return Y

if __name__ == "__main__":
    # load data, change datadir path if your data is elsewhere
    datadir = './'
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')

    X = torch.tensor(data['X'].astype(int)).float()
    X = X.reshape(1,19,19,200)
    X = X.movedim((0,1,2,3),(1,2,3,0))
    Y = torch.tensor(data['Y'].astype(int))
    Y = (Y+1)//2 # change labels from (-1,1) to (0,1)
    Y = Y.flatten()

    # Initialize error arrays
    err_rates_train = []
    err_rates_test = []

    # Number of trials
    nbr_trials = 100

    for i in range(nbr_trials):
        # Split the data
        X_train, X_test, Y_train, Y_test = splitData(X,Y,0.9)

        # Train the model
        net = trainSimpleCNN(X_train,Y_train)

        # Make predictions
        Y_train_pred = predictSimpleCNN(net, X_train)
        Y_test_pred = predictSimpleCNN(net, X_test)

        # Calculate training error rate
        err_train = torch.mean((Y_train_pred != Y_train).float()).item()
        err_rates_train.append(err_train)

        # Calculate testing error rate
        err_test = torch.mean((Y_test_pred != Y_test).float()).item()
        err_rates_test.append(err_test)

    # Calculate the average error rates
    avg_err_train = sum(err_rates_train) / nbr_trials
    avg_err_test = sum(err_rates_test) / nbr_trials

    # Print the results
    print("Average training error rate: ", avg_err_train)
    print("Average testing error rate: ", avg_err_test)