# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:24:03 2021

@author: shikh
"""

import numpy as np

def mini_batches(X,Y,batch_size):
    
    batches = []
    batch_labels = []
    
    for i in range(0,X.shape[0],batch_size):
        batches.append(X[i:i+batch_size])
        batch_labels.append(Y[i:i+batch_size])
    
    return batches,batch_labels

def one_hot(Y,class_size):
    
    Y_enc = np.zeros((Y.shape[0],class_size))
    for index in range(Y.shape[0]):
        Y_enc[index,Y[index]] = 1
    
    return Y_enc

def standardize(x):
    
    x = x-np.mean(x,axis=1,keepdims=True)/np.std(x,axis=1,keepdims=True)
    return x

def normalize(x):
    
    x = x-np.min(x,axis=1,keepdims=True)/x-np.max(x,axis=1,keepdims=True)
    return x