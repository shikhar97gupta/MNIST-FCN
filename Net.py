# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:22:58 2021

@author: shikh
"""

import numpy as np

class Network:
    
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        
    def add_layers(self,m,n):
        layer = np.zeros((m,n)) 
        self.layers.append(layer)
        
    def init_variables(self,n,l,vtype):
        
        if(vtype=="weight"):
            np.random.seed(20)
            variables = np.random.normal(loc=0.0, scale = np.sqrt(2/(n+l)),size = (n,l))
            self.weights.append(variables)
            
        elif(vtype=="bias"):
            variables = np.zeros((n,l))
            self.biases.append(variables)

class Forward_Feed:
    
    def relu(self,X):
        X[X<0]=0
        
    def softmax(self,X):
        _X = np.zeros_like(X)
        for index in range(X.shape[0]):
            #print("value: ", np.sum(np.e**(X[index])))
            _X[index] = np.e**(X[index])/np.sum(np.e**(X[index]))
        return _X
    
    def cross_entropy(self,layers,labels):
        loss = -np.sum(np.multiply(labels,np.log(layers[-1])),axis=1)
        return loss
    
    def feed_forward(self,layers,weights,biases):
        
        for index in range(len(weights)):
            
            layers[index+1] = np.dot(layers[index],weights[index])+biases[index]
            
            if(index!=len(weights)-1):
                self.relu(layers[index+1])
            else:
                layers[-1] = self.softmax(layers[index+1])

class Back_Propagate:
    
    def relu_derivative(self,X):
        
        _X = X[:]
        _X = _X>0
        
        return _X
    
    def cross_entropy_softmax_derivative(self,layers,labels):
        
        return (layers[-1]-labels)
    
    def prop_back(self,layers,weights,biases,labels,lr):
            
        dC  = self.cross_entropy_softmax_derivative(layers,labels)
        dw = np.dot(layers[-3].T,dC)
        db = np.sum(dC,axis=0)
        dl = np.dot(dC,weights[-1].T)
        
        weights[-1] = weights[-1] - lr*dw/layers[-1].shape[0]
        biases[-1] = biases[-1] - lr*db/layers[-1].shape[0]
        
        for index in range(len(weights)-2,-1,-1):
            
            dC = np.multiply(self.relu_derivative(layers[index+1]),dl)
            dw = np.dot(layers[index].T,dC)
            db = np.sum(dC,axis=0)
            dl = np.dot(dC,weights[index].T)
            
            weights[index] = weights[index] - lr*dw/layers[index+1].shape[0]
            biases[index] = biases[index] - lr*db/layers[index+1].shape[0]