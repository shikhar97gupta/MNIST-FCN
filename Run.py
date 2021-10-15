# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:25:29 2021

@author: shikh
"""

import numpy as np
from Net import Network, Forward_Feed, Back_Propagate
from Utils import mini_batches

def train(batches,layers,weights,biases,batch_labels,lr):
    
    ff = Forward_Feed()
    
    for index in range(len(batches)):
        layers[0] = batches[index]/255
        labels = batch_labels[index]
        
        ff.feed_forward(layers,weights,biases)
        
        bp = Back_Propagate()
        bp.prop_back(layers,weights,biases,labels,lr)
        
        loss = np.mean(ff.cross_entropy(layers,labels))
        print("loss: ", loss)
        
def predict(layers,weights,biases):
    
    ff = Forward_Feed()
    ff.feed_forward(layers,weights,biases)
    
    for i in range(layers[-1].shape[0]):
        _imax = np.argmax(layers[-1][i])
    
    return _imax

def run(lr, epochs, batch_size, batchset, nlayers=[784,200,10]):
    
    X, Y_enc, _X, _Y_enc = batchset
    
    in_size= nlayers[0]
    hidden_size = nlayers[1]
    class_size = nlayers[2]
    
    batches,batch_labels = mini_batches(X,Y_enc,batch_size=batch_size)
    _batches,_batch_labels = mini_batches(_X,_Y_enc,batch_size=batch_size)

    net = Network()
    net.add_layers(batch_size,in_size)
    net.add_layers(batch_size,hidden_size)

    net.add_layers(batch_size,class_size)
    net.add_layers(batch_size,class_size)
    
    net.weights = []
    net.biases = []
    net.init_variables(in_size,hidden_size,vtype="weight")
    net.init_variables(1,hidden_size,vtype="bias")
    net.init_variables(hidden_size,class_size,vtype="weight")
    net.init_variables(1,class_size,vtype="bias")

    net.layers[0] = batches
    print("Running Train")

    for epoch in range(epochs):
        
        train(batches,net.layers,net.weights,net.biases,batch_labels,lr)
    
    print("Done Training")
    
    return net.weights, net.biases