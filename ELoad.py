# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:37:23 2021

@author: shikh
"""

from Net import Network
from Run import predict

def load_model(batch_size, weights, biases, nlayers=[784,200,10]):
    
    in_size= nlayers[0]
    hidden_size = nlayers[1]
    class_size = nlayers[2]
    
    net = Network()
    net.add_layers(batch_size,in_size)
    net.add_layers(batch_size,hidden_size)
    net.add_layers(batch_size,class_size)
    net.add_layers(batch_size,class_size)

    #net.layers[0] = flat_img
    net.weights = weights
    net.biases = biases
    
    return net

def exe_pred(idx, flat_img, net):
     
    net.layers[0] = flat_img
    model_pred = predict(net.layers, net.weights, net.biases)
    return model_pred