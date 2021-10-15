# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:50:21 2021

@author: shikh
"""

import numpy as np
import pandas as pd

from Run import run
from ELoad import load_model, exe_pred
from Utils import one_hot

X = pd.read_csv("mnist/mnist_train.csv")
np.random.seed(20)
np.random.shuffle(np.asarray(X))
Y = X["label"]
X = X.drop("label",axis=1)
X = np.asarray(X)
Y = np.asarray(Y)

_X = pd.read_csv("mnist/mnist_test.csv")
_Y = _X["label"]
_X = _X.drop(["label"],axis=1)
_X = np.asarray(_X)
_Y = np.asarray(_Y)

#global in_size,hidden1_size,hidden2_size,class_size
#in_size= X.shape[1]
#hidden1_size = 200
class_size = 10
lr = 0.1
epochs = 6
idx = 0
flat_img = _X[idx]/255 
label = _Y[idx]
batch_size = 1

Y_enc = one_hot(Y,class_size)
_Y_enc = one_hot(_Y,class_size)

batchset = [X, Y_enc, _X, _Y_enc]

weights, biases = run(lr, epochs, 32, batchset)

net = load_model(batch_size, weights, biases)

pred = exe_pred(0, flat_img, net)
print("prediction: ", pred)