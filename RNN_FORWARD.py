# -*- coding: utf-8 -*-
"""
File: RNN_FORWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/27/19
Description: mplement the forward propagation of the recurrent neural network, containing every time step
"""
from scipy.special import softmax
import numpy as np
from RNN_CELL_FORWARD import rnn_cell_forward

def rnn_forward(x, a0, parameters):
    '''
    :param x: Input data for every time step, of shape (n_x, m, T_x)
    :param a0: Initial hidden state, of shape (n_a, m)
    :param parameters:
    :return:
    a: Hidden states for every time-step, of shape (n_a, m, T_x)
    y_pred: redictions for every time-step, of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    '''
    # nitialize "caches" which will contain the list of caches at all time steps
    caches = []

    # Retrieve dimensions from shapes of x and Wy
    n_x, m, T_x = x.shape   # n_x: dimension of feature of each time step
                            # m: batch_size
                            # T_x: length of each sequence
    n_y, n_a = parameters["Wya"].shape

    # Initialize "a" and "y"
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)
    return a, y_pred, caches

if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(3, 10, 4)
    a0 = np.random.randn(5, 10)
    Waa = np.random.randn(5, 5)
    Wax = np.random.randn(5, 3)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a, y_pred, caches = rnn_forward(x, a0, parameters)
    print("a[4][1] = ", a[4][1])
    print("a.shape = ", a.shape)
    print("y_pred[1][3] =", y_pred[1][3]) # y_pred[1][3] means the 1st dimension feature, 3rd sample, all time steps
    print("y_pred.shape = ", y_pred.shape)  # y_pred shape is [dimension of y, batch_size, length of time step]
    print("caches[1][1][3] =", caches[1][1][3]) #[1] for x, [1] for 1st feature, [3] for 3rd sample
    print("len(caches) = ", len(caches))

    # Expected output:
    '''
    a[4][1] =  [-0.99999375  0.77911235 -0.99861469 -0.99833267]
    a.shape =  (5, 10, 4)
    y_pred[1][3] = [ 0.79560373  0.86224861  0.11118257  0.81515947]
    y_pred.shape =  (2, 10, 4)
    caches[1][1][3] = [-1.1425182  -0.34934272 -0.20889423  0.58662319]
    len(caches) =  2
    '''