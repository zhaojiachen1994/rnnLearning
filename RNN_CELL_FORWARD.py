# -*- coding: utf-8 -*-
"""
File: RNN_CELL_FORWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/27/19
Description: rnn_cell 是只考虑一个时间步（x_t, a_prev）->(y_t, a_next)
"""

from scipy.special import softmax
import numpy as np
def rnn_cell_forward(xt, a_prev, parameters):
    '''
    :param xt: input data at time t, [n_x, m], m is the batch_size, vectorize over m examples.
    :param a_prev: hidden state at time t, [n_a, m]
    :param parameters: Wax [n_a, n_x]; Waa [n_a, n_a]; Wya [n_y, n_a]; ba [n_a, 1]; by [n_y, 1]
    :return:
    a_next: next hidden state, of shape [n_a, m]
    yt_pred: prediction at timestep t, of shape [n_y, m]
    cache: tuple of values for the backward pass, contains [a_next, a_prev, xt, parameters]
    '''

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    # cache is the storage for backward propagation
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache

if __name__ == "__main__":
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    Waa = np.random.randn(5, 5)
    Wax = np.random.randn(5, 3)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    print("a_next[4] = ", a_next[4])    # a_next[4] means the 4th dimension feature for all 10 samples.
    print("a_next.shape = ", a_next.shape)
    print("yt_pred[1] =", yt_pred[1])
    print("yt_pred.shape = ", yt_pred.shape)

    # Expected output:
    """
    a_next[4] = [0.59584544  0.18141802  0.61311866  0.99808218  0.85016201  0.99980978
                 - 0.18887155  0.99815551  0.6531151   0.82872037]
    a_next.shape = (5, 10)
    yt_pred[1] = [0.9888161   0.01682021  0.21140899  0.36817467  0.98988387  0.88945212
                  0.36920224  0.9966312   0.9982559   0.17746526]
    yt_pred.shape = (2, 10)
    """


