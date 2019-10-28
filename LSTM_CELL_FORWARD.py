# -*- coding: utf-8 -*-
"""
File: LSTM_CELL_FORWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/27/19
Description: Implement a single forward step of the LSTM-cell
"""

from scipy.special import softmax
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    '''
    :param xt: input data at time t, [n_x, m]
    :param a_prev: Hidden state at time t-1, [n_a, m]
    :param c_prev: Memory state (cell state) at time t-1, [n_a, m]
    ! Here hidden state and cell state have the same dimension
    :param parameters:
                        Wf: Weight matrix of forget gate, [n_a, n_a+n_x]
                        bf: Bias of forget gate, [n_a, 1]
                        Wi: Weight matrix of update gate, [n_a, n_a+n_x]
                        bi: Bias of update gate, [n_a, 1]
                        Wc: Weight matrix from [a_<t-1>, x_t] to candidate cell state value c_tilde
                        bc: Bias corresponding to Wc
                        Wo: Weight matrix of output gate [n_a, n_a+n_x]
                        bo: Bias of output gate [n_a, 1]
                        Wy: Weight matrix from hidden state to output [n_y, n_a]
                        by: Bias corresponding to Wy
    :return:
            a_next: Next hidden state [n_a, m]
            c_next: Next memory state [n_a, m]
            yt_pred: prediction at time step t, [n_y, m]
            cache: tuple of values needed for the backward pass,
                contains (a_next, c_next, a_prev, c_prev, xt, parameters)
            c_prev: Memory state (cell state) at time t-1, [n_a, m]
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    '''

    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and x_t
    concat = np.concatenate([a_prev, xt], axis=0)

    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft*c_prev + it*cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot*np.tanh(c_next)
    yt_pred = softmax(np.dot(Wy, a_next) + by, axis=0)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache



if __name__ == "__main__":
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    c_prev = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
    print("a_next[4] = ", a_next[4])
    print("a_next.shape = ", c_next.shape)
    print("c_next[2] = ", c_next[2])
    print("c_next.shape = ", c_next.shape)
    print("yt[1] =", yt[1])
    print("yt.shape = ", yt.shape)
    print("cache[1][3] =", cache[1][3])
    print("len(cache) = ", len(cache))

    # Expected output:
    '''
    a_next[4] = [-0.66408471  0.0036921   0.02088357  0.22834167 - 0.85575339  0.00138482
                 0.76566531  0.34631421 - 0.00215674  0.43827275]
    a_next.shape = (5, 10)
    c_next[2] = [0.63267805  1.00570849  0.35504474  0.20690913 - 1.64566718  0.11832942
                 0.76449811 - 0.0981561 - 0.74348425 - 0.26810932]
    c_next.shape = (5, 10)
    yt[1] = [0.79913913  0.15986619  0.22412122  0.15606108  0.97057211  0.31146381
             0.00943007  0.12666353  0.39380172  0.07828381]
    yt.shape = (2, 10)
    cache[1][3] = [-0.16263996  1.03729328  0.72938082 - 0.54101719  0.02752074 - 0.30821874
                   0.07651101 - 1.03752894  1.41219977 - 0.37647422]
    len(cache) = 10    
    '''
