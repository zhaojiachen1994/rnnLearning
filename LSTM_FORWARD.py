# -*- coding: utf-8 -*-
"""
File: LSTM_FORWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/27/19
Description:
"""

import numpy as np
from LSTM_CELL_FORWARD import lstm_cell_forward

def lstm_forward(x, a0, parameters):
    '''
    :param x: Input data for every time step, [n_x, m, T_x]
    :param a0: Initial hidden state, [n_a, m]
    :param parameters:
    :return:
            a: Hidden states for every time step, [n_a, m, T_x]
            y: Predictions for every time step, [n_y, m, T_x]
    Note: the initial cell state and hidden state is same
    '''

    caches = []
    # Retrieve dimensions from shapes of x and Wy
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    a = np.zeros((n_a, m, T_x)) # hidden states for all time steps
    c = a   # cell stats for all time steps
    y = np.zeros((n_y, m, T_x))
    a_next = a0
    c_next = np.zeros(a_next.shape)

    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:, :, t] = yt
        # Save the value of the next cell state (≈1 line)
        c[:, :, t] = c_next
        # Append the cache into caches (≈1 line)
        caches.append(cache)
    caches = (caches, x)

    return a, y, c, caches


if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
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

    a, y, c, caches = lstm_forward(x, a0, parameters)
    print("a[4][3][6] = ", a[4][3][6])  # all hidden states, 4th dimension feature, 3rd sample, 6th time step
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])   # 1st dimension feature, 4th sample, 3rd time step
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1]) # all memory state, 1st dimension, 2nd sample, 1st time step
    print("len(caches) = ", len(caches))