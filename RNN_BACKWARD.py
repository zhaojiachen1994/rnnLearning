# -*- coding: utf-8 -*-
"""
File: RNN_BACKWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/28/19
Description:
"""

import numpy as np
from RNN_FORWARD import rnn_forward
from RNN_CELL_BACKWARD import rnn_cell_backward

def rnn_backward(da, caches):
    '''
    :param da: Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    :param caches: tuple containing information from the forward pass (rnn_forward)
    :return:Gradient:
            dx: Gradient of the input data, of shape (n_x, m, T_x)
            da0: Gradient of the initial hidden state, (n_a, m)
            dWax: Gradient of the input weight matrix, (n_a, n_x)
            dWaa: Gradient of the hidden state's weight matrix, (n_a, n_x)
            dba: Gradient of the bias, (n_a, 1)
    '''

    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈6 lines)
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        #NOTE: data[:,:,t] 是rnn_backward函数输入的导数，da_prevt是时间倒序循环到此处的导数
        # Retrieve derivatives from gradients (≈ 1 line)
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        # 对于RNN参数的求导结果需要延时间求和
    # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line)
    da0 = da_prevt

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    return gradients

if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(3, 10, 4)
    a0 = np.random.randn(5, 10)
    Wax = np.random.randn(5, 3)
    Waa = np.random.randn(5, 5)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    a, y, caches = rnn_forward(x, a0, parameters)
    da = np.random.randn(5, 10, 4)
    gradients = rnn_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dba\"].shape =", gradients["dba"].shape)