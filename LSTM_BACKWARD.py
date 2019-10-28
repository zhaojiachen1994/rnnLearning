# -*- coding: utf-8 -*-
"""
File: LSTM_BACKWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/28/19
Description:
"""

import numpy as np
from LETM_CELL_BACKWARD import lstm_cell_backward
from LSTM_FORWARD import lstm_forward

# def lstm_backward(da, caches):
#     '''
#     Implement the backward pass for the LSTM (over a whole sequence)
#     :param da: Gradient of the hidden states over all time step, of shape (n_a, m, T_x)
#     :param caches: cache storing information from the forward pass (lstm_forward)
#     :return:
#         gradients:
#             dx: Gradient of the inputs, of shape (n_x, m, T_x)
#             da0: Gradient of the previous hidden state, (n_a, m), THE HIDDEN STATES ARE DIFFERENT FOR SAMPLES
#             dWf: Gradient of the weight matrix of forget gate, (n_a, n_x+n_a)
#             dWi: Gradient of the weight matrix of input gate, (n_a, n_x+n_a)
#             dWc: Gradient of the weight matrix of memory gate (from [input, a_prev] to candidate cell state), (n_a, n_x+n_a)
#             dWo: Gradient of the weight matrix of forget gate, (n_a, n_x+n_a)
#             dbf: [n_a, 1]
#             dbi: [n_a, 1]
#             dbc: [n_a, 1]
#             dbo: [n_a, 1]
#     '''
#     (caches, x) = caches
#     (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
#
#     # Retrieve dimensions from da's and x1's shapes
#     n_a, m, T_x = da.shape
#     n_x, m = x1.shape
#
#     # initialize the gradients with the right sizes
#     dx = np.zeros((n_x, m, T_x))
#     da0 = np.zeros((n_a, m))
#     da_prevt = np.zeros(da0.shape)
#     dc_prevt = np.zeros(da0.shape)
#     dWf = np.zeros((n_a, n_a + n_x))
#     dWi = np.zeros(dWf.shape)
#     dWc = np.zeros(dWf.shape)
#     dWo = np.zeros(dWf.shape)
#     dbf = np.zeros((n_a, 1))
#     dbi = np.zeros(dbf.shape)
#     dbc = np.zeros(dbf.shape)
#     dbo = np.zeros(dbf.shape)
#
#     for t in reversed(range(T_x)):
#         gradients = lstm_cell_backward(da[:,:,t], dc_prevt, caches[t])
#         # Store or add the gradient to the parameters' previous step's gradient
#         dx[:, :, t] = gradients["dxt"]
#         dWf += gradients["dWf"]
#         dWi += gradients["dWi"]
#         dWc += gradients["dWc"]
#         dWo += gradients["dWo"]
#         dbf += gradients["dbf"]
#         dbi += gradients["dbi"]
#         dbc += gradients["dbc"]
#         dbo += gradients["dbo"]
#         # Set the first activation's gradient to the backpropagated gradient da_prev.
#     da0 = gradients["da_prev"]
#
#     # Store the gradients in a python dictionary
#     gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
#                  "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}
#
#     return gradients


def lstm_backward(da, caches):
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros(da0.shape)
    dc_prevt = np.zeros(da0.shape)
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros(dWf.shape)
    dWc = np.zeros(dWf.shape)
    dWo = np.zeros(dWf.shape)
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros(dbf.shape)
    dbc = np.zeros(dbf.shape)
    dbo = np.zeros(dbf.shape)

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients["da_prev"]

    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

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
    # Wy = np.random.randn(5, 5 + 3)
    # by = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)


    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)

    da = np.random.randn(5, 10, 4)
    gradients = lstm_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)