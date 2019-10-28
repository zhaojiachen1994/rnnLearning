# -*- coding: utf-8 -*-
"""
File: LETM_CELL_BACKWARD.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/28/19
Description:
"""

import numpy as np
from LSTM_CELL_FORWARD import lstm_cell_forward

def lstm_cell_backward(da_next, dc_next, cache):
    '''
    :param da_next: Gradient of
    :param dc_next:
    :param cache:
    :return:
    NOTE: Implement the backward for the LSTM-cell (single time step)
        From the da_next and dc_next to
        dxt, da_prev, dc_prev;
        dwf, dbf; dwi, dbi; dwc, dbc; dwo, dbo;
    '''

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    # cct stand for candidate of cell state at time t
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives
    dot = da_next*np.tanh(c_next)*ot*(1-ot)
    dcct = (dc_next*it + ot*(1-np.square(np.tanh(c_next)))*it*da_next) * (1-np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)
    concat = np.concatenate((a_prev, xt), axis=0)
    # Compute parameters related derivatives.
    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)
    dbf = np.sum(dft, axis=1 ,keepdims = True)
    dbi = np.sum(dit, axis=1, keepdims = True)
    dbc = np.sum(dcct, axis=1,  keepdims = True)
    dbo = np.sum(dot, axis=1, keepdims = True)
# Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + \
              np.dot(parameters['Wi'][:, :n_a].T, dit) + \
              np.dot(parameters['Wc'][:, :n_a].T, dcct) + \
              np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) \
          + np.dot(parameters['Wi'][:, n_a:].T, dit) +\
          np.dot(parameters['Wc'][:, n_a:].T, dcct) + \
          np.dot(parameters['Wo'][:, n_a:].T, dot)
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients

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

    da_next = np.random.randn(5, 10)
    dc_next = np.random.randn(5, 10)
    gradients = lstm_cell_backward(da_next, dc_next, cache)
    print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
    print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
    print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
    print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
    print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
    print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
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