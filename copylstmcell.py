# -*- coding: utf-8 -*-
"""
File: copylstmcell.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 11/1/19
Description:
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.modules.rnn as rnn
import torchvision.transforms as transforms
from torch.nn import Parameter
import math
import torch.nn.functional as F
import argparse

class LSTMCell_mine(rnn.RNNCellBase):
    """
    math:
        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    Inputs: input, (h_0, c_0)
        - **input** (batch, input_size): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.
    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch
    Attributes:
        weight_ih: the learnable input-hidden weights, of shape [4*hidden_size, input_size]
        weight_hh: the learnable hidden-hidden weights of shape [4*hidden_size, hidden_size]
        bias_ih: of shape (4*hidden_size)
        bias_hh: of shape (4*hidden_size)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_mine, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def LSTMCell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        if input.is_cuda:
            igates = F.linear(input, w_ih)
            hgates = F.linear(hidden[0], w_hh)
            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy) #

        return hy, cy

    def forward(self, input, hx):
        return self.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.lstmcell1 = LSTMCell_mine(input_size, hidden_size, bias=True)
        self.lstmcell2 = LSTMCell_mine(hidden_size,hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        '''
        :param input: with shape [batch_size, sequence_length, input_size]
        :return:
        '''
        h_t = torch.zeros(input.size(0), self.hidden_size)
        c_t = torch.zeros(input.size(0), self.hidden_size)
        h_t2 = torch.zeros(input.size(0), self.hidden_size)
        c_t2 = torch.zeros(input.size(0), self.hidden_size)

        for step, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstmcell1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
        output = self.linear(h_t2)  # shape with [batch_size, num_classes]
        return output

    def predict(self, image, label_true):
        self.eval()
        # print(image.size())
        # image = image.reshape(config.sequence_length, config.input_size).to(device)
        # print('image size', image.size())
        label_true = label_true.to(device)
        out = self.forward(image)
        _, label_pred = torch.max(out.data, 1, keepdim=False)
        result = (label_pred == label_true).item()
        label_pred = label_pred.item()
        return label_pred, result




def trainLSTM(model, train_loader, config):

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_step = len(train_loader)
    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, config.sequence_length, config.input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # BACKWARD AND OPTIMIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.num_epochs, i + 1, total_step, loss.item()))

def testLSTM(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.reshape(-1, config.sequence_length, config.input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def mnistdataloader():
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)    #共有60000张图片

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)
    return train_loader, test_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    train_loader, test_loader = mnistdataloader()
    model = LSTM(input_size=28, hidden_size=128, output_size=10)
    testLSTM(model, test_loader)
    trainLSTM(model, train_loader, config)
    testLSTM(model, test_loader)

def main2():
    #### STEP1, LOAD THE MODEL
    train_loader, test_loader = mnistdataloader()
    model = LSTM(input_size=28, hidden_size=128, output_size=10)

    ind = 45 # ind is indexed in the first batch, so should be in [0, 99]
    sample = next(iter(test_loader))
    x = sample[0][ind]
    y_true = sample[1][ind]
    y_pred, result = model.predict(x, y_true)
    print('Ground-truth label:', y_true.item(), 'Predicted label:', y_pred, '\tResult:', result)

if __name__ == "__main__":
    print('START lstmcellClassify!!!\n---------------------------\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='PyTorch lstm text')
    parser.add_argument('--sequence_length', type=int, default=28)
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--pathCheckpoint', type=str, default = './checkpoint/lstmcellclassify.ckpt')
    config = parser.parse_args()
    setup_seed(1)
    main()

"""

"""