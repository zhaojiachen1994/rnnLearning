# -*- coding: utf-8 -*-
"""
File: copyrnncell.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 11/1/19
Description: 从pytorch中继承torch.nn.modules.rnn中的RNNCellBase类，重新实现了torch中的rnncell类，并且以此
构造了一个两层的循环神经网络，完成mnist数据集的识别任务，结果rnn的识别精度与lstm差不多，可能是序列长度不长。
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
import argparse #Parser for command-line options, argyments and sub-commands

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
class RNNCell_mine(rnn.RNNCellBase):
    '''
    Copy the RNNCell from torch.nn.modules.rnn package
    JUST COPY, NOT MODIFY
    https://github.com/pytorch/pytorch/blob/v0.3.0/torch/nn/modules/rnn.py#L538
    '''
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell_mine, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            # print(weight)

    def forward(self, input, hx):
        if self.nonlinearity == "tanh":
            func = self.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = self.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
    def RNNReLUCell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
        return hy


    def RNNTanhCell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = torch.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
        return hy


class RNN(nn.Module):
    '''
    A model class using RNNCell_mine
    '''
    def __init__(self, config, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.rnncell1 = RNNCell_mine(input_size= self.input_size,
                                     hidden_size= self.hidden_size,
                                     bias=bias, nonlinearity=nonlinearity)
        self.rnncell2 = RNNCell_mine(input_size= self.hidden_size,
                                     hidden_size= self.hidden_size,
                                     bias=bias, nonlinearity=nonlinearity)
        self.linear = nn.Linear(self.hidden_size, config.num_classes)

    def forward(self, input):
        '''
        :param input: with shape[batch_size, sequence_length, input_size]
        :return:
        '''
        h_t1 = torch.zeros(input.size(0), self.hidden_size)
        h_t2 = torch.zeros(input.size(0), self.hidden_size)

        for step, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t1 = self.rnncell1(input_t.squeeze(dim=1), h_t1)
            h_t2 = self.rnncell2(h_t1, h_t2)
        output = self.linear(h_t2)
        return output


def trainRNN(model, train_loader, config):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_step = len(train_loader)
    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, images.size(2), images.size(3)).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # BACKWARD AND OPTIMIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100 ==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.num_epochs, i + 1, total_step, loss.item()))


def testRNN(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.reshape(-1, images.size(2), images.size(3)).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


def main():
    train_loader, test_loader = mnistdataloader()
    model = RNN(config, config.input_size, config.hidden_size, bias=True, nonlinearity="tanh")
    testRNN(model, test_loader)
    trainRNN(model, train_loader, config)
    testRNN(model, test_loader)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='PyTorch Copylstemcell')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)
    config = parser.parse_args()
    setup_seed(1)
    main()


"""
Expected output:
With random seed = 1, hidden_size=128, learning_rate=0.001, num_epochs=3

without training:   7.54%
with tanh activation:   96.04%
with relu activation: Test Accuracy of the model on the 10000 test images: 96.06 %
"""
