# -*- coding: utf-8 -*-
"""
File: lstmcellClassify.py
Project: rnnLearning
Author: Jiachen Zhao
Date: 10/29/19
Description: Achieve the same function with lstmClassi but with lstmcell
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
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

class RNNCELL(nn.Module):
    def __init__(self, config):
        super(RNNCELL, self).__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.lstmcell1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size, bias=True)
        self.lstmcell2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True)
        self.linear = nn.Linear(self.hidden_size, config.num_classes)

        # self.register_buffer('h_t', torch.zeros(config.batch_size, self.hidden_size))
        # self.register_buffer('c_t', torch.zeros(config.batch_size, self.hidden_size))
        # self.register_buffer('h_t2', torch.zeros(config.batch_size, self.hidden_size))
        # self.register_buffer('c_t2', torch.zeros(config.batch_size, self.hidden_size))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, input):   # input size [batch_size, 28, 28]

        h_t = torch.zeros(input.size(0), self.hidden_size)
        c_t = torch.zeros(input.size(0), self.hidden_size)
        h_t2 = torch.zeros(input.size(0), self.hidden_size)
        c_t2 = torch.zeros(input.size(0), self.hidden_size)

        for step, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstmcell1(input_t.squeeze(dim=1), (h_t, c_t))
            # print('t_step:', step, 'input_t size:', input_t.squeeze(dim=1).size())
            # print('h_t shape:', h_t.size())
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

def trainRNNCELL(model, train_loader, config):
    model.train()
    total_step = len(train_loader)
    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, config.sequence_length, config.input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = model.criterion(outputs, labels)

            # BACKWARD AND OPTIMIZE
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if (i+1)%100 ==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.num_epochs, i + 1, total_step, loss.item()))

def testRNNCELL(model, test_loader, config):
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

def saveRNN(model, config):
    '''
    :param model: A pytorch nn model
    :param config: The parameters needed for the code
    :return:
    '''
    checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict()}
    torch.save(checkpoint, config.pathCheckpoint)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loadRNN(config, mode='eval'):

    # First initialize the model and optimizer (here, the optimizer is contained in the model)
    model = RNNCELL(config)
    # Then load the model
    checkpoint = torch.load(config.pathCheckpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Finally, decide the model mode
    if mode == 'eval':
        model.eval()    # call model.eval() to set dropout and batch normalization layers
    elif mode =='train':
        model.train()   # call model.train() to resume the training
    return model

def main1(config):
    '''
        Directly train and test the model
        '''
    train_loader, test_loader = mnistdataloader()
    model = RNNCELL(config)
    trainRNNCELL(model, train_loader, config)
    testRNNCELL(model, test_loader, config)
    saveRNN(model, config)


def main2(config):
    #### STEP1, LOAD THE MODEL
    train_loader, test_loader = mnistdataloader()
    model = loadRNN(config, mode='eval')

    #### STEP2, TEST THE MODEL
    testRNNCELL(model, test_loader, config)

    #### STEP3, SEE THE PARAMETERS AND SHAPE
    lstm = model.lstmcell1
    for name, param in lstm.named_parameters():
        if param.requires_grad:
            print(name, '\t', param.size()) #在torch.nn.lstmcell中weight_ih包括了4个门的权重


    #### STEP4, SEE THE PREDICTED RESULTS
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
    main1(config)
    main2(config)

    print('\n---------------------------\nEND lstmcellClassify!!!')