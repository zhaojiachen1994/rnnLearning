# -*- coding: utf-8 -*-
"""
File: s1_lstmclassification.py
Project: lstmsegv1
Author: Jiachen Zhao
Date: 10/29/19
Description: perform classification task for MNIST data set using a two-layer lstm
Reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, image, label_true):
        self.eval()
        image = image.reshape(-1, config.sequence_length, config.input_size).to(device)
        label_true = label_true.to(device)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, image.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, image.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(image, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        _, label_pred = torch.max(out.data, 1, keepdim=False)
        result = (label_pred==label_true).item()
        label_pred = label_pred.item()
        return label_pred, result



def trainRNN(model, train_loader, config):
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model
    model.train()
    total_step = len(train_loader)
    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, config.sequence_length, config.input_size).to(device)
            labels = labels.to(device)
            # Forward pass
            # print(images.size())
            outputs = model(images) # images shape is [100, 28, 28], corresponding to [batch_size, sequence_length, input_size]
            loss = model.criterion(outputs, labels)
            # BACKWARD AND OPTIMIZE
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if (i+1)%100 ==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.num_epochs, i + 1, total_step, loss.item()))

def testRNN(model, test_loader, config):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
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

def loadRNN(config, mode='eval'):

    # First initialize the model and optimizer (here, the optimizer is contained in the model)
    model = RNN(config)
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
    model = RNN(config)
    trainRNN(model, train_loader, config)
    testRNN(model, test_loader, config)
    saveRNN(model, config)

def main2(config):
    """
    Load the model; test it; see the parameters;
    """
    #### STEP1, LOAD THE MODEL
    train_loader, test_loader = mnistdataloader()
    model = loadRNN(config, mode='eval')

    #### STEP2, TEST THE MODEL
    # testRNN(model, test_loader, config)

    #### STEP3, SEE THE PARAMETERS AND SHAPE
    lstm = model.lstm
    print(lstm.named_parameters())
    for name, param in lstm.named_parameters():
        if param.requires_grad:
            print(name, '\t', param.size())

    ### STEP4, SEE THE PREDICTED RESULTS.
    # ind = 50    # ind is indexed in the first batch, so should be in [0, 99]
    # sample = next(iter(test_loader))
    # x = sample[0][ind]
    # y_true = sample[1][ind]
    # y_pred, result = model.predict(x, y_true)
    # print('Ground-truth label:', y_true.item(), 'Predicted label:', y_pred, '\tResult:', result)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='PyTorch lstm text')
    parser.add_argument('--sequence_length', type=int, default=28)
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=28)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--pathCheckpoint', type=str, default = './checkpoint/RNN.ckpt')
    config = parser.parse_args()

    main1(config)
