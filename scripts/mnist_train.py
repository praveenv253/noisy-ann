#!/usr/bin/env python3

import argparse
import numpy as np
import torch

import models
from param_utils import init_params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


def train(net, data, params):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.adam_lr)

    # Training loop
    for epoch in range(params.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, train_data in enumerate(data.trainloader, 0):
            inputs, labels = train_data
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs.reshape((-1, 1, 28, 28)))
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print('[%d, %d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate', type=float, default=None)
    parser.add_argument('--noisy', nargs='?', const=True, default=False)
    parser.add_argument('--covrot', type=float, default=60.0)
    args = parser.parse_args()

    params = init_params()

    data = load_mnist_data(params)
    if args.rotate is not None:
        data.trainset = rotate_images(data.trainset, args.rotate, random=True)
    setup_dataloaders(data, params)

    # Initialize the network and train
    if args.noisy:
        oldnet = models.Mnist_6L_CNN()
        loadfile_path = '../saved-models/mnist-6l-cnn--vert.pth'
        oldnet.load_state_dict(torch.load(loadfile_path))
        if args.noisy == 'identity':
            cov = np.eye(32)
        else:
            cov = np.load('../saved-models/cov--rot-%.2f.npy' % args.covrot)
        if args.noisy == 'diagonal':
            cov *= np.eye(32)
        net = models.Noisy_Mnist_6L_CNN(oldnet, cov)
    else:
        net = models.Mnist_6L_CNN()
    train(net, data, params)
    print('Finished Training')

    # Save the trained model
    savefile_name = 'mnist-6l-cnn'
    if args.noisy:
        savefile_name += '--noisy' + ('-' + args.noisy if args.noisy is not True else '')
        savefile_name += '--covrot-%.2f' % args.covrot
    else:
        savefile_name += ('--vert' if args.rotate is None else
                          '--rot-%.2f' % args.rotate)
    savefile_name += '.pth'
    torch.save(net.state_dict(), '../saved-models/%s' % savefile_name)
