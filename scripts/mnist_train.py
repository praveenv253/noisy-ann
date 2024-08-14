#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from param_utils import Params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


def train(net, data, params):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.adam_lr)

    # Training loop
    loss_vals = []
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
                loss_vals.append(running_loss)
                running_loss = 0.0

    #plt.plot(loss_vals)
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate', type=int, default=None,
                        help='Rotation angle in degrees to apply to all training data')
    parser.add_argument('--noisy', nargs='?', const=True, default=False,
        help=('Instantiate the noiseless model if false. If true, instantiate '
              'the noisy model and load the covariance matrix computed using the '
              'rotation given by `covrot`. If zero, use the noisy model but add '
              'no noise (i.e., to train only post-noise layers). If diagonal, use '
              'only the diagonal of the covariance matrix. If identity, use an '
              'identity covariance matrix.'))
    parser.add_argument('--covrot', type=int, default=60,
                        help=('Rotation angle used to compute the covariance matrix '
                              'for adding noise while training.'))
    parser.add_argument('--iter', type=int, default=0,
                        help='Iteration number for multiple runs')
    args = parser.parse_args()

    params = Params(args)

    data = load_mnist_data(params)
    if args.rotate is not None:
        data.trainset = rotate_images(data.trainset, args.rotate, random=True)
    setup_dataloaders(data, params)

    # Initialize the network and train
    if args.noisy:
        oldnet = params.Net(params)
        oldnet.load_state_dict(torch.load(params.vert_model_filename()))
        if args.noisy == 'identity':
            cov = np.eye(params.NoisyNet.noise_dim)
        elif args.noisy == 'zero':
            cov = 0 * np.eye(params.NoisyNet.noise_dim)
        else:
            cov = np.load(params.cov_filename())
        if args.noisy == 'diagonal':
            cov *= np.eye(cov.shape[0])
        net = params.NoisyNet(oldnet, cov)
    else:
        net = params.Net(params)
    train(net, data, params)
    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), params.model_filename())
