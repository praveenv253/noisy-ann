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
    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'iter'])
    args = params.args

    data = load_mnist_data(params)
    if args.rotate is not None:
        data.trainset = rotate_images(data.trainset, args.rotate, random=True)
    setup_dataloaders(data, params)

    # Initialize the network and train
    net = params.Net(params, noisy=args.noisy)
    if args.noisy:
        net.load_state_dict(torch.load(params.vert_model_filename()))
        net.freeze_layers()
        if args.reinit:
            net.post_noise_reinit()
    train(net, data, params)
    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), params.model_filename())
