#!/usr/bin/env python3

import datetime as dt

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from param_utils import Params
from data_utils import MnistData


def train(net, data, params, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.adam_lr)

    # Training loop
    loss_vals = []
    for epoch in range(params.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, train_data in enumerate(data.loader):
            inputs, labels = train_data
            inputs, labels = inputs.to(device), labels.to(device)

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'iter'])
    args = params.args

    if args.rotate is not None:
        data = MnistData(params, rotation_angle=args.rotate, random=True)
    else:
        data = MnistData(params)

    # Initialize the network and train
    net = params.Net(params, noisy=args.noisy, device=device)
    if args.noisy:
        net.load_state_dict(torch.load(params.vert_model_filename()))
        net.freeze_layers()
        if args.reinit:
            net.post_noise_reinit()

    net.to(device)

    print(dt.datetime.now())
    train(net, data, params, device)
    print('Finished Training')
    print(dt.datetime.now())

    # Save the trained model
    torch.save(net.state_dict(), params.model_filename())
