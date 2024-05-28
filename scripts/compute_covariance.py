#!/usr/bin/env python3

import argparse

import numpy as np
import numpy.linalg as la
import torch

import models
from param_utils import init_params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--covrot', type=float, default=60.0,
                        help='Rotation angle to use for computing covariance')
    args = parser.parse_args()

    params = init_params()

    # Load and rotate training dataset
    data = load_mnist_data(params)
    data.trainset = rotate_images(data.trainset, args.covrot, random=True)
    setup_dataloaders(data, params, traineval=True)

    # Load the network
    net = models.Mnist_6L_CNN()
    loadfile_path = '../saved-models/mnist-6l-cnn--vert.pth'
    net.load_state_dict(torch.load(loadfile_path))

    # Compute covariance matrix on forward pass
    outs = []
    images, labels = next(iter(data.trainloader))
    with torch.no_grad():
        net(images.reshape((-1, 1, 28, 28)))  # Forward pass
        outs.append(net.outputs[-3].numpy())
    outs = np.array(outs).squeeze()
    cov = np.cov(outs.T)

    print(cov.shape)

    # Check basic statistics of the covariance matrix
    lamda = la.eigvalsh(cov)
    print('Eigenvalues of covariance matrix:')
    print(np.sort(lamda))

    # Save the covariance matrix
    savefile_name = '../saved-models/cov--rot-%.2f' % args.covrot
    np.save(savefile_name, cov)
